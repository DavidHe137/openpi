import logging
import pathlib
import multiprocessing
import shutil
from typing import List, Literal, Optional, Dict, Type
import datetime
import time

import numpy as np
from jaxtyping import Float
from libero.libero import benchmark
from openpi_client.client import BidirectionalWebsocket
from openpi_client.runtime import runtime as _runtime, subscriber as _subscriber
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client.action_chunkers import ActionChunkBrokerType, BrokerConfig
from openpi_client.schemas import RuntimeMetadata, ServerMetadata
import tyro
from dataclasses import dataclass, field

from examples.libero import utils
from examples.libero import logging_config
from examples.libero.env import LiberoSimEnvironment
from examples.libero.progress_manager import get_progress_manager
from examples.libero.subscribers.saver import Saver
from examples.libero.metrics import calculate_metrics, generate_all_plots
from examples.libero.subscribers.progress_subscriber import ProgressSubscriber

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclass
class Job:
    """A job is a task with a batch of episodes."""

    task_suite_name: str
    task: benchmark.Task
    task_id: int
    initial_states: Float[np.ndarray, "n_initial_states state_dim"]


@dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8080
    resize_size: int = 224
    action_chunk_broker_type: ActionChunkBrokerType = ActionChunkBrokerType.SYNC
    latency_ms: List[float] = field(
        default_factory=list
    )  # Optional per-robot artificial latency (ms); length <= num_robots

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_robot: int = 10  # Number of rollouts per robot per task
    max_steps: int = 2000  # Maximum number of control steps per episode

    #################################################################################################################
    # Multi-robot / threading parameters
    #################################################################################################################
    num_robots: int = 5  # Number of always-running sims (robots)
    control_hz: int = 20  # Target control frequency for each sim #NOTE: int because this is the fps of the video

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # Random Seed (for reproducibility)
    output_dir: str = "data/libero/multi_robot_videos"
    overwrite: bool = False
    progress_type: Literal["verbose", "concise", "logging", None] = "verbose"
    log_dir: Optional[str] = None
    debug: bool = False  # Run in single process with immediate progress output


def _latency_for_robot(args: Args, robot_idx: int) -> float:
    """Return the latency (in ms) to use for a given robot index."""
    if not args.latency_ms:
        return 0.0
    return float(args.latency_ms[robot_idx])


def delay_start(
    control_hz: int,
    server_metadata: ServerMetadata,
):
    """Return the period (in seconds) that a robot waits between requests."""
    period = server_metadata.action_horizon / control_hz
    delay = np.random.uniform(0, period)
    time.sleep(delay)


def init_worker(args: Args, counter, progress_queue) -> None:
    global robot_idx, ws_client, broker, agent, _progress_queue
    with counter.get_lock():
        robot_idx = counter.value
        counter.value += 1

    # Store queue globally for access in create_runtime
    _progress_queue = progress_queue

    ws_client = BidirectionalWebsocket(
        robot_id=f"robot_{robot_idx}",
        host=args.host,
        port=args.port,
    )

    # Create broker config and instantiate
    config = BrokerConfig(
        ws_client=ws_client,
        control_hz=args.control_hz,
    )
    broker = args.action_chunk_broker_type.create(config)
    agent = _policy_agent.PolicyAgent(broker=broker)


def create_runtime(args: Args, job: Job) -> _runtime.Runtime:
    env_raw, task_description = utils._get_libero_env(
        job.task,
        LIBERO_ENV_RESOLUTION,
        seed=args.seed + robot_idx,
    )
    env = LiberoSimEnvironment(
        env=env_raw,
        task_description=task_description,
        initial_states=job.initial_states,
        resize_size=args.resize_size,
        num_steps_wait=args.num_steps_wait,
        max_episode_steps=args.max_steps,
        latency_ms=_latency_for_robot(args, robot_idx),
        control_hz=args.control_hz,
    )

    # Create job info for progress subscriber
    job_info = {
        "task_suite_name": job.task_suite_name,
        "task_id": job.task_id,
        "num_episodes": len(job.initial_states),
    }

    subscribers: List[_subscriber.Subscriber] = [
        Saver(
            out_dir=pathlib.Path(args.output_dir),
            environment=env,
            action_chunk_broker=broker,
            task_suite_name=job.task_suite_name,
            task_id=job.task_id,
            task=job.task,
            robot_idx=robot_idx,
        ),
    ]
    if args.progress_type is not None and _progress_queue is not None:
        subscribers.append(
            ProgressSubscriber(
                queue=_progress_queue,
                robot_idx=robot_idx,
                job_info=job_info,
                environment=env,
                update_frequency=10,
            )
        )

    runtime = _runtime.Runtime(
        environment=env,
        agent=agent,
        subscribers=subscribers,
        max_hz=args.control_hz,
        num_episodes=len(job.initial_states),
        max_episode_steps=env._max_episode_steps,  # type: ignore[attr-defined]
    )
    return runtime


def _robot_worker(task_args) -> None:
    """Worker process that handles jobs for a specific robot index."""
    args, job, server_metadata = task_args
    runtime = create_runtime(args, job)
    delay_start(
        control_hz=args.control_hz,
        server_metadata=server_metadata,
    )

    runtime.run()
    runtime.close()


def run_robots(args: Args, jobs: List[Job], server_metadata: ServerMetadata) -> None:
    counter = multiprocessing.Value("i", 0)  # for assigning robot indices

    if args.debug:
        # Debug mode: no progress manager, single process for pdb compatibility
        init_worker(args, counter, None)
        for job in jobs:
            _robot_worker((args, job, server_metadata))
    else:
        with get_progress_manager(
            args.progress_type, max_steps=args.max_steps
        ) as progress_manager:
            # Pass queue to worker initializer
            with multiprocessing.Pool(
                processes=args.num_robots,
                initializer=init_worker,
                initargs=(args, counter, progress_manager.queue),
            ) as pool:
                try:
                    # use imap_unordered so that it exits immediately on any exception
                    _ = list(
                        pool.imap_unordered(
                            _robot_worker,
                            [(args, job, server_metadata) for job in jobs],
                        )
                    )
                except Exception as e:
                    logging.error(f"Error in robot worker: {e}")
                    raise e
                finally:
                    pool.close()
                    pool.join()


def create_jobs(args: Args) -> List[Job]:
    benchmark_dict: Dict[str, Type[benchmark.Benchmark]] = (
        benchmark.get_benchmark_dict()
    )
    task_suite: benchmark.Benchmark = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    logging.info(
        "Setting up multi-robot LIBERO runtime over suite '%s' with %d tasks, num_robots=%d, trials_per_robot=%d, control_hz=%d",
        args.task_suite_name,
        num_tasks_in_suite,
        args.num_robots,
        args.num_trials_per_robot,
        args.control_hz,
    )

    jobs: List[Job] = []
    for task_id in range(num_tasks_in_suite):
        task: benchmark.Task = task_suite.get_task(task_id)
        all_initial_states: Float[np.ndarray, "n_initial_states state_dim"] = (
            task_suite.get_task_init_states(task_id)
        )

        if len(all_initial_states) < args.num_trials_per_robot:
            logging.error(
                "Task %d has less initial states than trials per robot; skipping",
                task_id,
            )
            continue

        initial_states = all_initial_states[: args.num_trials_per_robot]
        job = Job(
            task=task,
            task_suite_name=args.task_suite_name,
            task_id=task_id,
            initial_states=initial_states,
        )
        jobs.append(job)

    logging.info("Created %d jobs", len(jobs))

    return jobs


def main(args: Args) -> None:
    if args.log_dir is not None:
        log_file_name = f"libero_multi_robot_runtime_{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        log_file_path = pathlib.Path(args.log_dir) / log_file_name
        pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        logging_config.setup_logging(log_path=log_file_path)
    else:
        logging_config.setup_logging()

    if not args.overwrite and pathlib.Path(args.output_dir).exists():
        raise ValueError(f"Output path {args.output_dir} already exists")
    if args.overwrite:
        if pathlib.Path(args.output_dir).exists():
            shutil.rmtree(args.output_dir)
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Validate latency specification
    if args.latency_ms and len(args.latency_ms) != args.num_robots:
        raise ValueError(
            f"latency_ms must either be empty or have exactly {args.num_robots} values "
            f"(one per robot), but got {len(args.latency_ms)} values"
        )

    np.random.seed(args.seed)

    jobs = create_jobs(args)

    # Connect to get server metadata
    temp_client = BidirectionalWebsocket(
        robot_id="robot",
        host=args.host,
        port=args.port,
    )
    server_metadata = temp_client.server_metadata
    temp_client.close()

    # Create runtime metadata
    runtime_metadata = RuntimeMetadata(
        task_suite_name=args.task_suite_name,
        num_steps_wait=args.num_steps_wait,
        num_trials_per_robot=args.num_trials_per_robot,
        max_steps=args.max_steps,
        num_robots=args.num_robots,
        control_hz=args.control_hz,
        broker_type=args.action_chunk_broker_type.value,
        seed=args.seed,
        resize_size=args.resize_size,
        latency_ms=args.latency_ms,
    )

    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    runtime_metadata.to_json(output_path / "runtime_metadata.json")
    logging.info(f"Saved runtime metadata to {output_path / 'runtime_metadata.json'}")

    server_metadata.to_json(output_path / "server_metadata.json")
    logging.info(f"Saved server metadata to {output_path / 'server_metadata.json'}")

    # Run robots
    run_robots(args, jobs, server_metadata)
    calculate_metrics(pathlib.Path(args.output_dir))
    generate_all_plots(pathlib.Path(args.output_dir))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # allows multiple processes with envs
    main(tyro.cli(Args))
