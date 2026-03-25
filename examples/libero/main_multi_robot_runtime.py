import json
import logging
import pathlib
import multiprocessing
import shutil
from typing import List, Literal, Optional, Dict, Type
import datetime


import numpy as np
from jaxtyping import Float
from libero.libero import benchmark
from openpi_client.client import BidirectionalWebsocket
from openpi_client.runtime import runtime as _runtime, subscriber as _subscriber
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client.action_chunkers import ActionChunkBrokerType, BrokerConfig
from openpi_client.schemas import RuntimeMetadata, ServerMetadata
import requests
import tyro
from dataclasses import dataclass, field

from examples.libero import utils
from examples.libero import logging_config
from examples.libero.env import LiberoSimEnvironment
from examples.libero.progress_manager import get_progress_manager
from examples.libero.subscribers.saver import Saver
from examples.libero.subscribers.task_metrics_publisher import TaskMetricsPublisher
from examples.libero.metrics import calculate_metrics, generate_all_plots
from examples.libero.subscribers.progress_subscriber import ProgressSubscriber

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

logger = logging.getLogger(__name__)

# One-time startup synchronization state.
_start_barrier = None
_has_synced_start = False


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
    execution_horizon: List[int] = field(default_factory=list)
    latency_ms: List[float] = field(
        default_factory=list
    )  # Optional per-robot artificial latency (ms); length <= num_robots

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10  # Number of rollouts per task
    max_steps: int = 600  # Maximum number of control steps per episode

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


def _execution_horizon_for_robot(args: Args, robot_idx: int) -> int:
    """Return the execution horizon for a given robot index."""
    if not args.execution_horizon:
        return 10
    return int(args.execution_horizon[robot_idx])


def init_worker(args: Args, counter, progress_queue, start_barrier) -> None:
    global \
        robot_idx, \
        ws_client, \
        broker, \
        agent, \
        _progress_queue, \
        _start_barrier, \
        _has_synced_start
    with counter.get_lock():
        robot_idx = counter.value
        counter.value += 1

    # Store queue globally for access in create_runtime
    _progress_queue = progress_queue
    _start_barrier = start_barrier
    _has_synced_start = False

    ws_client = BidirectionalWebsocket(
        robot_id=f"robot_{robot_idx}",
        host=args.host,
        port=args.port,
    )

    # Create broker config and instantiate
    config = BrokerConfig(
        ws_client=ws_client,
        control_hz=args.control_hz,
        execution_horizon=_execution_horizon_for_robot(args, robot_idx),
    )
    broker = args.action_chunk_broker_type.create(config)
    agent = _policy_agent.PolicyAgent(broker=broker)


def _wait_for_initial_start_sync() -> None:
    """Block on a one-time startup barrier before first control step."""
    global _has_synced_start  # NOTE: shared between workers. maybe can persist worker state elsewhere to avoid global, but I think this is fine
    if _has_synced_start:
        return

    if _start_barrier is None:
        _has_synced_start = True
        return

    _start_barrier.wait()
    _has_synced_start = True


class _StartupSyncSubscriber(_subscriber.Subscriber):
    """One-shot startup synchronization right before first episode steps."""

    def on_episode_start(self) -> None:
        _wait_for_initial_start_sync()

    def on_step(self, observation, action) -> None:
        return

    def on_episode_end(self) -> None:
        return


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
        _StartupSyncSubscriber(),
        Saver(
            out_dir=pathlib.Path(args.output_dir),
            environment=env,
            action_chunk_broker=broker,
            task_suite_name=job.task_suite_name,
            task_id=job.task_id,
            task=job.task,
            robot_idx=robot_idx,
        ),
        TaskMetricsPublisher(
            ws_client=ws_client,
            environment=env,
            task_suite_name=job.task_suite_name,
            task_id=job.task_id,
            task=job.task,
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
    args, job, _server_metadata = task_args
    runtime = create_runtime(args, job)
    runtime.run()
    runtime.close()


def run_robots(args: Args, jobs: List[Job], server_metadata: ServerMetadata) -> None:
    if not jobs:
        logging.info("No jobs to run; skipping robot startup")
        return

    counter = multiprocessing.Value("i", 0)  # for assigning robot indices

    if args.debug:
        # Debug mode: no progress manager, single process for pdb compatibility
        init_worker(args, counter, None, None)
        for job in jobs:
            _robot_worker((args, job, server_metadata))
    else:
        total_episodes = sum(len(job.initial_states) for job in jobs)
        active_workers = min(args.num_robots, len(jobs))
        start_barrier = multiprocessing.Barrier(active_workers)
        logging.info(
            "Using one-time startup barrier across %d worker(s)",
            active_workers,
        )
        with get_progress_manager(
            args.progress_type,
            total_jobs=len(jobs),
            total_episodes=total_episodes,
            max_steps=args.max_steps,
        ) as progress_manager:
            # Pass queue to worker initializer
            with multiprocessing.Pool(
                processes=active_workers,
                initializer=init_worker,
                initargs=(args, counter, progress_manager.queue, start_barrier),
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
        args.num_trials_per_task,
        args.control_hz,
    )

    base_jobs: List[Job] = []
    for task_id in range(num_tasks_in_suite):
        task: benchmark.Task = task_suite.get_task(task_id)
        all_initial_states: Float[np.ndarray, "n_initial_states state_dim"] = (
            task_suite.get_task_init_states(task_id)
        )

        if len(all_initial_states) < args.num_trials_per_task:
            logging.error(
                "Task %d has less initial states than trials per robot; skipping",
                task_id,
            )
            continue

        initial_states = all_initial_states[: args.num_trials_per_task]
        job = Job(
            task=task,
            task_suite_name=args.task_suite_name,
            task_id=task_id,
            initial_states=initial_states,
        )
        base_jobs.append(job)

    # If num_robots > num_tasks, repeat tasks cyclically so every robot gets a job.
    if args.num_robots > len(base_jobs):
        logging.warning(
            "num_robots (%d) > num_tasks (%d); repeating tasks cyclically to fill all robots",
            args.num_robots,
            len(base_jobs),
        )
        jobs: List[Job] = [
            base_jobs[i % len(base_jobs)] for i in range(args.num_robots)
        ]
    else:
        jobs = base_jobs

    logging.info("Created %d jobs", len(jobs))

    return jobs


def main(args: Args) -> None:
    if args.log_dir is not None:
        log_file_name = f"libero_multi_robot_runtime_{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        log_file_path = pathlib.Path(args.log_dir) / log_file_name
        pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        logging_config.setup_logging(
            log_path=log_file_path, level=logging.DEBUG if args.debug else logging.INFO
        )
    else:
        logging_config.setup_logging(
            level=logging.DEBUG if args.debug else logging.INFO
        )

    if not args.overwrite and pathlib.Path(args.output_dir).exists():
        raise ValueError(f"Output path {args.output_dir} already exists")
    if args.overwrite:
        if pathlib.Path(args.output_dir).exists():
            shutil.rmtree(args.output_dir, ignore_errors=True)
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Validate latency specification
    if args.latency_ms and len(args.latency_ms) != args.num_robots:
        raise ValueError(
            f"latency_ms must either be empty or have exactly {args.num_robots} values "
            f"(one per robot), but got {len(args.latency_ms)} values"
        )

    np.random.seed(args.seed)

    jobs = create_jobs(args)

    # Fetch server metadata over HTTP to avoid creating a temporary websocket robot.
    metadata_resp = requests.get(
        f"http://{args.host}:{args.port}/metadata", timeout=5.0
    )
    metadata_resp.raise_for_status()
    server_metadata = ServerMetadata(**metadata_resp.json())

    # Create runtime metadata
    runtime_metadata = RuntimeMetadata(
        task_suite_name=args.task_suite_name,
        num_steps_wait=args.num_steps_wait,
        num_trials_per_task=args.num_trials_per_task,
        max_steps=args.max_steps,
        num_robots=args.num_robots,
        control_hz=args.control_hz,
        broker_type=args.action_chunk_broker_type.value,
        seed=args.seed,
        resize_size=args.resize_size,
        latency_ms=args.latency_ms,
        execution_horizon=args.execution_horizon,
    )

    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    runtime_metadata.to_json(output_path / "runtime_metadata.json")
    logging.info(f"Saved runtime metadata to {output_path / 'runtime_metadata.json'}")

    server_metadata.to_json(output_path / "server_metadata.json")
    logging.info(f"Saved server metadata to {output_path / 'server_metadata.json'}")

    # Reset server-side metrics so this experiment gets a clean slate
    server_base = f"http://{args.host}:{args.port}"
    try:
        requests.post(f"{server_base}/reset", timeout=5.0)
        logging.info("Reset server metrics")
    except Exception as e:
        logging.warning(f"Could not reset server metrics: {e}")

    # Run robots
    run_robots(args, jobs, server_metadata)

    # Dump server-side metrics history for offline analysis
    try:
        history = requests.get(f"{server_base}/save-metrics", timeout=10.0).json()
        hist_path = output_path / "server_metrics_history.json"
        hist_path.write_text(json.dumps(history, indent=2))
        logging.info(f"Saved server metrics history to {hist_path}")
    except Exception as e:
        logging.warning(f"Could not fetch server metrics history: {e}")

    calculate_metrics(pathlib.Path(args.output_dir))
    generate_all_plots(pathlib.Path(args.output_dir))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # allows multiple processes with envs
    main(tyro.cli(Args))
