import pandas as pd
import logging
import pathlib
import multiprocessing
import shutil
from typing import Union, List, Literal, Optional
import datetime
import time

import numpy as np
from libero.libero import benchmark
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client.action_chunkers import (
    ActionChunkBrokerType,
    SyncBrokerConfig,
    RTCBrokerConfig,
)
import tyro
from dataclasses import dataclass, asdict, field
from rich.console import Console
from rich.table import Table

from examples.libero import utils
from examples.libero import logging_config
from examples.libero.env import LiberoSimEnvironment
from examples.libero.progress_manager import get_progress_manager
from examples.libero.subscribers.saver import Saver, Result
from examples.libero.subscribers.progress_subscriber import ProgressSubscriber

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclass
class Job:
    """A job is a task with a batch of episodes."""

    task_suite_name: str

    task: benchmark.Task
    task_id: int
    initial_states: np.ndarray  # batch, state_dim


@dataclass
class ActionChunkBrokerArgs:
    """Arguments for action chunk broker configuration."""

    broker_type: ActionChunkBrokerType = ActionChunkBrokerType.SYNC
    # RTC-specific params
    s_min: int = 5
    d_init: int = 4


@dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8080
    resize_size: int = 224
    action_horizon: int = (
        50  # Action horizon for ActionChunkBroker (matches Libero model config)
    )
    action_chunk_broker: ActionChunkBrokerArgs = field(
        default_factory=ActionChunkBrokerArgs
    )
    latency_ms: List[float] = field(
        default_factory=list
    )  # Optional per-robot artificial latency (ms); length <= num_robots

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_robot: int = 10  # Number of rollouts per robot per task
    max_steps: int = 300  # Maximum number of control steps per episode

    #################################################################################################################
    # Multi-robot / threading parameters
    #################################################################################################################
    num_robots: int = 9  # Number of always-running sims (robots)
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
    save_debug_data: bool = False  # Save debug data (obs before/after preprocess, noise, output) to npy files

    def create_broker_config(self, policy) -> Union[SyncBrokerConfig, RTCBrokerConfig]:
        """Helper to create the appropriate broker config from args."""
        if self.action_chunk_broker.broker_type == ActionChunkBrokerType.RTC:
            return RTCBrokerConfig(
                policy=policy,
                action_horizon=self.action_horizon,
                s_min=self.action_chunk_broker.s_min,
                d_init=self.action_chunk_broker.d_init,
                return_debug_data=self.save_debug_data,
            )
        else:  # SYNC
            return SyncBrokerConfig(
                policy=policy,
                action_horizon=self.action_horizon,
                return_debug_data=self.save_debug_data,
            )


def _latency_for_robot(args: Args, robot_idx: int) -> float:
    """Return the latency (in ms) to use for a given robot index."""
    if not args.latency_ms:
        return 0.0
    return float(args.latency_ms[robot_idx])


def delay_start(
    action_horizon: int,
    control_hz: int,
    action_chunk_broker_config: ActionChunkBrokerArgs,
):
    """Return the period (in seconds) that a robot waits between requests."""
    period = 0.0
    if action_chunk_broker_config.broker_type == ActionChunkBrokerType.SYNC:
        period = action_horizon / control_hz
    elif action_chunk_broker_config.broker_type == ActionChunkBrokerType.RTC:
        period = action_chunk_broker_config.s_min / control_hz
    else:
        raise ValueError(
            f"Unknown action chunk broker type: {action_chunk_broker_config.broker_type}"
        )

    delay = np.random.uniform(0, period)
    time.sleep(delay)


def init_worker(args: Args, counter, progress_queue) -> None:
    global robot_idx, ws_client, broker, agent, _progress_queue
    with counter.get_lock():
        robot_idx = counter.value
        counter.value += 1

    # Store queue globally for access in create_runtime
    _progress_queue = progress_queue

    ws_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
    )

    # Create broker config and instantiate
    config = args.create_broker_config(ws_client)
    broker = args.action_chunk_broker.broker_type.create(config)
    agent = _policy_agent.PolicyAgent(policy=broker)


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

    subscribers = [
        Saver(
            out_dir=pathlib.Path(args.output_dir),
            environment=env,
            action_chunk_broker=broker,
            task_suite_name=job.task_suite_name,
            task_id=job.task_id,
            robot_idx=robot_idx,
        ),
    ]
    if args.progress_type is not None:
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
    args, job = task_args
    runtime = create_runtime(args, job)
    delay_start(
        action_horizon=args.action_horizon,
        control_hz=args.control_hz,
        action_chunk_broker_config=args.action_chunk_broker,
    )

    runtime.run()
    runtime.close()


def run_robots(args: Args, jobs: List[Job]) -> None:
    counter = multiprocessing.Value("i", 0)  # for assigning robot indices

    with get_progress_manager(
        args.progress_type, max_steps=args.max_steps
    ) as progress_manager:
        if args.debug:
            init_worker(args, counter, progress_manager.queue)
            for job in jobs:
                _robot_worker((args, job))
        else:
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
                            _robot_worker, [(args, job) for job in jobs]
                        )
                    )
                except Exception as e:
                    logging.error(f"Error in robot worker: {e}")
                    raise e
                finally:
                    pool.close()
                    pool.join()


def create_jobs(args: Args) -> List[Job]:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    logging.info(
        "Setting up multi-robot LIBERO runtime over suite '%s' with %d tasks, num_robots=%d, trials_per_robot=%d, action_horizon=%d, control_hz=%d",
        args.task_suite_name,
        num_tasks_in_suite,
        args.num_robots,
        args.num_trials_per_robot,
        args.action_horizon,
        args.control_hz,
    )

    jobs: List[Job] = []
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        if task_id != 8:
            continue
        all_initial_states = task_suite.get_task_init_states(task_id)

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

    n_repeats = (args.num_robots // len(jobs)) + 1
    jobs = jobs * n_repeats
    jobs = jobs[: args.num_robots]

    logging.info("Created %d jobs", len(jobs))

    return jobs


def aggregate_results(output_path: pathlib.Path) -> None:
    metadata_files = list(output_path.glob("**/metadata.json"))

    results: List[Result] = []
    for metadata_file in metadata_files:
        result = Result.from_json(metadata_file)
        results.append(result)

    results_df = pd.DataFrame([asdict(result) for result in results])
    results_df.to_csv(output_path / "results.csv", index=False)
    summary = results_df.groupby(["task_suite_name", "task_id"]).agg(
        {
            "success": "mean",
        }
    )
    summary.reset_index().to_csv(output_path / "summary.csv", index=False)

    # Display results using rich
    console = Console()
    table = Table(title="Task Success Summary")
    table.add_column("Task Suite", style="cyan")
    table.add_column("Task ID", style="magenta")
    table.add_column("Success Rate", style="green")

    for _, row in summary.reset_index().iterrows():
        table.add_row(
            str(row["task_suite_name"]), str(row["task_id"]), f"{row['success']:.2%}"
        )

    console.print(table)
    console.print(
        f"\n[bold green]Total success rate: {summary['success'].mean():.2%}[/bold green]"
    )


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
    _ = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
    )  # to wait for the server to be ready
    run_robots(args, jobs)
    aggregate_results(pathlib.Path(args.output_dir))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # allows multiple processes with envs
    main(tyro.cli(Args))
