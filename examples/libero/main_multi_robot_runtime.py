import json
import logging
import pathlib
import multiprocessing
import queue
import shutil
from typing import List, Literal, Optional, Dict, Type
import random
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

# Per-worker globals (set in init_worker, used in _robot_worker)
robot_idx: int = 0
ws_client = None
broker = None
agent = None
_progress_queue = None
_task_suite = None
_first_episode = None
_worker_progress_subscriber: Optional[ProgressSubscriber] = None
_episode_queue = None  # multiprocessing.Queue inherited via initargs


@dataclass
class Episode:
    """A single episode: one task, one initial state."""

    task_suite_name: str
    task_id: int
    task: benchmark.Task
    initial_state: np.ndarray

    def __str__(self) -> str:
        return f"Episode(task_suite_name={self.task_suite_name}, task_id={self.task_id}, task={self.task.language})"


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


def init_worker(
    args: Args, counter, progress_queue, start_barrier, episode_queue=None
) -> None:
    global \
        robot_idx, \
        ws_client, \
        broker, \
        agent, \
        _progress_queue, \
        _start_barrier, \
        _has_synced_start, \
        _task_suite, \
        _first_episode, \
        _worker_progress_subscriber, \
        _episode_queue
    with counter.get_lock():
        robot_idx = counter.value
        counter.value += 1

    # Store queue globally for access in _robot_worker
    _progress_queue = progress_queue
    _start_barrier = start_barrier
    _has_synced_start = False

    # to avoid flooding the server with simultaneous warmup
    time.sleep(robot_idx * 0.5)

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

    benchmark_dict: Dict[str, Type[benchmark.Benchmark]] = (
        benchmark.get_benchmark_dict()
    )
    _task_suite = benchmark_dict[args.task_suite_name]()

    # Store episode queue globally so _robot_worker can access it without pickling
    _episode_queue = episode_queue

    # Pre-fetch first episode from queue
    _first_episode = None
    if episode_queue is not None:
        try:
            _first_episode = episode_queue.get_nowait()
        except queue.Empty:
            pass

    # Create progress subscriber using first episode's task info
    _worker_progress_subscriber = None
    if progress_queue is not None and _first_episode is not None:
        job_info = {
            "task_suite_name": _first_episode.task_suite_name,
            "task_id": _first_episode.task_id,
        }
        _worker_progress_subscriber = ProgressSubscriber(
            queue=progress_queue,
            robot_idx=robot_idx,
            job_info=job_info,
            environment=None,
            update_frequency=10,
        )


def _wait_for_initial_start_sync() -> None:
    """Block on a one-time startup barrier before first control step."""
    global _has_synced_start  # NOTE: shared between workers. maybe can persist worker state elsewhere to avoid global, but I think this is fine
    if _has_synced_start:
        return

    if _start_barrier is None:
        _has_synced_start = True
    else:
        _start_barrier.wait()
        _has_synced_start = True

    # Notify the progress manager that this worker has crossed the start barrier.
    # The manager sets its start_time on the first such message it receives.
    if _progress_queue is not None:
        try:
            _progress_queue.put_nowait({"type": "run_start"})
        except Exception:
            pass


class _StartupSyncSubscriber(_subscriber.Subscriber):
    """One-shot startup synchronization right before first episode steps."""

    def on_episode_start(self) -> None:
        _wait_for_initial_start_sync()

    def on_step(self, observation, action) -> None:
        return

    def on_episode_end(self) -> None:
        return


def _robot_worker(task_args) -> None:
    """Worker process that pulls episodes from the shared queue until empty."""
    args, server_metadata = task_args

    episode = _first_episode  # pre-fetched in init_worker
    total_completed = 0
    total_successes = 0

    while episode is not None:
        raw_env, _ = utils._get_libero_env(
            _task_suite.get_task(episode.task_id),
            LIBERO_ENV_RESOLUTION,
            seed=args.seed + robot_idx,
        )
        env = LiberoSimEnvironment(
            env=raw_env,
            task_description=episode.task.language,
            initial_states=np.array([episode.initial_state]),
            resize_size=args.resize_size,
            num_steps_wait=args.num_steps_wait,
            max_episode_steps=args.max_steps,
            latency_ms=_latency_for_robot(args, robot_idx),
            control_hz=args.control_hz,
        )

        if _worker_progress_subscriber is not None:
            _worker_progress_subscriber.environment = env

        subscribers: List[_subscriber.Subscriber] = [
            _StartupSyncSubscriber(),
            Saver(
                out_dir=pathlib.Path(args.output_dir),
                environment=env,
                action_chunk_broker=broker,
                task_suite_name=episode.task_suite_name,
                task_id=episode.task_id,
                task=episode.task,
                robot_idx=robot_idx,
            ),
            TaskMetricsPublisher(
                ws_client=ws_client,
                environment=env,
                task_suite_name=episode.task_suite_name,
                task_id=episode.task_id,
                task=episode.task,
            ),
        ]
        if _worker_progress_subscriber is not None:
            subscribers.append(_worker_progress_subscriber)

        runtime = _runtime.Runtime(
            environment=env,
            agent=agent,
            subscribers=subscribers,
            max_hz=args.control_hz,
            num_episodes=1,
            max_episode_steps=env._max_episode_steps,  # type: ignore[attr-defined]
        )
        runtime.run()
        raw_env.close()

        total_completed += 1
        if env.current_success:
            total_successes += 1

        try:
            episode = _episode_queue.get_nowait()
        except queue.Empty:
            episode = None

    # Emit worker_complete
    if _worker_progress_subscriber is not None:
        _worker_progress_subscriber.close(total_completed, total_successes)


def run_robots(
    args: Args, episodes: List[Episode], server_metadata: ServerMetadata
) -> None:
    if not episodes:
        logging.info("No episodes to run; skipping robot startup")
        return

    counter = multiprocessing.Value("i", 0)  # for assigning robot indices

    if args.debug:
        # Debug mode: no progress manager, single process for pdb compatibility
        ep_queue: queue.Queue = queue.Queue()
        for ep in episodes:
            ep_queue.put(ep)
        init_worker(args, counter, None, None, ep_queue)
        _robot_worker((args, server_metadata))
    else:
        total_episodes = len(episodes)
        active_workers = min(args.num_robots, total_episodes)
        start_barrier = multiprocessing.Barrier(active_workers, timeout=60)
        logging.info(
            "Using one-time startup barrier across %d worker(s)",
            active_workers,
        )

        # Build shared episode queue
        mp_episode_queue: multiprocessing.Queue = multiprocessing.Queue()
        for ep in episodes:
            mp_episode_queue.put(ep)

        with get_progress_manager(
            args.progress_type,
            total_episodes=total_episodes,
            max_steps=args.max_steps,
        ) as progress_manager:
            # Pass queue to worker initializer
            with multiprocessing.Pool(
                processes=active_workers,
                initializer=init_worker,
                initargs=(
                    args,
                    counter,
                    progress_manager.queue,
                    start_barrier,
                    mp_episode_queue,
                ),
            ) as pool:
                try:
                    # use imap_unordered so that it exits immediately on any exception
                    _ = list(
                        pool.imap_unordered(
                            _robot_worker,
                            [(args, server_metadata)] * active_workers,
                        )
                    )
                except Exception as e:
                    logging.error(f"Error in robot worker: {e}")
                    raise e
                finally:
                    pool.close()
                    pool.join()


def create_episodes(args: Args) -> List[Episode]:
    benchmark_dict: Dict[str, Type[benchmark.Benchmark]] = (
        benchmark.get_benchmark_dict()
    )
    task_suite: benchmark.Benchmark = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    logging.info(
        "Setting up multi-robot LIBERO runtime over suite '%s' with %d tasks, num_robots=%d, trials_per_task=%d, control_hz=%d",
        args.task_suite_name,
        num_tasks_in_suite,
        args.num_robots,
        args.num_trials_per_task,
        args.control_hz,
    )

    episodes: List[Episode] = []
    for task_id in range(num_tasks_in_suite):
        task: benchmark.Task = task_suite.get_task(task_id)
        all_initial_states: Float[np.ndarray, "n_initial_states state_dim"] = (
            task_suite.get_task_init_states(task_id)
        )

        if len(all_initial_states) < args.num_trials_per_task:
            logging.error(
                "Task %d has less initial states than trials per task; skipping",
                task_id,
            )
            continue

        initial_states = all_initial_states[: args.num_trials_per_task]
        for state in initial_states:
            episodes.append(
                Episode(
                    task_suite_name=args.task_suite_name,
                    task_id=task_id,
                    task=task,
                    initial_state=state,
                )
            )

    logging.info(
        "Created %d episodes across %d tasks", len(episodes), num_tasks_in_suite
    )

    random.seed(args.seed)
    random.shuffle(episodes)

    return episodes


def main(args: Args) -> None:
    # Set up a temporary console-only logger until the output dir is ready.
    logging_config.setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    if not args.overwrite and pathlib.Path(args.output_dir).exists():
        raise ValueError(f"Output path {args.output_dir} already exists")
    if args.overwrite:
        if pathlib.Path(args.output_dir).exists():
            shutil.rmtree(args.output_dir, ignore_errors=True)
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Now that the output dir exists (and won't be deleted), open the log file.
    if args.log_dir is not None:
        log_file_name = f"libero_multi_robot_runtime_{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        log_file_path = pathlib.Path(args.log_dir) / log_file_name
        pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        logging_config.setup_logging(
            log_path=log_file_path, level=logging.DEBUG if args.debug else logging.INFO
        )

    # Validate latency specification
    if args.latency_ms and len(args.latency_ms) != args.num_robots:
        raise ValueError(
            f"latency_ms must either be empty or have exactly {args.num_robots} values "
            f"(one per robot), but got {len(args.latency_ms)} values"
        )

    np.random.seed(args.seed)

    episodes = create_episodes(args)

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
        episodes=[str(ep) for ep in episodes],
        execution_horizon=args.execution_horizon,
    )
    output_path = pathlib.Path(args.output_dir)

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
    run_robots(args, episodes, server_metadata)

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
