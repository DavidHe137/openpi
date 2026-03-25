import json
import logging
import pathlib
import multiprocessing
import queue
import shutil
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Dict,
    Type,
)  # Any used for shared globals
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

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """A single episode: one task, one initial state."""

    idx: int  # 1-indexed
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
    output_dir: pathlib.Path = pathlib.Path("data/libero/multi_robot_videos")
    overwrite: bool = False
    progress_type: Literal["verbose", "concise", "logging", None] = "verbose"
    log_dir: Optional[pathlib.Path] = None
    debug: bool = False  # Run in single process with immediate progress output

    # FIXME: naming/convention on this
    def latency_for_robot(self, robot_idx: int) -> float:
        if not self.latency_ms:
            return 0.0
        return float(self.latency_ms[robot_idx])

    def execution_horizon_for_robot(self, robot_idx: int) -> int:
        if not self.execution_horizon:
            return 10
        return int(self.execution_horizon[robot_idx])

    @property
    def http_base(self) -> str:
        return f"http://{self.host}:{self.port}"


# Shared worker state: set via pool initializer so these are inherited by spawned
# processes rather than pickled as task arguments (multiprocessing.Queue and Barrier
# cannot be pickled after spawning).
_episode_queue: Optional[Any] = None
_progress_queue: Optional[Any] = None
_start_barrier: Optional[Any] = None


def _init_worker_shared(episode_queue, progress_queue, start_barrier) -> None:
    global _episode_queue, _progress_queue, _start_barrier
    _episode_queue = episode_queue
    _progress_queue = progress_queue
    _start_barrier = start_barrier


@dataclass
class _WorkerArgs:
    args: Args
    server_metadata: ServerMetadata
    robot_idx: int


class _StartupSyncSubscriber(_subscriber.Subscriber):
    """One-shot startup synchronization right before first episode steps."""

    def __init__(self) -> None:
        self._done = False

    def on_episode_start(self) -> None:
        if self._done:
            return
        if _start_barrier is not None:
            _start_barrier.wait()
        self._done = True
        # Notify the progress manager that this worker has crossed the start barrier.
        # The manager sets its start_time on the first such message it receives.
        if _progress_queue is not None:
            try:
                _progress_queue.put_nowait({"type": "run_start"})
            except Exception:
                pass

    def on_step(self, observation, action) -> None:
        return

    def on_episode_end(self) -> None:
        return


def _robot_worker(worker_args: _WorkerArgs) -> None:
    """Worker process: initialize, then pull episodes from the shared queue until empty."""
    args = worker_args.args
    robot_idx = worker_args.robot_idx

    # Stagger startup to avoid flooding the server with simultaneous warmup.
    time.sleep(robot_idx * 0.5)

    ws_client = BidirectionalWebsocket(
        robot_id=f"robot_{robot_idx}",
        host=args.host,
        port=args.port,
    )
    config = BrokerConfig(
        ws_client=ws_client,
        control_hz=args.control_hz,
        execution_horizon=args.execution_horizon_for_robot(robot_idx),
    )
    broker = args.action_chunk_broker_type.create(config)
    agent = _policy_agent.PolicyAgent(broker=broker)

    benchmark_dict: Dict[str, Type[benchmark.Benchmark]] = (
        benchmark.get_benchmark_dict()
    )
    task_suite = benchmark_dict[args.task_suite_name]()

    # Single instance reused across episodes so _done persists across iterations.
    startup_sync = _StartupSyncSubscriber()

    while True:
        try:
            episode = _episode_queue.get_nowait()
        except queue.Empty:
            break

        raw_env, _ = utils._get_libero_env(
            task_suite.get_task(episode.task_id),
            seed=args.seed + robot_idx,
        )
        env = LiberoSimEnvironment(
            env=raw_env,
            task_description=episode.task.language,
            initial_states=np.array([episode.initial_state]),
            resize_size=args.resize_size,
            max_episode_steps=args.max_steps,
            control_hz=args.control_hz,
        )

        subscribers: List[_subscriber.Subscriber] = [
            startup_sync,
            Saver(
                out_dir=args.output_dir,
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
            ProgressSubscriber(
                queue=_progress_queue,
                robot_idx=robot_idx,
                episode=episode,
                environment=env,
                update_frequency=10,
            ),
        ]

        runtime = _runtime.Runtime(
            environment=env,
            agent=agent,
            subscribers=subscribers,
            max_hz=args.control_hz,
            num_episodes=1,
            max_episode_steps=env._max_episode_steps,  # type: ignore[attr-defined]
        )
        runtime.run()
        runtime.close()


def run_robots(
    args: Args, episodes: List[Episode], server_metadata: ServerMetadata
) -> None:
    if args.debug:
        # Debug mode: single process for pdb compatibility, no progress manager.
        ep_queue: queue.Queue = queue.Queue()
        for ep in episodes:
            ep_queue.put(ep)
        _init_worker_shared(ep_queue, None, None)
        _robot_worker(
            _WorkerArgs(args=args, server_metadata=server_metadata, robot_idx=0)
        )
    else:
        total_episodes = len(episodes)
        active_workers = min(args.num_robots, total_episodes)
        start_barrier = multiprocessing.Barrier(active_workers, timeout=60)
        logging.info(
            "Using one-time startup barrier across %d worker(s)", active_workers
        )

        mp_episode_queue: multiprocessing.Queue = multiprocessing.Queue()
        for ep in episodes:
            mp_episode_queue.put(ep)

        with get_progress_manager(
            args.progress_type,
            total_episodes=total_episodes,
            max_steps=args.max_steps,
        ) as progress_manager:
            worker_args = [
                _WorkerArgs(args=args, server_metadata=server_metadata, robot_idx=i)
                for i in range(active_workers)
            ]
            with multiprocessing.Pool(
                processes=active_workers,
                initializer=_init_worker_shared,
                initargs=(mp_episode_queue, progress_manager.queue, start_barrier),
            ) as pool:
                try:
                    # use imap_unordered so that exceptions surface immediately
                    for _ in pool.imap_unordered(_robot_worker, worker_args):
                        pass
                except Exception as e:
                    logging.error(f"Error in robot worker: {e}")
                    raise
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
                    idx=len(episodes) + 1,
                    task_suite_name=args.task_suite_name,
                    task_id=task_id,
                    task=task,
                    initial_state=state,
                )
            )

    logging.info(
        "Created %d episodes across %d tasks", len(episodes), num_tasks_in_suite
    )
    random.shuffle(episodes)

    return episodes


def fetch_server_metadata(args: Args, timeout_s: float = 120.0) -> ServerMetadata:
    """Fetch server metadata, retrying until timeout_s seconds have elapsed."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            resp = requests.get(f"{args.http_base}/metadata", timeout=5.0)
            resp.raise_for_status()
            return ServerMetadata(**resp.json())
        except Exception as e:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Server at {args.http_base} did not respond within {timeout_s:.0f}s"
                ) from e
            logging.info("Waiting for server to be ready (%s); retrying in 5s...", e)
            time.sleep(5.0)


def reset_server(args: Args) -> None:
    try:
        requests.post(f"{args.http_base}/reset", timeout=5.0)
        logging.info("Reset server metrics")
    except Exception as e:
        logging.warning(f"Could not reset server metrics: {e}")


def save_server_metrics_history(args: Args) -> None:
    try:
        history = requests.get(f"{args.http_base}/save-metrics", timeout=10.0).json()
        hist_path = args.output_dir / "server_metrics_history.json"
        hist_path.write_text(json.dumps(history, indent=2))
        logging.info(f"Saved server metrics history to {hist_path}")
    except Exception as e:
        logging.warning(f"Could not fetch server metrics history: {e}")


def validate_args(args: Args) -> None:
    assert args.overwrite or not args.output_dir.exists(), (
        f"Output path {args.output_dir} already exists"
    )
    assert not args.latency_ms or len(args.latency_ms) == args.num_robots, (
        f"latency_ms must either be empty or have exactly {args.num_robots} values (one per robot), but got {len(args.latency_ms)} values"
    )
    assert (
        not args.execution_horizon or len(args.execution_horizon) == args.num_robots
    ), (
        f"execution_horizon must either be empty or have exactly {args.num_robots} values (one per robot), but got {len(args.execution_horizon)} values"
    )
    assert args.num_robots > 0, "num_robots must be positive"
    assert args.num_trials_per_task > 0, "num_trials_per_task must be positive"
    assert args.max_steps > 0, "max_steps must be positive"
    assert args.seed >= 0, "seed must be non-negative"
    assert args.resize_size > 0, "resize_size must be positive"


def main(args: Args) -> None:
    validate_args(args)

    if args.overwrite:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.log_dir is not None:
        log_file_name = f"libero_multi_robot_runtime_{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        log_file_path = args.log_dir / log_file_name
        args.log_dir.mkdir(parents=True, exist_ok=True)
        logging_config.setup_logging(
            log_path=log_file_path, level=logging.DEBUG if args.debug else logging.INFO
        )
    else:
        logging_config.setup_logging(
            level=logging.DEBUG if args.debug else logging.INFO
        )

    utils.seed_everything(args.seed)
    episodes = create_episodes(args)

    server_metadata = fetch_server_metadata(args)

    runtime_metadata = RuntimeMetadata(
        task_suite_name=args.task_suite_name,
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

    runtime_metadata.to_json(args.output_dir / "runtime_metadata.json")
    logging.info(
        f"Saved runtime metadata to {args.output_dir / 'runtime_metadata.json'}"
    )

    server_metadata.to_json(args.output_dir / "server_metadata.json")
    logging.info(f"Saved server metadata to {args.output_dir / 'server_metadata.json'}")

    reset_server(args)
    run_robots(args, episodes, server_metadata)

    save_server_metrics_history(args)

    calculate_metrics(args.output_dir)
    generate_all_plots(args.output_dir)


if __name__ == "__main__":
    # FIXME: look into this
    multiprocessing.set_start_method("spawn")  # allows multiple processes with envs
    main(tyro.cli(Args))
