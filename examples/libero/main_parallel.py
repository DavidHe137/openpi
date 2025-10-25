import collections
import dataclasses
import logging
import math
from multiprocessing import Manager
from multiprocessing import Pool
import os
import pathlib
import threading
import time
from typing import Optional, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.benchmark import Task
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


_worker_env: Optional[OffScreenRenderEnv] = None
_worker_task_description: Optional[str] = None
_worker_client: Optional[_websocket_client_policy.WebsocketClientPolicy] = None
_worker_status_dict: Optional[dict] = None
_worker_results_dict: Optional[dict] = None


def init_worker(task: Task, args: Args, status_dict, results_dict) -> None:
    global _worker_env, _worker_task_description
    global _worker_client, _worker_status_dict, _worker_results_dict  # noqa: PLW0603
    _worker_env, _worker_task_description = _get_libero_env(
        task, LIBERO_ENV_RESOLUTION, args.seed
    )
    _worker_client = _websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port
    )
    _worker_status_dict = status_dict
    _worker_results_dict = results_dict

    assert _worker_status_dict is not None
    pid = os.getpid()
    _worker_status_dict[pid] = "initialized"


def _eval_libero_wrapper(task_args):
    """Wrapper to unpack arguments for eval_libero."""
    return eval_libero(*task_args)


def eval_libero(
    args: Args,
    initial_states: np.ndarray,
    episode_idx: int,
    max_steps: int,
) -> bool:
    assert _worker_env is not None
    assert _worker_client is not None
    assert _worker_task_description is not None
    assert _worker_status_dict is not None
    assert _worker_results_dict is not None
    env = _worker_env
    client = _worker_client
    task_description = _worker_task_description

    pid = os.getpid()
    _worker_status_dict[pid] = f"ep{episode_idx}: resetting"

    # Reset environment
    env.reset()
    action_plan = collections.deque()

    # Set initial states
    obs = env.set_init_state(initial_states[episode_idx])

    # Setup
    t = 0
    replay_images = []

    success = False
    _worker_status_dict[pid] = f"ep{episode_idx}: t=0 waiting"
    while t < max_steps + args.num_steps_wait:
        try:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                _worker_status_dict[pid] = f"ep{episode_idx}: t={t} waiting"
                continue

            # Get preprocessed image
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            _worker_status_dict[pid] = f"ep{episode_idx}: t={t} preprocessing"
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(
                obs["robot0_eye_in_hand_image"][::-1, ::-1]
            )
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    wrist_img, args.resize_size, args.resize_size
                )
            )

            # Save preprocessed image for replay video
            replay_images.append(img)

            if not action_plan:
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                _worker_status_dict[pid] = f"ep{episode_idx}: t={t} preparing obs"
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(_worker_task_description),
                }

                # Query model to get action
                _worker_status_dict[pid] = f"ep{episode_idx}: t={t} waiting for server"
                action_chunk = client.infer(element)["actions"]
                assert len(action_chunk) >= args.replan_steps, (
                    f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                )
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()

            # Execute action in environment
            _worker_status_dict[pid] = f"ep{episode_idx}: t={t} stepping"
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

        except Exception as e:
            logging.error(f"Caught exception: {e}")
            break

    # Save a replay video of the episode
    _worker_status_dict[pid] = f"ep{episode_idx}: saving video"
    suffix = "success" if success else "failure"
    task_segment = task_description.replace(" ", "_")
    imageio.mimwrite(
        pathlib.Path(args.video_out_path)
        / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4",
        [np.asarray(x) for x in replay_images],
        fps=10,
    )

    _worker_status_dict[pid] = f"ep{episode_idx}: done ({suffix})"
    _worker_results_dict[episode_idx] = success
    return success


def _clear_lines(num_lines):
    """Move cursor up and clear lines."""
    import sys

    sys.stdout.write(f"\033[{num_lines}A")  # Move cursor up
    sys.stdout.write("\033[J")  # Clear from cursor to end


def _print_worker_status(status_dict, num_workers):
    """Print worker status, one worker per line."""
    import sys

    statuses = list(status_dict.values())
    for i in range(num_workers):
        status = statuses[i] if i < len(statuses) else "idle"
        sys.stdout.write(f"  W{i}: {status}\n")
    sys.stdout.flush()


def monitor_worker_status(status_dict, stop_event, pbar, num_workers=4):
    """Background thread to monitor and update worker status display."""
    last_status = {}

    while not stop_event.is_set():
        current_status = dict(status_dict)
        if current_status != last_status:
            if last_status:
                _clear_lines(num_workers)
            _print_worker_status(current_status, num_workers)
            last_status = current_status
        time.sleep(0.1)


def main(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Start episodes
        task_episodes, task_successes = 0, 0

        # Create shared state for worker status tracking
        with Manager() as manager:
            status_dict = manager.dict()
            results_dict = manager.dict()

            with Pool(
                processes=4,
                initializer=init_worker,
                initargs=(task, args, status_dict, results_dict),
            ) as pool:
                # Create progress bar
                pbar = tqdm.tqdm(
                    pool.imap(
                        _eval_libero_wrapper,
                        [
                            (args, initial_states, episode_idx, max_steps)
                            for episode_idx in range(args.num_trials_per_task)
                        ],
                    ),
                    total=args.num_trials_per_task,
                    desc=f"Task {task_id}",
                )

                # Start background thread to monitor worker status
                stop_event = threading.Event()
                monitor_thread = threading.Thread(
                    target=monitor_worker_status, args=(status_dict, stop_event, pbar)
                )
                monitor_thread.start()

                try:
                    # Collect results
                    results = list(pbar)

                    # Count successes
                    task_episodes = len(results)
                    task_successes = sum(results)
                    total_episodes += task_episodes
                    total_successes += task_successes
                finally:
                    # Stop monitoring thread
                    stop_event.set()
                    monitor_thread.join()
                    pbar.close()

        # Log task results
        logging.info(
            f"Task {task_id} success rate: {task_successes}/{task_episodes} ({task_successes / task_episodes * 100:.1f}%)"
        )
        logging.info(
            f"Cumulative success rate: {total_successes}/{total_episodes} ({total_successes / total_episodes * 100:.1f}%)"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(
    task: Task, resolution: int, seed: int
) -> Tuple[OffScreenRenderEnv, str]:
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
