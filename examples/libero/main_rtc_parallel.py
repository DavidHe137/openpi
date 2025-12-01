import collections
import csv
import dataclasses
from datetime import datetime
import logging
import math
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Pool
import os
import pathlib
import threading
import time
from typing import Optional, Tuple, Union, List

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.benchmark import Task
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import action_chunk_broker
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
    port: int = 8080
    resize_size: int = 224
    action_horizon: int = 10  # Action horizon for ActionChunkBroker (matches Libero model config)
    latency_ms: Tuple[float, ...] = (0.0,)  # Artificial latency to inject during inference (in milliseconds). Can be a single float or list of floats.

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 20  # Number of rollouts per task
    num_workers: int = 6  # Number of parallel workers

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_csv_path: str = "data/libero/results.csv"  # Path to save CSV results

    seed: int = 18  # Random Seed (for reproducibility)

    use_rtc: bool = False


# Worker global variables
_worker_env: Optional[OffScreenRenderEnv] = None
_worker_task_description: Optional[str] = None
_worker_client: Optional[action_chunk_broker.ActionChunkBroker] = None
_worker_status_dict: Optional[dict] = None
_worker_results_dict: Optional[dict] = None
_worker_args: Optional[Args] = None


def init_worker(task: Task, args: Args, status_dict, results_dict) -> None:
    """Initialize worker process with environment and policy client."""
    global _worker_env, _worker_task_description
    global _worker_client, _worker_status_dict, _worker_results_dict, _worker_args  # noqa: PLW0603

    _worker_env, _worker_task_description = _get_libero_env(
        task, LIBERO_ENV_RESOLUTION, args.seed
    )
    
    # Determine s and d parameters based on action horizon
    s = None
    d = None
    if args.action_horizon == 10:
        s = 5
        d = 3
    elif args.action_horizon == 25:
        s = 12
        d = 6
    elif args.action_horizon == 50:
        s = 25
        d = 6
    elif args.action_horizon == 100:
        s = 40
        d = 6
    else:
        raise ValueError(f"Unknown action horizon: {args.action_horizon}")
    
    # Initialize ActionChunkBroker with RTC support
    ws_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port, latency_ms=args.latency_ms)
    _worker_client = action_chunk_broker.ActionChunkBroker(
        policy=ws_client,
        action_horizon=args.action_horizon,
        is_rtc=args.use_rtc,
        s=s,
        d=d,
    )
    
    _worker_status_dict = status_dict
    _worker_results_dict = results_dict
    _worker_args = args

    assert _worker_status_dict is not None
    pid = os.getpid()
    _worker_status_dict[pid] = "initialized"


def _eval_libero_wrapper(task_args):
    """Wrapper to unpack arguments for eval_libero."""
    return eval_libero(*task_args)


def eval_libero(
    task_id: int,
    initial_states: np.ndarray,
    episode_idx: int,
    max_steps: int,
    video_out_path: pathlib.Path,
) -> dict:
    """Run a single episode evaluation."""
    assert _worker_env is not None
    assert _worker_client is not None
    assert _worker_task_description is not None
    assert _worker_status_dict is not None
    assert _worker_results_dict is not None
    assert _worker_args is not None
    
    env = _worker_env
    client = _worker_client
    task_description = _worker_task_description
    args = _worker_args

    pid = os.getpid()
    _worker_status_dict[pid] = f"ep{episode_idx}: resetting"

    # Reset environment
    env.reset()
    
    # Reset the action chunk broker for each episode
    client.reset()

    # Set initial states
    obs = env.set_init_state(initial_states)

    # Setup
    t = 0
    replay_images = []
    last_gripper = -1.0

    success = False
    _worker_status_dict[pid] = f"ep{episode_idx}: t=0 waiting"

    inference_times = []
    additional_delay_times = []
    
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
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

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
                "prompt": str(task_description),
            }

            # Query model to get action
            _worker_status_dict[pid] = f"ep{episode_idx}: t={t} inferring"
            inference_start = time.time()
            action = client.infer(element)["actions"]
            actual_inference_time = time.time() - inference_start
            inference_times.append(actual_inference_time)
            # Calculate additional delay needed
            target_latency_seconds = args.latency_ms / 1000.0
            additional_delay = max(0, target_latency_seconds - actual_inference_time)
            additional_delay_times.append(additional_delay)
            # Simulate the additional delay by repeating last action
            if additional_delay > 0:
                _worker_status_dict[pid] = f"ep{episode_idx}: t={t} delaying"
                delay_steps = int(additional_delay / 0.05)  # Assuming 20Hz
                for _ in range(delay_steps):
                    # Save frame during delay to show pause in video
                    replay_images.append(img)
                    obs, reward, done, info = env.step([0.0] * 6 + [last_gripper])
                    # t += 1
                    # if done or t >= max_steps:
                    #     break
                    if done:
                        success = True
                        break
                if done:
                    break

            # Save current frame for replay video (after delay)
            replay_images.append(img)

            # Execute action in environment
            _worker_status_dict[pid] = f"ep{episode_idx}: t={t} stepping"
            obs, reward, done, info = env.step(action.tolist())
            last_gripper = action.tolist()[6]
            if done:
                success = True
                break
            t += 1

        except Exception as e:
            logging.error(f"Worker {pid} caught exception: {e}")
            break

    # Save video
    _worker_status_dict[pid] = f"ep{episode_idx}: saving video"
    suffix = "succ" if success else "fail"
    task_segment = task_description.replace(" ", "_")
    
    video_filename = video_out_path / f"task_{task_id}_ep{episode_idx}_rtc{args.use_rtc}_hrzn{args.action_horizon}_{suffix}.mp4"
    imageio.mimwrite(
        video_filename,
        [np.asarray(x) for x in replay_images],
        fps=10,
    )

    # Prepare result dict
    episode_result = {
        'task_id': task_id,
        'task_description': task_description,
        'episode_idx': episode_idx,
        'success': bool(success),
        'steps_taken': t,
        'max_steps': max_steps + args.num_steps_wait,
        'task_suite': args.task_suite_name,
        'seed': args.seed,
        'use_rtc': args.use_rtc,
        'action_horizon': args.action_horizon,
        'latency_ms': args.latency_ms,
        'avg_inference_time': np.mean(inference_times),
        'avg_additional_delay_time': np.mean(additional_delay_times)
    }

    _worker_status_dict[pid] = f"ep{episode_idx}: done ({suffix})"
    _worker_results_dict[episode_idx] = success
    
    return episode_result


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


def monitor_worker_status(status_dict, stop_event, num_workers):
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


def run_experiment_for_latency(args: Args, latency_value: float, timestamp: str, task_suite, num_tasks_in_suite: int) -> None:
    """Run the full experiment for a single latency value."""
    # Create timestamped output directories to avoid overwriting old data
    timestamp_folder = f"{timestamp}_rtc{args.use_rtc}_lat{latency_value}_hrzn{args.action_horizon}_parallel"
    horizon_folder = f"horizon_{args.action_horizon}"
    
    video_out_path_with_horizon = pathlib.Path(args.video_out_path) / timestamp_folder / horizon_folder
    results_csv_path_with_horizon = pathlib.Path(args.results_csv_path).parent / timestamp_folder / horizon_folder / pathlib.Path(args.results_csv_path).name
    
    video_out_path_with_horizon.mkdir(parents=True, exist_ok=True)
    results_csv_path_with_horizon.parent.mkdir(parents=True, exist_ok=True)
    
    # Log output paths
    logging.info(f"=" * 80)
    logging.info(f"Running experiment with latency: {latency_value} ms")
    logging.info(f"=" * 80)
    logging.info(f"Videos will be saved to: {video_out_path_with_horizon}")
    logging.info(f"Results will be saved to: {results_csv_path_with_horizon}")
    logging.info(f"Number of workers: {args.num_workers}")
    logging.info(f"RTC setting: {args.use_rtc}, action horizon: {args.action_horizon}")

    # Temporarily set args.latency_ms to the current latency value for workers
    original_latency = args.latency_ms
    args.latency_ms = latency_value
    
    # Initialize results tracking
    all_results_data = []

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
    elif args.task_suite_name == "moving_bowl":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):

        if task_id != 8:
            continue
            
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Start episodes with multiprocessing
        task_episodes, task_successes = 0, 0

        # Create shared state for worker status tracking
        with Manager() as manager:
            status_dict = manager.dict()
            results_dict = manager.dict()

            with Pool(
                processes=args.num_workers,
                initializer=init_worker,
                initargs=(task, args, status_dict, results_dict),
            ) as pool:
                # Create task arguments for each episode
                task_args_list = [
                    (task_id, initial_states[episode_idx], episode_idx, max_steps, video_out_path_with_horizon)
                    for episode_idx in range(args.num_trials_per_task)
                ]

                # Start background thread to monitor worker status
                stop_event = threading.Event()
                monitor_thread = threading.Thread(
                    target=monitor_worker_status,
                    args=(status_dict, stop_event, args.num_workers),
                )
                monitor_thread.start()

                try:
                    # Execute episodes in parallel and collect results
                    results = pool.imap(_eval_libero_wrapper, task_args_list)
                    
                    # Collect results
                    episode_results = list(results)
                    all_results_data.extend(episode_results)

                    # Count successes
                    task_episodes = len(episode_results)
                    task_successes = sum(1 for r in episode_results if r['success'])
                    total_episodes += task_episodes
                    total_successes += task_successes
                    
                finally:
                    # Stop monitoring thread
                    stop_event.set()
                    monitor_thread.join()

        # Log task results
        logging.info(
            f"Task {task_id} success rate: {task_successes}/{task_episodes} ({task_successes / task_episodes * 100:.1f}%)"
        )
        logging.info(
            f"Cumulative success rate: {total_successes}/{total_episodes} ({total_successes / total_episodes * 100:.1f}%)"
        )

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # Save results to CSV
    if all_results_data:
        fieldnames = all_results_data[0].keys()
        with open(results_csv_path_with_horizon, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results_data)
        
        logging.info(f"Results saved to: {results_csv_path_with_horizon}")
        
        # Calculate and save summary statistics
        summary_csv_path = str(results_csv_path_with_horizon).replace('.csv', '_summary.csv')
        task_summaries = []
        
        # Calculate per-task success rates
        for task_id in range(num_tasks_in_suite):
            task_results = [r for r in all_results_data if r['task_id'] == task_id]
            if task_results:
                task_successes = sum(1 for r in task_results if r['success'])
                task_episodes = len(task_results)
                task_success_rate = task_successes / task_episodes
                
                task_summary = {
                    'task_id': task_id,
                    'task_description': task_results[0]['task_description'],
                    'total_episodes': task_episodes,
                    'successes': task_successes,
                    'success_rate': task_success_rate,
                    'task_suite': args.task_suite_name,
                    'seed': args.seed,
                    'use_rtc': args.use_rtc,
                    'action_horizon': args.action_horizon,
                    'latency_ms': latency_value,
                    'avg_inference_time': np.mean([r['avg_inference_time'] for r in task_results]),
                    'avg_additional_delay_time': np.mean([r['avg_additional_delay_time'] for r in task_results])
                }
                task_summaries.append(task_summary)
        
        # Add overall summary
        overall_summary = {
            'task_id': 'OVERALL',
            'task_description': 'All tasks combined',
            'total_episodes': total_episodes,
            'successes': total_successes,
            'success_rate': total_successes / total_episodes if total_episodes > 0 else 0,
            'task_suite': args.task_suite_name,
            'seed': args.seed,
            'use_rtc': args.use_rtc,
            'action_horizon': args.action_horizon,
            'latency_ms': latency_value,
            'avg_inference_time': np.mean([r['avg_inference_time'] for r in all_results_data]),
            'avg_additional_delay_time': np.mean([r['avg_additional_delay_time'] for r in all_results_data])
        }
        task_summaries.append(overall_summary)
        
        # Save summary CSV
        if task_summaries:
            with open(summary_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=task_summaries[0].keys())
                writer.writeheader()
                writer.writerows(task_summaries)
            
            logging.info(f"Summary results saved to: {summary_csv_path}")
    
    # Restore original latency value
    args.latency_ms = original_latency


def main(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    latency_values = [float(lat) for lat in args.latency_ms]
    
    logging.info(f"Running experiments with latency values: {latency_values}")
    
    # Run experiment for each latency value
    for latency_value in latency_values:
        run_experiment_for_latency(args, latency_value, timestamp, task_suite, num_tasks_in_suite)


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
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
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
    multiprocessing.set_start_method("spawn")
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)

