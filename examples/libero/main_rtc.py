import csv
import dataclasses
from datetime import datetime
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import time

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8081
    resize_size: int = 224
    action_horizon: int = 10  # Action horizon for ActionChunkBroker (matches Libero model config)
    latency_ms: float = 0.0  # Artificial latency to inject during inference (in milliseconds)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 20  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_csv_path: str = "data/libero/results.csv"  # Path to save CSV results

    seed: int = 18  # Random Seed (for reproducibility)

    use_rtc: bool = False


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # Create timestamped output directories to avoid overwriting old data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_folder = f"{timestamp}_rtc{args.use_rtc}_lat{args.latency_ms}_hrzn{args.action_horizon}"
    horizon_folder = f"horizon_{args.action_horizon}"
    
    video_out_path_with_horizon = pathlib.Path(args.video_out_path) / timestamp_folder / horizon_folder
    results_csv_path_with_horizon = pathlib.Path(args.results_csv_path).parent / timestamp_folder / horizon_folder / pathlib.Path(args.results_csv_path).name
    
    video_out_path_with_horizon.mkdir(parents=True, exist_ok=True)
    results_csv_path_with_horizon.parent.mkdir(parents=True, exist_ok=True)
    
    # Log output paths
    logging.info(f"Videos will be saved to: {video_out_path_with_horizon}")
    logging.info(f"Results will be saved to: {results_csv_path_with_horizon}")

    # Initialize results tracking
    results_data = []

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

    s = None
    d = None
    if args.action_horizon == 10:
        s = 5
        d = 4
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

    ws_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port, latency_ms=args.latency_ms)
    client = action_chunk_broker.ActionChunkBroker(
        policy=ws_client,
        action_horizon=args.action_horizon,
        is_rtc=args.use_rtc,
        s=s,
        d=d,
    )

    print("RTC setting: ", args.use_rtc, "action horizon: ", args.action_horizon, "s: ", s, "d: ", d)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if task_id == 1:
            continue
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            
            # Reset the action chunk broker for each episode
            client.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            last_gripper = -1.0

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
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

                    inference_start = time.time()
                    action = client.infer(element)["actions"]
                    actual_inference_time = time.time() - inference_start

                    # Calculate additional delay needed
                    target_latency_seconds = args.latency_ms / 1000.0
                    additional_delay = max(0, target_latency_seconds - actual_inference_time)

                    # Simulate the additional delay by repeating last action
                    if additional_delay > 0:
                        delay_steps = int(additional_delay / 0.05)  # Assuming 20Hz
                        for _ in range(delay_steps):
                            obs, reward, done, info = env.step([0.0] * 6 + [last_gripper])
                            t += 1
                            if done or t >= max_steps:
                                break

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    last_gripper = action.tolist()[6]
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Record episode results
            episode_result = {
                'task_id': task_id,
                'task_description': task_description,
                'episode_idx': episode_idx,
                'success': bool(done),
                'steps_taken': t,
                'max_steps': max_steps + args.num_steps_wait,
                'task_suite': args.task_suite_name,
                'seed': args.seed,
                'use_rtc': args.use_rtc,
                'action_horizon': args.action_horizon
            }
            results_data.append(episode_result)

            # Save a replay video of the episode
            suffix = "succ" if done else "fail"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                video_out_path_with_horizon / f"task_{task_id}_ep{episode_idx}_rtc{args.use_rtc}_hrzn{args.action_horizon}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # Save results to CSV
    if results_data:
        fieldnames = results_data[0].keys()
        with open(results_csv_path_with_horizon, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_data)
        
        logging.info(f"Results saved to: {results_csv_path_with_horizon}")
        
        # Calculate and save summary statistics
        summary_csv_path = str(results_csv_path_with_horizon).replace('.csv', '_summary.csv')
        task_summaries = []
        
        # Calculate per-task success rates
        for task_id in range(num_tasks_in_suite):
            task_results = [r for r in results_data if r['task_id'] == task_id]
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
                    'action_horizon': args.action_horizon
                }
                task_summaries.append(task_summary)
        
        # Add overall summary
        overall_summary = {
            'task_id': 'OVERALL',
            'task_description': 'All tasks combined',
            'total_episodes': total_episodes,
            'successes': total_successes,
            'success_rate': total_successes / total_episodes,
            'task_suite': args.task_suite_name,
            'seed': args.seed,
            'use_rtc': args.use_rtc,
            'action_horizon': args.action_horizon
        }
        task_summaries.append(overall_summary)
        
        # Save summary CSV
        if task_summaries:
            with open(summary_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=task_summaries[0].keys())
                writer.writeheader()
                writer.writerows(task_summaries)
            
            logging.info(f"Summary results saved to: {summary_csv_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
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
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
