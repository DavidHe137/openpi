import collections
import dataclasses
import logging
import pathlib

import imageio
from libero.libero import benchmark
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from examples.libero import visualize
from examples.libero import utils
from examples.libero import logging_config

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8080
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    #################################################################################################################
    # Visualization parameters
    #################################################################################################################
    visualize_chunks: bool = True  # Whether to overlay action chunk visualization
    viz_time_window: int = 10  # Number of timesteps to show before/after playhead


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = TASK_SUITE_MAX_STEPS.get(args.task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(
        robot_id="robot_0",
        host=args.host,
        port=args.port,
    )

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = utils._get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            # Tracking data for action chunk visualization
            frame_metadata = []  # List of ActionFrameMetadata
            current_chunk_id = 0
            active_chunk_id = None  # ID of the chunk currently being executed

            logging.info(f"Starting episode {task_episodes + 1}...")
            pbar = tqdm.tqdm(
                total=max_steps + args.num_steps_wait,
                desc=f"Episode {task_episodes + 1}",
            )
            done = False
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
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            img, args.resize_size, args.resize_size
                        )
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
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    utils._quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                        # Track new chunk prediction
                        active_chunk_id = current_chunk_id
                        current_chunk_id += 1
                    assert active_chunk_id is not None, "active_chunk_id is not set"

                    action = action_plan.popleft()

                    # Track action execution for visualization
                    action_index = args.replan_steps - len(action_plan) - 1
                    frame_metadata.append(
                        visualize.ActionFrameMetadata(
                            timestep=t,
                            chunk_id=active_chunk_id,
                            action_index=action_index,
                        )
                    )

                    # Execute action in environment
                    obs, reward, success, info = env.step(action.tolist())
                    if success:
                        task_successes += 1
                        total_successes += 1
                        done = True
                        break
                    t += 1
                    pbar.update(1)

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # client.save_data()

            # Apply visualization overlay to frames
            if args.visualize_chunks:
                visualized_frames = visualize.add_action_chunk_visualization(
                    replay_images,
                    frame_metadata,
                    replan_steps=args.replan_steps,
                    time_window=args.viz_time_window,
                )
            else:
                visualized_frames = replay_images

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in visualized_frames],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        # Log final results
        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    logging_config.setup_logging()
    tyro.cli(eval_libero)
