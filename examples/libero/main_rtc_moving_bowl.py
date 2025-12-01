"""
Evaluation script for OpenPI VLA with moving bowl in LIBERO KITCHEN_SCENE1.
Task: put the black bowl on the plate (with the bowl moving periodically)

Features:
- Moving bowl: Moves periodically in Y direction (within MIN_Y to MAX_Y range)
  - Movement pauses when bowl is grasped or relocated to prevent physics glitches
  - Movement is based on simulation steps (independent of artificial latency)
- RTC support: Can run with or without Real-Time Control
  - RTC mode (use_rtc=True): Uses ActionChunkBroker for temporal ensembling
  - Non-RTC mode (use_rtc=False): Uses simple replanning every N steps (like main.py)
- Latency simulation: Can inject artificial latency (latency_ms parameter)
  - Bowl movement continues at normal rate during latency delays
  - Ensures consistent bowl behavior across different latency conditions

The bowl movement is automatically paused when:
1. The bowl is grasped by the gripper (detected via Z coordinate lift)
2. The bowl has been moved significantly from its starting position (e.g., placed on plate)
   - Threshold is dynamically set to 1.5x the maximum movement range

This prevents physics glitches and avoids teleporting the bowl back after placement.
Bowl movement timing is independent of artificial latency to ensure fair comparisons.
"""
import csv
import dataclasses
from datetime import datetime
import logging
import pathlib
import time
from typing import Optional

import imageio
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import torch
import tqdm
import tyro

# LIBERO imports
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8080
    resize_size: int = 224
    action_horizon: int = 10  # Action horizon for ActionChunkBroker (RTC) or action chunk size (non-RTC)
    replan_steps: int = 5  # When use_rtc=False, replan every N steps (similar to main.py)
    latency_ms: float = 0.0  # Artificial latency to inject during inference (in milliseconds)

    #################################################################################################################
    # Environment parameters
    #################################################################################################################
    bddl_file: str = "examples/libero/moving_obj_scene/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"  # Path to BDDL file
    init_file: str = "examples/libero/moving_obj_scene/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.init"  # Path to .init file with initial states
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials: int = 20  # Number of rollouts
    max_steps: int = 400  # Maximum steps per episode
    n_obj_steps: int = 1  # Move object every N steps
    obj_step: float = 0.01  # Object movement step size
    
    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "examples/libero/data/moving_bowl/videos"  # Path to save videos
    results_csv_path: str = "examples/libero/data/moving_bowl/results.csv"  # Path to save CSV results
    seed: int = 18  # Random Seed (for reproducibility)
    use_rtc: bool = False  # Use RTC mode


def _quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.
    Copied from robosuite.
    """
    import math
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


def create_libero_env_from_bddl(bddl_file: str, resolution: int, seed: int):
    """Create a LIBERO environment from a BDDL file."""
    bddl_file_path = pathlib.Path(bddl_file)
    if not bddl_file_path.is_absolute():
        # Relative path from current directory
        bddl_file_path = pathlib.Path.cwd() / bddl_file_path
    
    env_args = {
        "bddl_file_name": str(bddl_file_path),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    
    # Extract task description from BDDL file
    task_description = "put the black bowl on the plate"
    
    return env, task_description


def get_observation_dict(env_obs, task_description: str, resize_size: int = 224):
    """
    Convert raw environment observation to format expected by VLA.
    IMPORTANT: Rotate images 180 degrees to match training preprocessing.
    """
    # Get images and rotate 180 degrees
    agentview_img = np.ascontiguousarray(env_obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(env_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    
    # Resize with padding
    agentview_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agentview_img, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    
    # Extract robot state
    eef_pos = env_obs["robot0_eef_pos"]
    eef_quat = env_obs["robot0_eef_quat"]
    gripper_qpos = env_obs["robot0_gripper_qpos"]
    
    # Construct observation dict for VLA
    obs_dict = {
        "observation/image": agentview_img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (eef_pos, _quat2axisangle(eef_quat), gripper_qpos)
        ),
        "prompt": str(task_description),
    }
    
    return obs_dict


def find_bowl_state_index(env, verbose=False):
    """
    Find the state index for the akita_black_bowl object.
    In LIBERO/robosuite, object positions come after robot joint AND gripper positions.
    
    For Panda robot in LIBERO:
    - 7 robot arm joint positions (0-6)
    - 2 robot gripper joint positions (7-8) for left and right fingers
    - Then objects: each object has 7 values [x, y, z, qw, qx, qy, qz]
    - Bowl is typically the first object
    """
    # Get current state to inspect structure
    curr_state = env.sim.get_state()
    state_array = np.array(curr_state.qpos)
    
    if verbose:
        logging.info(f"State array length: {len(state_array)}")
        logging.info(f"All state values: {state_array}")
    
    # For LIBERO kitchen scene with Panda robot:
    # Indices 0-6: robot arm joints (7 values)
    # Indices 7-8: robot gripper joints (2 values for left/right fingers)
    # Indices 9-15: first object (bowl) - [x, y, z, qw, qx, qy, qz]
    # Bowl Y position is at index 10 (9 + 1)
    bowl_start_idx = 9  # Skip 7 arm joints + 2 gripper joints
    bowl_y_idx = bowl_start_idx + 1  # Y coordinate
    
    if verbose:
        logging.info(f"Robot arm joints (0-6): {state_array[0:7]}")
        logging.info(f"Robot gripper joints (7-8): {state_array[7:9]}")
        logging.info(f"Bowl state starts at index: {bowl_start_idx}")
        logging.info(f"Bowl Y position at index: {bowl_y_idx}")
        if len(state_array) > bowl_start_idx + 6:
            logging.info(f"Bowl XYZ: {state_array[bowl_start_idx:bowl_start_idx+3]}")
            logging.info(f"Bowl quaternion (qw,qx,qy,qz): {state_array[bowl_start_idx+3:bowl_start_idx+7]}")
    
    return bowl_y_idx


def is_bowl_grasped(env, bowl_idx_start=9, initial_bowl_z=None, z_lift_threshold=0.02):
    """
    Check if the bowl is currently grasped by the gripper.
    Uses Z coordinate - if bowl is lifted above its initial position, it's grasped.
    
    Args:
        env: Environment instance
        bowl_idx_start: Starting index of bowl in state array (default 9)
        initial_bowl_z: Initial Z position of bowl when on table (default None)
        z_lift_threshold: How much higher (in meters) the bowl needs to be to be considered grasped
    
    Returns:
        bool: True if bowl is grasped (lifted), False otherwise
    """
    # Get bowl position from simulator state
    curr_state = env.sim.get_state()
    state_array = np.array(curr_state.qpos)
    bowl_z = state_array[bowl_idx_start + 2]  # Z coordinate (index 2 after bowl_start)
    
    # If we don't have initial Z, use a reasonable table height for kitchen scene
    if initial_bowl_z is None:
        initial_bowl_z = -0.012  # Typical bowl Z position when on table in this scene
    
    # Bowl is grasped if it's been lifted significantly above initial position
    is_grasped = bowl_z > (initial_bowl_z + z_lift_threshold)
    
    return is_grasped


def is_bowl_moved_from_start(env, bowl_idx_start=9, initial_bowl_xy=None, distance_threshold=0.05):
    """
    Check if the bowl has been moved significantly from its starting position.
    This is used to stop moving the bowl once it's been placed elsewhere (e.g., on the plate).
    
    Args:
        env: Environment instance
        bowl_idx_start: Starting index of bowl in state array (default 9)
        initial_bowl_xy: Initial XY position of bowl (numpy array)
        distance_threshold: Distance threshold to consider bowl as moved (in meters)
                          Should be set larger than the periodic movement range
    
    Returns:
        bool: True if bowl has moved from start position, False otherwise
    """
    if initial_bowl_xy is None:
        return False
    
    # Get current bowl position
    curr_state = env.sim.get_state()
    state_array = np.array(curr_state.qpos)
    curr_bowl_xy = state_array[bowl_idx_start:bowl_idx_start+2]  # X, Y coordinates
    
    # Calculate distance from initial position
    distance = np.linalg.norm(curr_bowl_xy - initial_bowl_xy)
    
    # Bowl has moved if distance exceeds threshold
    return distance > distance_threshold


def move_bowl_if_needed(env, sim_step, n_obj_steps, obj_step, bowl_y_idx, min_y, max_y, 
                        obj_dir, initial_bowl_z, initial_bowl_xy, relocation_threshold,
                        episode_idx, log_grasped=False, bowl_relocated_logged=False):
    """
    Move the bowl periodically based on simulation steps (not policy steps).
    This ensures bowl movement is independent of artificial latency.
    
    Args:
        env: Environment instance
        sim_step: Current simulation step counter
        n_obj_steps: Move object every N simulation steps
        obj_step: Object movement step size
        bowl_y_idx: Index of bowl Y position in state array
        min_y, max_y: Y position bounds for bowl movement
        obj_dir: Current direction of movement (1 or -1)
        initial_bowl_z: Initial Z position of bowl (for grasp detection)
        initial_bowl_xy: Initial XY position of bowl (for relocation detection)
        relocation_threshold: Threshold distance to consider bowl relocated
        episode_idx: Current episode index (for logging)
        log_grasped: Whether to log when bowl is grasped (only for first episode)
        bowl_relocated_logged: Whether relocation has already been logged
    
    Returns:
        tuple: (new_obj_dir, step_dy, new_bowl_relocated_logged)
               new_obj_dir: Updated direction
               step_dy: How much the bowl moved in Y
               new_bowl_relocated_logged: Updated relocation logged flag
    """
    # Only move at specified intervals
    if sim_step == 0 or sim_step % n_obj_steps != 0:
        return obj_dir, 0.0, bowl_relocated_logged
    
    # Check if bowl is currently grasped (lifted off table)
    bowl_grasped = is_bowl_grasped(env, bowl_idx_start=9, initial_bowl_z=initial_bowl_z)
    
    # Check if bowl has been moved from starting position (e.g., placed on plate)
    bowl_relocated = is_bowl_moved_from_start(env, bowl_idx_start=9, initial_bowl_xy=initial_bowl_xy, 
                                              distance_threshold=relocation_threshold)
    
    if bowl_grasped and log_grasped and sim_step < 200:
        # Log when bowl movement is skipped (only for first episode and early steps)
        logging.info(f"Episode {episode_idx}, sim_step {sim_step}: Skipping bowl movement (bowl is lifted)")
    
    if bowl_relocated and episode_idx == 0 and not bowl_relocated_logged:
        # Log once when bowl has been relocated (task likely complete or in progress)
        logging.info(f"Episode {episode_idx}, sim_step {sim_step}: Bowl has been moved from start position, stopping movement")
        bowl_relocated_logged = True
    
    if not bowl_grasped and not bowl_relocated:
        # Get current state
        curr_state = env.sim.get_state()
        state_array = np.array(curr_state.qpos)
        
        # Get current bowl Y position
        curr_bowl_y = state_array[bowl_y_idx]
        step_dy = obj_step * obj_dir
        
        # Check bounds and reverse direction if needed
        if curr_bowl_y + step_dy > max_y or curr_bowl_y + step_dy < min_y:
            obj_dir *= -1
            step_dy = np.clip(step_dy, min_y - curr_bowl_y, max_y - curr_bowl_y)
        
        # Move bowl by modifying simulator state
        state_array[bowl_y_idx] += step_dy
        curr_state.qpos[:] = state_array
        env.sim.set_state(curr_state)
        env.sim.forward()
        
        return obj_dir, step_dy, bowl_relocated_logged
    
    return obj_dir, 0.0, bowl_relocated_logged


def eval_moving_bowl(
    env,
    policy_client: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    episode_idx: int,
    video_out_path: pathlib.Path,
    task_description: str,
    initial_states: np.ndarray,
    action_chunk_broker_params: dict = None,
) -> dict:
    """Run a single episode evaluation with moving bowl."""
    
    # Reset environment
    env.reset()
    
    # Set initial states
    env_obs = env.set_init_state(initial_states)
    
    # Setup action planning based on RTC mode
    import collections
    if args.use_rtc:
        # Use ActionChunkBroker for RTC mode
        broker = action_chunk_broker.ActionChunkBroker(
            policy=policy_client,
            **action_chunk_broker_params
        )
        broker.reset()
        action_plan = None
    else:
        # Use simple deque for non-RTC mode (like main.py)
        broker = None
        action_plan = collections.deque()
    
    # Setup for object movement
    obj_dir = 1
    MIN_Y, MAX_Y = -0.05, 0.05
    # Find bowl Y index (verbose for first episode only)
    BOWL_Y_IDX = find_bowl_state_index(env, verbose=(episode_idx == 0))
    obj_offset_accum = 0.0
    
    # Calculate relocation threshold: should be larger than the max movement range
    # to avoid falsely detecting bowl movement as relocation
    movement_range = max(abs(MIN_Y), abs(MAX_Y))
    RELOCATION_THRESHOLD = movement_range * 1.5  # 1.5x the max movement to be safe
    
    # Will be set after objects stabilize
    INITIAL_BOWL_Z = None
    INITIAL_BOWL_XY = None
    bowl_relocated_logged = False  # Track if we've logged bowl relocation
    
    # Tracking
    t = 0  # Policy step counter
    sim_step = 0  # Actual simulation step counter (independent of latency)
    replay_images = []
    last_action = None
    success = False
    total_reward = 0.0
    
    inference_times = []
    additional_delay_times = []
    
    while t < args.max_steps + args.num_steps_wait:
        try:
            # Wait for objects to stabilize
            if t < args.num_steps_wait:
                env_obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                sim_step += 1
                continue
            
            # Record initial bowl position after objects have stabilized
            if INITIAL_BOWL_Z is None:
                curr_state = env.sim.get_state()
                state_array = np.array(curr_state.qpos)
                INITIAL_BOWL_Z = state_array[9 + 2]  # Bowl Z coordinate (index 9 + 2)
                INITIAL_BOWL_XY = state_array[9:9+2].copy()  # Bowl X, Y coordinates
                
                if episode_idx == 0:
                    logging.info(f"Initial bowl position (after stabilization): XY={INITIAL_BOWL_XY}, Z={INITIAL_BOWL_Z}")
                    logging.info(f"Bowl will be considered grasped if Z > {INITIAL_BOWL_Z + 0.02}")
                    logging.info(f"Bowl movement range: Y in [{MIN_Y}, {MAX_Y}]")
                    logging.info(f"Bowl movement will stop if moved > {RELOCATION_THRESHOLD:.3f}m from start position")
                    logging.info(f"Bowl movement based on sim_step (independent of artificial latency)")
            
            # Move bowl periodically based on simulation steps (not policy steps)
            # This ensures bowl movement is independent of artificial latency
            obj_dir, step_dy, bowl_relocated_logged = move_bowl_if_needed(
                env, sim_step, args.n_obj_steps, args.obj_step, BOWL_Y_IDX,
                MIN_Y, MAX_Y, obj_dir, INITIAL_BOWL_Z, INITIAL_BOWL_XY,
                RELOCATION_THRESHOLD, episode_idx, log_grasped=(episode_idx == 0),
                bowl_relocated_logged=bowl_relocated_logged
            )
            
            if step_dy != 0.0:
                obj_offset_accum += step_dy
                # Take a dummy step to refresh observations after moving the bowl
                env_obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                sim_step += 1
                total_reward += reward
                if done:
                    success = env.check_success()
                    break
            
            # Get preprocessed observation for VLA
            obs_dict = get_observation_dict(env_obs, task_description, args.resize_size)
            
            # Query model to get action (different logic for RTC vs non-RTC)
            inference_start = time.time()
            if args.use_rtc:
                # RTC mode: use ActionChunkBroker
                action = broker.infer(obs_dict)["actions"]
            else:
                # Non-RTC mode: use action plan with replanning
                if not action_plan:
                    # Need to replan - query model for new action chunk
                    action_chunk = policy_client.infer(obs_dict)["actions"]
                    assert len(action_chunk) >= args.replan_steps, \
                        f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[:args.replan_steps])
                
                # Pop next action from plan
                action = action_plan.popleft()
            
            actual_inference_time = time.time() - inference_start
            inference_times.append(actual_inference_time)
            
            # Calculate additional delay needed
            target_latency_seconds = args.latency_ms / 1000.0
            additional_delay = max(0, target_latency_seconds - actual_inference_time)
            additional_delay_times.append(additional_delay)
            
            # Simulate the additional delay by repeating last action
            # IMPORTANT: Bowl movement happens here too to maintain consistent movement regardless of latency
            if additional_delay > 0:
                delay_steps = int(additional_delay / 0.05)  # Assuming 20Hz
                for _ in range(delay_steps):
                    # Increment sim_step counter
                    sim_step += 1
                    
                    # Move bowl if needed (maintains consistent movement during delay)
                    obj_dir, step_dy, bowl_relocated_logged = move_bowl_if_needed(
                        env, sim_step, args.n_obj_steps, args.obj_step, BOWL_Y_IDX,
                        MIN_Y, MAX_Y, obj_dir, INITIAL_BOWL_Z, INITIAL_BOWL_XY,
                        RELOCATION_THRESHOLD, episode_idx, log_grasped=False,
                        bowl_relocated_logged=bowl_relocated_logged
                    )
                    if step_dy != 0.0:
                        obj_offset_accum += step_dy
                    
                    # Save frame during delay to show pause in video
                    replay_images.append(obs_dict["observation/image"])
                    if last_action is not None:
                        env_obs, reward, done, info = env.step(last_action.tolist())
                        total_reward += reward
                    else:
                        env_obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        total_reward += reward
                    if done:
                        success = env.check_success()
                        break
                if done:
                    break
            
            # Save current frame for replay video (after delay)
            replay_images.append(obs_dict["observation/image"])
            
            # Execute action in environment
            env_obs, reward, done, info = env.step(action.tolist())
            last_action = action
            total_reward += reward
            sim_step += 1  # Increment simulation step counter
            
            if done:
                success = env.check_success()
                break
            
            t += 1  # Increment policy step counter
            
        except Exception as e:
            logging.error(f"Episode {episode_idx} caught exception: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Save video
    suffix = "succ" if success else "fail"
    video_filename = video_out_path / f"ep{episode_idx}_rtc{args.use_rtc}_hrzn{args.action_horizon}_{suffix}.mp4"
    imageio.mimwrite(
        video_filename,
        [np.asarray(x) for x in replay_images],
        fps=10,
    )
    
    logging.info(f"Episode {episode_idx}: {'SUCCESS' if success else 'FAIL'} - {t} steps, reward: {total_reward:.3f}")
    
    # Prepare result dict
    episode_result = {
        'episode_idx': episode_idx,
        'success': bool(success),
        'steps_taken': t,
        'max_steps': args.max_steps + args.num_steps_wait,
        'total_reward': float(total_reward),
        'task_description': task_description,
        'seed': args.seed,
        'use_rtc': args.use_rtc,
        'action_horizon': args.action_horizon,
        'replan_steps': args.replan_steps if not args.use_rtc else None,
        'latency_ms': args.latency_ms,
        'n_obj_steps': args.n_obj_steps,
        'obj_step': args.obj_step,
        'avg_inference_time': np.mean(inference_times) if inference_times else 0.0,
        'avg_additional_delay_time': np.mean(additional_delay_times) if additional_delay_times else 0.0,
    }
    
    return episode_result


def load_initial_states(init_file_path: str):
    """Load initial states from .init file."""
    init_file_path = pathlib.Path(init_file_path)
    if not init_file_path.is_absolute():
        # Relative path from current directory
        init_file_path = pathlib.Path.cwd() / init_file_path
    
    # Load the initial states using torch.load (LIBERO format)
    try:
        # Try with weights_only=False for newer PyTorch versions
        initial_states = torch.load(str(init_file_path), map_location='cpu', weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only parameter
        initial_states = torch.load(str(init_file_path), map_location='cpu')
    return initial_states


def main(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create timestamped output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.use_rtc:
        timestamp_folder = f"{timestamp}_rtc{args.use_rtc}_lat{args.latency_ms}_hrzn{args.action_horizon}_nobj{args.n_obj_steps}_replan_na"
    else:
        timestamp_folder = f"{timestamp}_rtc{args.use_rtc}_lat{args.latency_ms}_hrzn{args.action_horizon}_nobj{args.n_obj_steps}_replan{args.replan_steps}"

    video_out_path = pathlib.Path(args.video_out_path) / timestamp_folder
    results_csv_path = pathlib.Path(args.results_csv_path).parent / timestamp_folder / pathlib.Path(args.results_csv_path).name
    
    video_out_path.mkdir(parents=True, exist_ok=True)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env, task_description = create_libero_env_from_bddl(args.bddl_file, LIBERO_ENV_RESOLUTION, args.seed)
    
    # Log output paths
    logging.info(f"=" * 80)
    logging.info(f"Moving Bowl Evaluation - LIBERO KITCHEN_SCENE1")
    logging.info(f"=" * 80)
    logging.info(f"Task: {task_description}")
    logging.info(f"BDDL file: {args.bddl_file}")
    logging.info(f"Init file: {args.init_file}")
    logging.info(f"Videos will be saved to: {video_out_path}")
    logging.info(f"Results will be saved to: {results_csv_path}")
    if args.use_rtc:
        logging.info(f"Mode: RTC enabled, action horizon: {args.action_horizon}")
    else:
        logging.info(f"Mode: RTC disabled, action horizon: {args.action_horizon}, replan every {args.replan_steps} steps")
    logging.info(f"Object movement: every {args.n_obj_steps} steps, step size: {args.obj_step}")
    
    # Load initial states from .init file
    logging.info("Loading initial states from .init file...")
    initial_states_list = load_initial_states(args.init_file)
    logging.info(f"Loaded {len(initial_states_list)} initial states")
    
    # Use only the number of trials requested
    if args.num_trials > len(initial_states_list):
        logging.warning(f"Requested {args.num_trials} trials but only {len(initial_states_list)} initial states available. Using {len(initial_states_list)} trials.")
        args.num_trials = len(initial_states_list)
    else:
        initial_states_list = initial_states_list[:args.num_trials]
    
    # Initialize websocket policy client
    policy_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port, latency_ms=args.latency_ms)
    
    # Prepare ActionChunkBroker parameters (only used if RTC mode is enabled)
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
    
    broker_params = {
        'action_horizon': args.action_horizon,
        'is_rtc': args.use_rtc,
        's': s,
        'd': d,
    }
    
    # Run evaluation episodes
    all_results = []
    total_successes = 0
    
    for episode_idx in tqdm.tqdm(range(args.num_trials), desc="Running episodes"):
        result = eval_moving_bowl(
            env=env,
            policy_client=policy_client,
            args=args,
            episode_idx=episode_idx,
            video_out_path=video_out_path,
            task_description=task_description,
            initial_states=initial_states_list[episode_idx],
            action_chunk_broker_params=broker_params,
        )
        all_results.append(result)
        if result['success']:
            total_successes += 1
    
    # Calculate statistics
    success_rate = total_successes / args.num_trials if args.num_trials > 0 else 0.0
    
    logging.info(f"=" * 80)
    logging.info(f"Evaluation complete!")
    logging.info(f"Success rate: {total_successes}/{args.num_trials} ({success_rate * 100:.1f}%)")
    logging.info(f"=" * 80)
    
    # Save results to CSV
    if all_results:
        fieldnames = all_results[0].keys()
        with open(results_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        logging.info(f"Results saved to: {results_csv_path}")
        
        # Save summary
        summary = {
            'task_description': task_description,
            'total_episodes': args.num_trials,
            'successes': total_successes,
            'success_rate': success_rate,
            'seed': args.seed,
            'use_rtc': args.use_rtc,
            'action_horizon': args.action_horizon,
            'replan_steps': args.replan_steps if not args.use_rtc else None,
            'latency_ms': args.latency_ms,
            'n_obj_steps': args.n_obj_steps,
            'obj_step': args.obj_step,
            'avg_inference_time': np.mean([r['avg_inference_time'] for r in all_results]),
            'avg_additional_delay_time': np.mean([r['avg_additional_delay_time'] for r in all_results]),
            'avg_steps': np.mean([r['steps_taken'] for r in all_results]),
            'avg_reward': np.mean([r['total_reward'] for r in all_results]),
        }
        
        summary_csv_path = str(results_csv_path).replace('.csv', '_summary.csv')
        with open(summary_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)
        
        logging.info(f"Summary saved to: {summary_csv_path}")
    
    # Clean up
    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)

