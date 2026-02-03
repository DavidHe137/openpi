"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""

import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
import cv2
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.python_utils as PyUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
import robosuite.utils.camera_utils as CameraUtils
import robosuite.utils.transform_utils as T

try:
    import mujoco
except ImportError:
    mujoco = None


marker_params = []
RENDER_SIZE = 256


def _get_eef_pose_from_obs(obs):
    # Support both robosuite v1 (prefixed keys) and older versions
    pos_key = None
    quat_key = None
    for k in obs.keys():
        if k.endswith("eef_pos"):
            pos_key = k
        if k.endswith("eef_quat"):
            quat_key = k
    if pos_key is None and ("eef_pos" in obs):
        pos_key = "eef_pos"
    if quat_key is None and ("eef_quat" in obs):
        quat_key = "eef_quat"
    eef_pos = obs[pos_key] if pos_key is not None else None
    eef_quat = obs[quat_key] if quat_key is not None else None
    if eef_pos is None or eef_quat is None:
        return None, None
    # quaternion to rotation matrix
    eef_rot = T.quat2mat(eef_quat)
    return eef_pos, eef_rot


def _denormalize_action_sequence_if_needed(policy_wrapper, action_seq_np):
    # action_seq_np: [T, Da] in numpy
    stats = getattr(policy_wrapper, "action_normalization_stats", None)
    if stats is None:
        return action_seq_np
    # Use the same conversion as RolloutPolicy.__call__
    action_keys = policy_wrapper.policy.global_config.train.action_keys
    action_shapes = {k: stats[k]["offset"].shape[1:] for k in stats}
    # vector -> dict (per-timestep)
    denorm = []
    for a in action_seq_np:
        ac_dict = PyUtils.vector_to_action_dict(
            a, action_shapes=action_shapes, action_keys=action_keys
        )
        ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=stats)
        denorm.append(PyUtils.action_dict_to_vector(ac_dict, action_keys=action_keys))
    return np.stack(denorm, axis=0)


def _integrate_eef_deltas_world(eef_pos, eef_rot, action_seq_np, action_scale=1.0):
    # Assume first 3 dims are delta xyz in EEF frame (OSC delta pose). Ignore rotation for drawing.
    # Transform to world using current EEF rotation as approximation.
    points = [eef_pos.copy()]
    curr = eef_pos.copy()
    for a in action_seq_np:
        if a.shape[0] < 3:
            break
        d_eef = a[:3]
        d_world = eef_rot @ d_eef
        curr = curr + d_world
        curr = curr + d_eef * action_scale
        points.append(curr.copy())
    return np.stack(points, axis=0)  # [N,3]


def add_action_traj_to_video(traj_all, frames, world_to_camera, action_traj_length=8):
    # Using rollout as action traj to add up to the video (not easy to get the delta action traj directly)
    # traj: list of len N, with eef pos traj
    # frames: list of len N, with frame images
    # world_to_camera: world to camera transform matrix

    total_frames = len(frames)
    action_chunk_num = len(traj_all) // action_traj_length
    remainder_start = action_chunk_num * action_traj_length

    def process_frame(frame_idx, chunk_idx, obj_pixels, is_new_chunk=False):
        """Helper function to process a single frame with annotations"""
        if frame_idx >= total_frames:
            return

        frame = frames[frame_idx].copy()
        h, w = frame.shape[:2]

        # Visual indicator for new action chunk (first frame of chunk)
        if is_new_chunk:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 4)

        if obj_pixels is not None:
            for px in obj_pixels:
                x = int(px[1])
                y = int(px[0])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

        # Add text annotations: step number and action chunk index
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background for better visibility

        # Step number text
        step_text = f"Step: {frame_idx}"
        (text_width, text_height), baseline = cv2.getTextSize(
            step_text, font, font_scale, thickness
        )
        cv2.rectangle(
            frame, (5, 5), (10 + text_width, 10 + text_height + baseline), bg_color, -1
        )
        cv2.putText(
            frame,
            step_text,
            (5, 10 + text_height),
            font,
            font_scale,
            text_color,
            thickness,
        )

        # Action chunk index text
        chunk_text = f"Chunk: {chunk_idx}"
        (text_width2, text_height2), baseline2 = cv2.getTextSize(
            chunk_text, font, font_scale, thickness
        )
        cv2.rectangle(
            frame,
            (5, 15 + text_height + baseline),
            (10 + text_width2, 20 + text_height + baseline + text_height2 + baseline2),
            bg_color,
            -1,
        )
        cv2.putText(
            frame,
            chunk_text,
            (5, 20 + text_height + baseline + text_height2),
            font,
            font_scale,
            text_color,
            thickness,
        )

        frames[frame_idx] = frame

    # Process complete chunks
    for i in range(action_chunk_num):
        start_idx = i * action_traj_length
        end_idx = (i + 1) * action_traj_length
        eef_pos = traj_all[start_idx:end_idx]
        eef_pos = np.stack(eef_pos, axis=0)

        obj_pixels = CameraUtils.project_points_from_world_to_camera(
            points=eef_pos,
            world_to_camera_transform=world_to_camera,
            camera_height=RENDER_SIZE,
            camera_width=RENDER_SIZE,
        )

        for j in range(action_traj_length):
            frame_idx = start_idx + j
            process_frame(frame_idx, i, obj_pixels, is_new_chunk=(j == 0))

    # Handle remaining partial chunk
    if remainder_start < len(traj_all) and remainder_start < total_frames:
        eef_pos = traj_all[remainder_start:]
        eef_pos = np.stack(eef_pos, axis=0)

        obj_pixels = CameraUtils.project_points_from_world_to_camera(
            points=eef_pos,
            world_to_camera_transform=world_to_camera,
            camera_height=RENDER_SIZE,
            camera_width=RENDER_SIZE,
        )

        remainder_len = min(len(eef_pos), total_frames - remainder_start)
        for j in range(remainder_len):
            frame_idx = remainder_start + j
            process_frame(
                frame_idx, action_chunk_num, obj_pixels, is_new_chunk=(j == 0)
            )

    return frames


def rollout(
    policy,
    env,
    horizon,
    render=False,
    video_writer=None,
    video_skip=5,
    return_obs=False,
    camera_names=None,
    n_obj_steps=10,
    obj_step=0.01,
):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        img_size (int): size of the image to render
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu.
            They are excluded by default because the low-dimensional simulation states should be a minimal
            representation of the environment.
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()

    world_to_camera = CameraUtils.get_camera_transform_matrix(
        sim=env.env.sim,
        camera_name="agentview",
        camera_height=RENDER_SIZE,
        camera_width=RENDER_SIZE,
    )

    # env.env.sim._render_context_offscreen.render = types.MethodType(render_marker, env.env.sim._render_context_offscreen)
    # Ensure offscreen render override is bound after reset when context exists
    # try:
    #     sim = None
    #     if hasattr(env, 'env') and hasattr(env.env, 'sim'):
    #         sim = env.env.sim
    #     elif hasattr(env, 'base_env') and hasattr(env.base_env, 'sim'):
    #         sim = env.base_env.sim
    #     if sim is not None and hasattr(sim, '_render_context_offscreen') and sim._render_context_offscreen is not None:
    #         sim._render_context_offscreen.render = types.MethodType(render, sim._render_context_offscreen)
    #         # clear any previous markers
    #         marker_params.clear()
    # except Exception:
    #     pass
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    video_count = 0  # video frame counter
    total_reward = 0.0
    traj = dict(
        actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict
    )
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        N = n_obj_steps  # 10
        obj_step = obj_step  # 0.01
        obj_dir = 1
        MIN_Y, MAX_Y = -0.03, 0.03
        OBJ_Y_IDX = 11

        obj_offset_accum = 0.0

        traj_all = []
        img_all = []

        for step_i in range(horizon):
            do_move = step_i > 0 and step_i % N == 0

            if do_move:
                curr_obj = obs["object"].copy()
                step_dy = obj_step * obj_dir

                if curr_obj[1] + step_dy > MAX_Y or curr_obj[1] + step_dy < MIN_Y:
                    obj_dir *= -1
                    step_dy = np.clip(step_dy, MIN_Y - curr_obj[1], MAX_Y - curr_obj[1])
                curr_state = env.get_state()
                curr_state["states"][OBJ_Y_IDX] += step_dy
                obj_offset_accum += step_dy
                obs = env.reset_to(curr_state)
                state_dict = env.get_state()

                # print(f"curr obj {curr_obj}, curr state {curr_state['states'][10:13]}")
            else:
                step_dy = 0.0

            # TODO: zhenyang fix action chunk handling
            # reset object offset when new inference starts
            if len(policy.policy.action_queue) == 0:
                obj_offset_accum = 0.0

            # if len(policy.policy.action_queue) > 0:
            #     action_chunk = policy.policy.action_queue

            # get action from policy
            act = policy(ob=obs)

            # TODO: no object offest
            # act[1] += obj_offset_accum

            # # modify action to account for object movement
            # pred_step = step_dy * 2
            # new_act = act[1] + pred_step

            # if new_act > MAX_Y:
            #     # exceed upper bound, reflect excess
            #     act[1] = MAX_Y - (new_act - MAX_Y)
            # elif new_act < MIN_Y:
            #     # exceed lower bound, reflect excess
            #     act[1] = MIN_Y + (MIN_Y - new_act)
            # else:
            #     act[1] = new_act

            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    # Prepare markers before rendering (compute once per frame)
                    # world_pts = None

                    eef_pos, eef_rot = _get_eef_pose_from_obs(obs)
                    traj_all.append(eef_pos)

                    frame = env.render(
                        mode="rgb_array",
                        height=RENDER_SIZE,
                        width=RENDER_SIZE,
                        camera_name="agentview",
                    )
                    frame = np.ascontiguousarray(frame)
                    video_img.append(frame)
                    video_img = np.concatenate(
                        video_img, axis=1
                    )  # concatenate horizontally # NOTE assume different camera
                    img_all.append(video_img)
                    # video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

        if video_writer is not None:
            world_to_camera = CameraUtils.get_camera_transform_matrix(
                sim=env.env.sim,
                camera_name="agentview",
                camera_height=RENDER_SIZE,
                camera_width=RENDER_SIZE,
            )

            # breakpoint()
            img_all = add_action_traj_to_video(
                traj_all, img_all, world_to_camera, action_traj_length=8
            )
            for img in img_all:
                video_writer.append_data(img)

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
            traj["next_obs"]
        )

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    # some arg checking
    write_video = args.video_path is not None
    assert not (args.render and write_video)  # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=True
    )

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=args.render,
        render_offscreen=(args.video_path is not None),
        verbose=True,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = args.dataset_path is not None
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in range(rollout_num_episodes):
        print(f"Starting rollout {i + 1}/{rollout_num_episodes}")
        stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            n_obj_steps=args.n_obj_steps,
            obj_step=args.obj_step,
        )
        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset(
                        "obs/{}".format(k), data=np.array(traj["obs"][k])
                    )
                    ep_data_grp.create_dataset(
                        "next_obs/{}".format(k), data=np.array(traj["next_obs"][k])
                    )

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"][
                    "model"
                ]  # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[
                0
            ]  # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {k: np.mean(rollout_stats[k]) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(
            env.serialize(), indent=4
        )  # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action="store_true",
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action="store_true",
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    parser.add_argument(
        "--n_obj_steps",
        type=int,
        default=10,
        help="number of object steps interval",
    )

    parser.add_argument(
        "--obj_step",
        type=float,
        default=0.01,
        help="object step size",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()
    run_trained_agent(args)
