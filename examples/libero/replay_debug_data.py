"""
Replay debug data from a saved episode.

This script loads saved debug data (observations, noise, actions) and replays them
in the LIBERO environment. There are two modes:

1. --use_saved_actions: Directly use the saved output_actions from debug data
2. Default: Send saved observation and noise to policy to reproduce actions

Usage:
    # Use saved actions directly (fastest, guaranteed deterministic)
    python examples/libero/replay_debug_data.py \
        --debug_data_dir data/libero/multi_robot_videos/0/0_libero_10_8_success \
        --use_saved_actions

    # Re-infer actions from policy with saved noise (verifies reproducibility)
    python examples/libero/replay_debug_data.py \
        --debug_data_dir data/libero/multi_robot_videos/0/0_libero_10_8_success \
        --host localhost --port 8080

    # Re-infer actions from policy in RTC mode (requires saved rtc params in debug data)
    python examples/libero/replay_debug_data.py \
        --debug_data_dir data/libero/multi_robot_videos_rtc/0/0_libero_10_0_success \
        --host localhost --port 8080 --use_rtc

The script will:
1. Load metadata to get task info
2. Load debug data chunks
3. For each chunk, either use saved actions or re-infer from policy
4. Execute the actions in the environment
5. Save a replay video and report success/failure
"""

import argparse
import csv
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
from libero.libero import benchmark
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from openpi_client import websocket_client_policy as _websocket_client_policy
from examples.libero import utils
from examples.libero.env import LiberoSimEnvironment

LIBERO_ENV_RESOLUTION = 256


@dataclass
class ReplayConfig:
    """Configuration for replay."""

    debug_data_dir: pathlib.Path
    host: str
    port: int
    seed: int = 7
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: int = 500
    control_hz: int = 20
    action_horizon: int = 10  # Number of actions per chunk (model's action horizon)
    action_dim: int = 7  # Actual robot action dimension (6 DoF + gripper for LIBERO)
    output_video: Optional[str] = None
    use_saved_actions: bool = False  # If True, use saved output_actions directly
    use_rtc: bool = False  # If True, use RTC inference mode (requires saved rtc params)
    return_debug_data: bool = (
        False  # If True, request debug payloads from policy (if supported)
    )
    debug_report_path: Optional[str] = (
        None  # Where to write per-chunk debug comparison report (jsonl)
    )
    plot_gt_horse_tails: bool = True  # Plot ground-truth discarded actions
    plot_pred_horse_tails: bool = False  # Plot predicted discarded actions
    output_html: Optional[str] = None  # Where to write interactive HTML plot
    no_html: bool = False  # If True, skip HTML output
    analyze_overlap: bool = True  # If True, compute action overlap/reuse statistics


def load_metadata(debug_data_dir: pathlib.Path) -> dict:
    """Load episode metadata."""
    metadata_path = debug_data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path) as f:
        return json.load(f)


def load_debug_chunks(debug_data_dir: pathlib.Path) -> List[dict]:
    """Load all debug data chunks in order."""
    chunk_dir = debug_data_dir / "debug_data"
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Debug data directory not found: {chunk_dir}")

    chunk_files = sorted(chunk_dir.glob("chunk_*.npy"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")

    chunks = []
    for chunk_file in chunk_files:
        data = np.load(chunk_file, allow_pickle=True).item()
        chunks.append(data)

    return chunks


def load_timestamps(debug_data_dir: pathlib.Path) -> Dict[int, Tuple[int, int]]:
    """Load per-step action selection info from timestamps.csv.

    Returns a mapping:
        env_step -> (action_chunk_index, action_index)
    """
    path = debug_data_dir / "timestamps.csv"
    if not path.exists():
        raise FileNotFoundError(f"Timestamps file not found: {path}")

    out: Dict[int, Tuple[int, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            env_step = int(row["env_step"])
            chunk_idx = int(row["action_chunk_index"])
            action_idx = int(row["action_index"])
            out[env_step] = (chunk_idx, action_idx)
    return out


def unflatten_debug_data(flat_data: dict) -> dict:
    """Convert flattened debug data back to nested structure.

    Special handling for 'raw_obs' which contains keys with '/' in them
    (like 'observation/image'). These should NOT be split further.
    """
    result = {}
    for key, value in flat_data.items():
        # Special handling for raw_obs - only split on first '/'
        if key.startswith("raw_obs/"):
            if "raw_obs" not in result:
                result["raw_obs"] = {}
            # The rest of the key (after 'raw_obs/') should be kept as-is
            inner_key = key[len("raw_obs/") :]
            result["raw_obs"][inner_key] = value
        else:
            # Normal nested structure handling
            parts = key.split("/")
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
    return result


def create_observation_from_debug(
    debug_data: dict, prompt: str, step_override: Optional[int] = None
) -> dict:
    """Create observation dict from debug data for policy inference.

    Prefers 'raw_obs' (the exact observation before any transforms) if available,
    otherwise falls back to 'obs_before_preprocess' with reverse transform.
    """
    # Prefer raw_obs if available (exact observation before any transforms)
    if "raw_obs" in debug_data:
        raw_obs = debug_data["raw_obs"]
        # raw_obs is the exact dict that was passed to policy.infer()
        obs = dict(raw_obs)
        if step_override is not None:
            obs["step"] = step_override
        return obs

    raise ValueError("No raw_obs found in debug data")


def get_noise_from_debug(debug_data: dict) -> np.ndarray:
    """Extract noise from debug data."""
    noise = debug_data.get("noise")
    if noise is None:
        raise ValueError("No noise found in debug data")
    # Remove batch dimension if present
    if noise.ndim == 3 and noise.shape[0] == 1:
        noise = noise[0]
    return noise


def get_saved_actions_from_debug(debug_data: dict, action_dim: int = 7) -> np.ndarray:
    """Extract saved final actions from debug data.

    Prefers 'final_actions' (post-processed, unnormalized actions ready for robot)
    over 'output_actions' (raw model output before unnormalization).

    Args:
        debug_data: Debug data dictionary containing 'final_actions' or 'output_actions'
        action_dim: The actual robot action dimension (default: 7 for LIBERO)

    Returns:
        Actions with shape (action_horizon, action_dim)
    """
    # Prefer final_actions (post-processed) over output_actions (raw model output)
    actions = debug_data.get("final_actions")
    if actions is None:
        # Fallback to output_actions for older debug data
        actions = debug_data.get("output_actions")
        if actions is None:
            raise ValueError("No final_actions or output_actions found in debug data")
        # output_actions needs dimension slicing since it's padded
        # Remove batch dimension if present
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        # Extract only the actual action dimensions (first action_dim values)
        actions = actions[:, :action_dim]
    else:
        # final_actions is already the correct shape, just remove batch dim if present
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
    return actions


def get_saved_output_actions_from_debug(debug_data: dict) -> np.ndarray:
    """Extract raw model output_actions from debug data.

    This is the (typically normalized) model output before output transforms and slicing.
    Expected shape: (action_horizon, model_action_dim) (e.g. (50, 32) for Pi0).
    """
    actions = debug_data.get("output_actions")
    if actions is None:
        raise ValueError("No output_actions found in debug data")
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    return np.asarray(actions)


def get_rtc_params_from_debug(debug_data: dict) -> Tuple[np.ndarray, int, int]:
    """Extract RTC params needed for INFERENCE_TIME_RTC replay.

    Expected structure (saved by InferenceTimeRTCBroker):
        debug_data["rtc"] = {"prev_action": <np.ndarray>, "s_param": <int>, "d_param": <int>}
    """
    rtc = debug_data.get("rtc")
    if not isinstance(rtc, dict):
        raise ValueError("No rtc params found in debug data (missing key 'rtc')")
    prev_action = rtc.get("prev_action")
    s_param = rtc.get("s_param")
    d_param = rtc.get("d_param")
    if prev_action is None or s_param is None or d_param is None:
        raise ValueError("rtc params incomplete; expected prev_action/s_param/d_param")
    return np.asarray(prev_action), int(s_param), int(d_param)


def _compute_array_diff(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "mean_abs": None,
            "max_abs": None,
        }
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "shape": list(a.shape),
        "mean_abs": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
    }


def _safe_get(d: dict, path: List[str]):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _append_jsonl(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def compute_action_overlap_stats(
    gt_horse_tails: List[dict],
    all_saved_actions: List[np.ndarray],
    chunk_boundaries: List[int],
    timestamps_by_step: Dict[int, Tuple[int, int]],
    distance_thresholds: Optional[List[float]] = None,
) -> Dict:
    """Compute statistics on how many discarded actions could be reused.
    
    For each horse tail (discarded actions from chunk N), we check how many
    of those actions are "close enough" to the actions actually used from
    the subsequent chunk N+1, meaning they could have been reused.
    
    Args:
        gt_horse_tails: List of dicts with {"start_step", "actions"} for discarded actions
        all_saved_actions: List of action arrays for each chunk
        chunk_boundaries: List of step indices where new chunks begin
        timestamps_by_step: Mapping from env_step to (chunk_idx, action_idx)
        distance_thresholds: List of L2 distance thresholds to test (default: [0.001, 0.01, 0.05, 0.1, 0.5])
    
    Returns:
        Dictionary with overlap statistics for different metrics
    """
    if distance_thresholds is None:
        distance_thresholds = [0.001, 0.01, 0.05, 0.1, 0.1375, 0.175, 0.25, 0.5]
    
    stats = {
        "total_discarded_actions": 0,
        "total_subsequent_actions": 0,
        "per_threshold": {thresh: {"reusable_count": 0, "comparisons": 0} for thresh in distance_thresholds},
        "per_tail_analysis": [],
    }
    
    sorted_steps = sorted(timestamps_by_step.keys())
    
    for tail_idx, tail in enumerate(gt_horse_tails):
        start_step = int(tail["start_step"])
        tail_actions = np.asarray(tail["actions"])
        
        if tail_actions.size == 0:
            continue
        
        stats["total_discarded_actions"] += len(tail_actions)
        
        # Find which chunk this tail came from and what the next chunk is
        # The tail starts at start_step, so we need to find what happens at/after start_step
        subsequent_actions = []
        tail_chunk_idx = None
        next_chunk_idx = None
        
        # Find the chunk index that generated this tail
        for step in sorted_steps:
            if step < start_step:
                continue
            chunk_idx, action_idx = timestamps_by_step[step]
            if tail_chunk_idx is None:
                # First step at or after start_step tells us the next chunk
                next_chunk_idx = chunk_idx
                # The tail came from the previous chunk
                if next_chunk_idx > 0:
                    tail_chunk_idx = next_chunk_idx - 1
                break
        
        if tail_chunk_idx is None or next_chunk_idx is None:
            continue
        
        # Collect actions from the subsequent chunk (next_chunk_idx)
        for step in sorted_steps:
            if step < start_step:
                continue
            chunk_idx, action_idx = timestamps_by_step[step]
            if chunk_idx == next_chunk_idx and 0 <= chunk_idx < len(all_saved_actions):
                if action_idx >= 0 and action_idx < len(all_saved_actions[chunk_idx]):
                    subsequent_actions.append(all_saved_actions[chunk_idx][action_idx])
            elif chunk_idx > next_chunk_idx:
                break
        
        if not subsequent_actions:
            continue
        
        subsequent_actions = np.array(subsequent_actions)
        stats["total_subsequent_actions"] += len(subsequent_actions)
        
        # Compare each discarded action with subsequent actions to find matches
        tail_reusable = {thresh: 0 for thresh in distance_thresholds}
        min_distances = []
        
        compare_len = min(len(tail_actions), len(subsequent_actions))
        
        for i in range(compare_len):
            tail_action = tail_actions[i]
            subseq_action = subsequent_actions[i]
            
            # Compute L2 distance
            l2_dist = np.linalg.norm(tail_action - subseq_action)
            min_distances.append(l2_dist)
            
            # Check against each threshold
            for thresh in distance_thresholds:
                if l2_dist <= thresh:
                    tail_reusable[thresh] += 1
                    stats["per_threshold"][thresh]["reusable_count"] += 1
                stats["per_threshold"][thresh]["comparisons"] += 1
        
        stats["per_tail_analysis"].append({
            "tail_idx": tail_idx,
            "start_step": start_step,
            "tail_chunk_idx": tail_chunk_idx,
            "next_chunk_idx": next_chunk_idx,
            "tail_length": len(tail_actions),
            "compared_length": compare_len,
            "min_distances": min_distances,
            "reusable_per_threshold": tail_reusable,
        })
    
    # Compute percentages
    for thresh in distance_thresholds:
        comparisons = stats["per_threshold"][thresh]["comparisons"]
        if comparisons > 0:
            reusable = stats["per_threshold"][thresh]["reusable_count"]
            stats["per_threshold"][thresh]["percentage"] = 100.0 * reusable / comparisons
        else:
            stats["per_threshold"][thresh]["percentage"] = 0.0
    
    return stats


def plot_action_comparison_with_reusability(
    replay_actions: np.ndarray,
    saved_actions: np.ndarray,
    output_path: pathlib.Path,
    action_horizon: int = 50,
    action_dim_names: Optional[List[str]] = None,
    gt_horse_tails: Optional[List[dict]] = None,
    overlap_stats: Optional[Dict] = None,
    distance_threshold: float = 0.05,
    chunk_boundaries: Optional[List[int]] = None,
) -> None:
    """Plot action comparison with reusable horse tail sections highlighted.
    
    Args:
        replay_actions: Actions used during replay, shape (num_steps, action_dim)
        saved_actions: Original saved actions, shape (num_steps, action_dim)
        output_path: Path to save the plot image
        action_horizon: Number of actions per chunk
        action_dim_names: Optional names for each action dimension
        gt_horse_tails: List of dicts with {"start_step", "actions"}
        overlap_stats: Overlap statistics from compute_action_overlap_stats
        distance_threshold: Distance threshold to use for marking reusable actions
        chunk_boundaries: Optional list of step indices where new chunks begin
    """
    num_steps, action_dim = replay_actions.shape
    
    if action_dim_names is None:
        action_dim_names = ["X", "Y", "Z", "RX", "RY", "RZ", "Gripper"]
        if action_dim > len(action_dim_names):
            action_dim_names.extend(
                [f"Dim {i}" for i in range(len(action_dim_names), action_dim)]
            )
    
    # Extract reusability info for this threshold from overlap_stats
    reusable_info = {}
    if overlap_stats and "per_tail_analysis" in overlap_stats:
        for tail_data in overlap_stats["per_tail_analysis"]:
            tail_idx = tail_data["tail_idx"]
            start_step = tail_data["start_step"]
            min_distances = tail_data["min_distances"]
            
            # Mark which actions in this tail are reusable
            reusable_mask = [d <= distance_threshold for d in min_distances]
            reusable_info[tail_idx] = {
                "start_step": start_step,
                "reusable_mask": reusable_mask,
                "distances": min_distances,
            }
    
    # Create figure
    fig, axes = plt.subplots(action_dim, 1, figsize=(14, 3 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]
    
    timesteps = np.arange(num_steps)
    differences = np.abs(replay_actions - saved_actions)
    max_diff = np.max(differences, axis=0)
    mean_diff = np.mean(differences, axis=0)
    
    for dim in range(action_dim):
        ax = axes[dim]
        
        # Plot saved actions (original)
        ax.plot(
            timesteps,
            saved_actions[:, dim],
            "b-",
            linewidth=2,
            label="Saved (Original)",
            alpha=0.8,
            zorder=2,
        )
        
        # Plot replay actions
        ax.plot(
            timesteps,
            replay_actions[:, dim],
            "r--",
            linewidth=2,
            label="Replay",
            alpha=0.8,
            zorder=2,
        )
        
        # Plot horse tails with reusability highlighting
        reusable_label_added = False
        not_reusable_label_added = False
        
        if gt_horse_tails:
            for idx, tail in enumerate(gt_horse_tails):
                start_step = int(tail["start_step"])
                tail_actions = np.asarray(tail["actions"])
                if tail_actions.size == 0 or dim >= tail_actions.shape[1]:
                    continue
                
                tail_steps = np.arange(start_step, start_step + tail_actions.shape[0])
                
                # Check if we have reusability info for this tail
                if idx in reusable_info:
                    reusable_mask = reusable_info[idx]["reusable_mask"]
                    
                    # Plot reusable and non-reusable sections separately
                    for i in range(len(tail_actions)):
                        if i >= len(reusable_mask):
                            break
                        
                        if i < len(tail_actions) - 1:
                            step_range = [tail_steps[i], tail_steps[i + 1]]
                            action_range = [tail_actions[i, dim], tail_actions[i + 1, dim]]
                            
                            if reusable_mask[i]:
                                # Reusable action - bright green, thicker
                                ax.plot(
                                    step_range,
                                    action_range,
                                    color="lime",
                                    linewidth=2.5,
                                    alpha=0.8,
                                    zorder=3,
                                    label="Reusable (discarded)" if not reusable_label_added else None,
                                )
                                reusable_label_added = True
                            else:
                                # Not reusable - red, thinner
                                ax.plot(
                                    step_range,
                                    action_range,
                                    color="red",
                                    linewidth=1.5,
                                    alpha=0.4,
                                    zorder=1,
                                    label="Not reusable (discarded)" if not not_reusable_label_added else None,
                                )
                                not_reusable_label_added = True
                else:
                    # No reusability info, plot normally
                    ax.plot(
                        tail_steps,
                        tail_actions[:, dim],
                        color="gray",
                        linewidth=1,
                        alpha=0.35,
                        zorder=1,
                        label="GT horse tail" if idx == 0 and not reusable_label_added else None,
                    )
        
        # Shade the difference between replay and saved
        ax.fill_between(
            timesteps,
            saved_actions[:, dim],
            replay_actions[:, dim],
            alpha=0.2,
            color="purple",
            label="Difference" if dim == 0 else None,
            zorder=0,
        )
        
        # Title with stats
        dim_name = action_dim_names[dim] if dim < len(action_dim_names) else f"Dim {dim}"
        ax.set_title(
            f"{dim_name} | Max Diff: {max_diff[dim]:.6f}, Mean Diff: {mean_diff[dim]:.6f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Value")
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, which="major")
        
        # Add vertical lines at chunk boundaries
        if chunk_boundaries:
            for boundary in chunk_boundaries:
                if 0 <= boundary <= num_steps:
                    ax.axvline(x=boundary, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        else:
            for boundary in range(0, num_steps + 1, action_horizon):
                ax.axvline(x=boundary, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    
    # Set x-axis ticks
    if chunk_boundaries:
        axes[-1].set_xticks(chunk_boundaries)
    else:
        xticks = np.arange(0, num_steps + 1, action_horizon)
        axes[-1].set_xticks(xticks)
    axes[-1].set_xlabel("Timestep")
    
    # Overall title
    total_max_diff = np.max(differences)
    total_mean_diff = np.mean(differences)
    
    reuse_pct = 0.0
    if overlap_stats and distance_threshold in overlap_stats["per_threshold"]:
        reuse_pct = overlap_stats["per_threshold"][distance_threshold]["percentage"]
    
    fig.suptitle(
        f"Action Comparison with Reusability Highlighting (Threshold: {distance_threshold:.4f})\n"
        f"Total Max Diff: {total_max_diff:.8f} | Total Mean Diff: {total_mean_diff:.8f}\n"
        f"Reusable Actions: {reuse_pct:.1f}%",
        fontsize=14,
        fontweight="bold",
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_analysis(
    overlap_stats: Dict,
    output_path: pathlib.Path,
) -> None:
    """Create visualizations for action overlap/reuse analysis.
    
    Creates two plots:
    1. Bar chart showing reuse percentage vs distance threshold
    2. Heatmap/matrix showing per-tail reuse statistics
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.35)
    
    # Plot 1: Reuse percentage vs threshold (bar chart)
    ax1 = fig.add_subplot(gs[0, :])
    thresholds = sorted(overlap_stats["per_threshold"].keys())
    percentages = [overlap_stats["per_threshold"][t]["percentage"] for t in thresholds]
    reusable_counts = [overlap_stats["per_threshold"][t]["reusable_count"] for t in thresholds]
    comparisons = [overlap_stats["per_threshold"][t]["comparisons"] for t in thresholds]
    
    x_pos = np.arange(len(thresholds))
    bars = ax1.bar(x_pos, percentages, alpha=0.7, edgecolor='black')
    
    # Color bars by percentage
    colors = plt.cm.RdYlGn(np.array(percentages) / 100.0)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax1.set_xlabel('L2 Distance Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Reusable Actions (%)', fontsize=12, fontweight='bold')
    ax1.set_title(
        f'Action Reuse Potential Across Distance Thresholds\n'
        f'Total Discarded: {overlap_stats["total_discarded_actions"]}, '
        f'Total Comparisons: {comparisons[0] if comparisons else 0}',
        fontsize=14, fontweight='bold'
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{t:.3f}' for t in thresholds])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])
    
    # Add value labels on bars
    for i, (bar, pct, count) in enumerate(zip(bars, percentages, reusable_counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%\n({count})',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Per-tail reuse matrix
    if overlap_stats["per_tail_analysis"]:
        ax2 = fig.add_subplot(gs[1, :])
        
        tail_indices = [t["tail_idx"] for t in overlap_stats["per_tail_analysis"]]
        reuse_matrix = []
        
        for tail_data in overlap_stats["per_tail_analysis"]:
            reuse_row = [
                tail_data["reusable_per_threshold"].get(t, 0) 
                for t in thresholds
            ]
            reuse_matrix.append(reuse_row)
        
        reuse_matrix = np.array(reuse_matrix)
        
        im = ax2.imshow(reuse_matrix.T, aspect='auto', cmap='RdYlGn', 
                       interpolation='nearest', vmin=0, vmax=reuse_matrix.max())
        
        ax2.set_xlabel('Horse Tail Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Distance Threshold', fontsize=12, fontweight='bold')
        ax2.set_title('Reusable Action Count Per Tail', fontsize=13, fontweight='bold')
        
        ax2.set_xticks(np.arange(len(tail_indices)))
        ax2.set_xticklabels(tail_indices)
        ax2.set_yticks(np.arange(len(thresholds)))
        ax2.set_yticklabels([f'{t:.3f}' for t in thresholds])
        
        # Add text annotations
        for i in range(len(thresholds)):
            for j in range(len(tail_indices)):
                text = ax2.text(j, i, f'{reuse_matrix[j, i]:.0f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax2, label='Reusable Actions')
    
    # Plot 3: Distribution of minimum distances
    ax3 = fig.add_subplot(gs[2, 0])
    all_distances = []
    for tail_data in overlap_stats["per_tail_analysis"]:
        all_distances.extend(tail_data["min_distances"])
    
    if all_distances:
        ax3.hist(all_distances, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('L2 Distance', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax3.set_title('Distribution of Action Distances\n(Tail vs Subsequent)', 
                     fontsize=12, fontweight='bold')
        ax3.axvline(np.median(all_distances), color='red', linestyle='--', 
                   label=f'Median: {np.median(all_distances):.4f}')
        ax3.axvline(np.mean(all_distances), color='orange', linestyle='--', 
                   label=f'Mean: {np.mean(all_distances):.4f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add log scale option if distances vary widely
        if max(all_distances) / (min(all_distances) + 1e-10) > 100:
            ax3.set_yscale('log')
    
    # Plot 4: Cumulative reuse potential
    ax4 = fig.add_subplot(gs[2, 1])
    if all_distances:
        sorted_distances = np.sort(all_distances)
        cumulative_pct = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
        
        ax4.plot(sorted_distances, cumulative_pct, linewidth=2)
        ax4.set_xlabel('L2 Distance Threshold', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Cumulative Reusable Actions (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Cumulative Distribution of Reuse Potential', 
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, min(1.0, np.percentile(sorted_distances, 95))])
        
        # Add reference lines at key thresholds
        for thresh in [0.01, 0.05, 0.1]:
            idx = np.searchsorted(sorted_distances, thresh)
            if idx < len(cumulative_pct):
                ax4.axvline(thresh, color='gray', linestyle='--', alpha=0.5)
                ax4.text(thresh, cumulative_pct[idx], f'{cumulative_pct[idx]:.1f}%',
                        fontsize=8, ha='right')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_action_comparison(
    replay_actions: np.ndarray,
    saved_actions: np.ndarray,
    output_path: pathlib.Path,
    action_horizon: int = 50,
    action_dim_names: Optional[List[str]] = None,
    gt_horse_tails: Optional[List[dict]] = None,
    pred_horse_tails: Optional[List[dict]] = None,
    chunk_boundaries: Optional[List[int]] = None,
) -> None:
    """Plot comparison of replay actions vs saved actions for each dimension.

    Args:
        replay_actions: Actions used during replay, shape (num_steps, action_dim)
        saved_actions: Original saved actions, shape (num_steps, action_dim)
        output_path: Path to save the plot image
        action_horizon: Number of actions per chunk (for grid spacing)
        action_dim_names: Optional names for each action dimension
        gt_horse_tails: Optional list of dicts with keys {"start_step", "actions"} to plot
            discarded ground-truth actions as faint "horse tail" lines.
        pred_horse_tails: Optional list of dicts with keys {"start_step", "actions"} to plot
            discarded predicted actions as faint "horse tail" lines.
        chunk_boundaries: Optional list of step indices where a new chunk begins.
    """
    num_steps, action_dim = replay_actions.shape

    gt_tail_colors = None
    pred_tail_colors = None
    if gt_horse_tails:
        cmap = plt.get_cmap("tab20")
        gt_tail_colors = [cmap(i % cmap.N) for i in range(len(gt_horse_tails))]
    if pred_horse_tails:
        cmap = plt.get_cmap("tab20b")
        pred_tail_colors = [cmap(i % cmap.N) for i in range(len(pred_horse_tails))]

    if action_dim_names is None:
        # Default names for LIBERO 7-DoF actions
        action_dim_names = ["X", "Y", "Z", "RX", "RY", "RZ", "Gripper"]
        if action_dim > len(action_dim_names):
            action_dim_names.extend(
                [f"Dim {i}" for i in range(len(action_dim_names), action_dim)]
            )

    # Create figure with subplots for each action dimension
    fig, axes = plt.subplots(action_dim, 1, figsize=(14, 3 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]

    timesteps = np.arange(num_steps)

    # Calculate differences for summary
    differences = np.abs(replay_actions - saved_actions)
    max_diff = np.max(differences, axis=0)
    mean_diff = np.mean(differences, axis=0)

    for dim in range(action_dim):
        ax = axes[dim]

        # Plot saved actions (original)
        ax.plot(
            timesteps,
            saved_actions[:, dim],
            "b-",
            linewidth=2,
            label="Saved (Original)",
            alpha=0.8,
        )

        # Plot replay actions
        ax.plot(
            timesteps,
            replay_actions[:, dim],
            "r--",
            linewidth=2,
            label="Replay",
            alpha=0.8,
        )

        # Plot ground-truth horse tail predictions (discarded actions)
        if gt_horse_tails:
            added_label = False
            for idx, tail in enumerate(gt_horse_tails):
                start_step = int(tail["start_step"])
                tail_actions = np.asarray(tail["actions"])
                if tail_actions.size == 0 or dim >= tail_actions.shape[1]:
                    continue
                tail_steps = np.arange(
                    start_step, start_step + tail_actions.shape[0]
                )
                ax.plot(
                    tail_steps,
                    tail_actions[:, dim],
                    color=gt_tail_colors[idx],
                    linewidth=1,
                    alpha=0.35,
                    label="GT horse tail (discarded)" if not added_label else None,
                )
                added_label = True

        # Plot predicted horse tail predictions (discarded actions)
        if pred_horse_tails:
            added_label = False
            for idx, tail in enumerate(pred_horse_tails):
                start_step = int(tail["start_step"])
                tail_actions = np.asarray(tail["actions"])
                if tail_actions.size == 0 or dim >= tail_actions.shape[1]:
                    continue
                tail_steps = np.arange(
                    start_step, start_step + tail_actions.shape[0]
                )
                ax.plot(
                    tail_steps,
                    tail_actions[:, dim],
                    color=pred_tail_colors[idx],
                    linewidth=1,
                    alpha=0.35,
                    linestyle="--",
                    label="Pred horse tail (discarded)" if not added_label else None,
                )
                added_label = True

        # Shade the difference
        ax.fill_between(
            timesteps,
            saved_actions[:, dim],
            replay_actions[:, dim],
            alpha=0.3,
            color="purple",
            label="Difference",
        )

        # Title with difference stats
        dim_name = (
            action_dim_names[dim] if dim < len(action_dim_names) else f"Dim {dim}"
        )
        ax.set_title(
            f"{dim_name} | Max Diff: {max_diff[dim]:.6f}, Mean Diff: {mean_diff[dim]:.6f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, which="major")

        # Add vertical lines at chunk boundaries
        if chunk_boundaries:
            for boundary in chunk_boundaries:
                if 0 <= boundary <= num_steps:
                    ax.axvline(
                        x=boundary,
                        color="gray",
                        linestyle="--",
                        linewidth=1,
                        alpha=0.5,
                    )
        else:
            for boundary in range(0, num_steps + 1, action_horizon):
                ax.axvline(
                    x=boundary,
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                )

    # Set x-axis ticks at chunk boundaries (fallback to action horizon boundaries)
    if chunk_boundaries:
        axes[-1].set_xticks(chunk_boundaries)
    else:
        xticks = np.arange(0, num_steps + 1, action_horizon)
        axes[-1].set_xticks(xticks)
    axes[-1].set_xlabel("Timestep")

    # Overall title with determinism verdict
    total_max_diff = np.max(differences)
    total_mean_diff = np.mean(differences)

    fig.suptitle(
        f"Action Comparison: Replay vs Saved\n"
        f"Total Max Diff: {total_max_diff:.8f} | Total Mean Diff: {total_mean_diff:.8f}\n",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_action_comparison_html(
    replay_actions: np.ndarray,
    saved_actions: np.ndarray,
    output_path: pathlib.Path,
    action_horizon: int = 50,
    action_dim_names: Optional[List[str]] = None,
    gt_horse_tails: Optional[List[dict]] = None,
    pred_horse_tails: Optional[List[dict]] = None,
    chunk_boundaries: Optional[List[int]] = None,
) -> None:
    """Create an interactive HTML plot with hoverable traces."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "plotly is required for HTML output; install it or pass --no_html"
        ) from exc

    num_steps, action_dim = replay_actions.shape
    if action_dim_names is None:
        action_dim_names = ["X", "Y", "Z", "RX", "RY", "RZ", "Gripper"]
        if action_dim > len(action_dim_names):
            action_dim_names.extend(
                [f"Dim {i}" for i in range(len(action_dim_names), action_dim)]
            )

    fig = make_subplots(
        rows=action_dim,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            action_dim_names[i] if i < len(action_dim_names) else f"Dim {i}"
            for i in range(action_dim)
        ],
    )

    timesteps = np.arange(num_steps)
    for dim in range(action_dim):
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=saved_actions[:, dim],
                mode="lines",
                name="Saved (Original)",
                line=dict(color="blue"),
                hovertemplate="Saved<br>step=%{x}<br>value=%{y}<extra></extra>",
            ),
            row=dim + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=replay_actions[:, dim],
                mode="lines",
                name="Replay",
                line=dict(color="red", dash="dash"),
                hovertemplate="Replay<br>step=%{x}<br>value=%{y}<extra></extra>",
            ),
            row=dim + 1,
            col=1,
        )

        if gt_horse_tails:
            for idx, tail in enumerate(gt_horse_tails):
                tail_actions = np.asarray(tail["actions"])
                if tail_actions.size == 0 or dim >= tail_actions.shape[1]:
                    continue
                start_step = int(tail["start_step"])
                tail_steps = np.arange(
                    start_step, start_step + tail_actions.shape[0]
                )
                fig.add_trace(
                    go.Scatter(
                        x=tail_steps,
                        y=tail_actions[:, dim],
                        mode="lines",
                        name=f"GT Tail {idx + 1}",
                        line=dict(width=1),
                        opacity=0.5,
                        hovertemplate=(
                            f"GT Tail {idx + 1}<br>step=%{{x}}<br>value=%{{y}}"
                            "<extra></extra>"
                        ),
                    ),
                    row=dim + 1,
                    col=1,
                )

        if pred_horse_tails:
            for idx, tail in enumerate(pred_horse_tails):
                tail_actions = np.asarray(tail["actions"])
                if tail_actions.size == 0 or dim >= tail_actions.shape[1]:
                    continue
                start_step = int(tail["start_step"])
                tail_steps = np.arange(
                    start_step, start_step + tail_actions.shape[0]
                )
                fig.add_trace(
                    go.Scatter(
                        x=tail_steps,
                        y=tail_actions[:, dim],
                        mode="lines",
                        name=f"Pred Tail {idx + 1}",
                        line=dict(width=1, dash="dot"),
                        opacity=0.5,
                        hovertemplate=(
                            f"Pred Tail {idx + 1}<br>step=%{{x}}<br>value=%{{y}}"
                            "<extra></extra>"
                        ),
                    ),
                    row=dim + 1,
                    col=1,
                )

    boundaries = chunk_boundaries or list(range(0, num_steps + 1, action_horizon))
    for boundary in boundaries:
        if 0 <= boundary <= num_steps:
            fig.add_vline(
                x=boundary, line_width=1, line_dash="dash", line_color="gray", opacity=0.5
            )

    fig.update_layout(
        height=250 * action_dim,
        title="Action Comparison: Replay vs Saved (Interactive)",
        showlegend=True,
        clickmode="event+select",
    )
    fig.update_xaxes(title_text="Timestep", row=action_dim, col=1)
    div_id = "action_comparison_plot"
    plot_html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        div_id=div_id,
    )
    highlight_script = f"""
<script>
  (function() {{
    const plot = document.getElementById("{div_id}");
    if (!plot) return;
    const resetButton = document.getElementById("reset-highlight");
    function applyHighlightByName(traceName) {{
      const n = plot.data.length;
      const widths = [];
      const opacities = [];
      for (let i = 0; i < n; i++) {{
        const isSelected = (plot.data[i].name === traceName);
        widths.push(isSelected ? 3 : 1);
        opacities.push(isSelected ? 1.0 : 0.2);
      }}
      Plotly.restyle(plot, {{"line.width": widths, "opacity": opacities}});
    }}
    function clearHighlight() {{
      const n = plot.data.length;
      const widths = Array(n).fill(1);
      const opacities = Array(n).fill(1.0);
      Plotly.restyle(plot, {{"line.width": widths, "opacity": opacities}});
    }}
    plot.on('plotly_click', function(evt) {{
      if (!evt || !evt.points || !evt.points.length) return;
      const traceName = evt.points[0].data && evt.points[0].data.name;
      if (!traceName) return;
      applyHighlightByName(traceName);
    }});
    plot.on('plotly_doubleclick', function() {{
      clearHighlight();
    }});
    document.addEventListener('keydown', function(evt) {{
      if (evt.key === 'Escape') {{
        clearHighlight();
      }}
    }});
    if (resetButton) {{
      resetButton.addEventListener('click', function() {{
        clearHighlight();
      }});
    }}
  }})();
</script>
"""
    html = "<!doctype html><html><head><meta charset='utf-8'></head><body>"
    html += (
        "<div style='margin:8px 0;'>"
        "<button id='reset-highlight' type='button'>Reset highlight</button>"
        "</div>"
    )
    html += plot_html + highlight_script + "</body></html>"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def replay_episode(
    config: ReplayConfig,
    policy: Optional[_websocket_client_policy.WebsocketClientPolicy],
    console: Console,
) -> Tuple[bool, np.ndarray, np.ndarray, List[dict], List[dict], List[int]]:
    """Replay a single episode from debug data.

    Args:
        config: Replay configuration
        policy: Policy client (only needed if not using saved actions)
        console: Rich console for output

    Returns:
        Tuple of (success, replay_actions, saved_actions) where:
        - success: True if episode was successful, False otherwise
        - replay_actions: Actions used during replay, shape (num_steps, action_dim)
        - saved_actions: Original saved actions, shape (num_steps, action_dim)
        - gt_horse_tails: Ground-truth discarded action tails
        - pred_horse_tails: Predicted discarded action tails
        - chunk_boundaries: Step indices where a new action chunk began
    """
    # Load metadata and chunks
    metadata = load_metadata(config.debug_data_dir)
    chunks = load_debug_chunks(config.debug_data_dir)
    timestamps_by_step = load_timestamps(config.debug_data_dir)

    console.print(f"[bold blue]Loaded {len(chunks)} debug chunks[/bold blue]")
    console.print(f"  Task Suite: {metadata['task_suite_name']}")
    console.print(f"  Task ID: {metadata['task_id']}")
    console.print(f"  Original Success: {metadata['success']}")
    console.print(
        f"  Mode: {'Using saved actions' if config.use_saved_actions else 'Re-inferring from policy'}"
    )
    if not config.use_saved_actions:
        console.print(f"  Infer Type: {'RTC' if config.use_rtc else 'SYNC'}")
    console.print()

    # Setup environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[metadata["task_suite_name"]]()
    task = task_suite.get_task(metadata["task_id"])

    # Get initial state for this episode
    all_initial_states = task_suite.get_task_init_states(metadata["task_id"])
    episode_idx = (
        metadata.get("episode_idx", 1) - 1
    )  # episode_idx is 1-based after increment
    if episode_idx >= len(all_initial_states):
        episode_idx = 0
    initial_state = all_initial_states[episode_idx : episode_idx + 1]

    env_raw, task_description = utils._get_libero_env(
        task, LIBERO_ENV_RESOLUTION, seed=config.seed
    )

    env = LiberoSimEnvironment(
        env=env_raw,
        task_description=task_description,
        initial_states=initial_state,
        resize_size=config.resize_size,
        num_steps_wait=config.num_steps_wait,
        max_episode_steps=config.max_steps,
        control_hz=config.control_hz,
    )

    console.print("[bold green]Environment initialized[/bold green]")
    console.print(f"  Task: {task_description}")
    console.print()

    # Reset environment
    env.reset()

    # Pre-extract all saved actions from chunks for comparison
    all_saved_actions = []
    for chunk in chunks:
        chunk_data = unflatten_debug_data(chunk)
        saved_chunk_actions = get_saved_actions_from_debug(
            chunk_data, config.action_dim
        )
        all_saved_actions.append(saved_chunk_actions)

    # Precompute chunk boundaries based on timestamps.
    chunk_boundaries: List[int] = []
    if timestamps_by_step:
        sorted_steps = sorted(timestamps_by_step.keys())
        prev_chunk_idx, prev_action_idx = timestamps_by_step[sorted_steps[0]]
        chunk_boundaries.append(sorted_steps[0])
        for step_idx in range(1, len(sorted_steps)):
            step = sorted_steps[step_idx]
            chunk_idx, action_idx = timestamps_by_step[step]
            if chunk_idx != prev_chunk_idx:
                chunk_boundaries.append(step)
            prev_chunk_idx, prev_action_idx = chunk_idx, action_idx

    # Replay loop
    frames: List[np.ndarray] = []
    replay_actions_list: List[np.ndarray] = []  # Track actions used during replay
    saved_actions_list: List[np.ndarray] = []  # Track corresponding saved actions
    step = 0
    chunk_debug_report_path: Optional[pathlib.Path] = None
    if config.return_debug_data:
        if config.debug_report_path is None:
            chunk_debug_report_path = (
                config.debug_data_dir / "triton_debug_compare.jsonl"
            )
        else:
            chunk_debug_report_path = pathlib.Path(config.debug_report_path)
        # Reset report file for a clean run.
        if chunk_debug_report_path.exists():
            chunk_debug_report_path.unlink()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task("[cyan]Replaying episode...", total=None)
        ran_out_of_chunks = False
        last_action = None
        inferred_actions_by_chunk: Dict[int, np.ndarray] = {}
        # SYNC-mode iterator state (unused when using timestamps mapping).
        chunk_idx = 0
        action_idx = 0
        current_actions = None
        current_saved_actions = None  # Saved actions for current chunk
        use_timestamp_map = bool(timestamps_by_step) and not config.use_rtc

        def _null_action_from_chunk(chunk_actions: np.ndarray) -> np.ndarray:
            null_action = np.asarray(chunk_actions[-1]).copy()
            null_action[:-1] = 0.0
            return null_action

        while not env.is_episode_complete() and step < config.max_steps:
            # Get observation for frame capture
            obs = env.get_observation()
            frames.append(obs["observation/image"])

            # Determine which action to use
            if config.use_rtc or use_timestamp_map:
                # RTC and replan-sync execution use per-step (chunk_idx, action_idx) mapping recorded during rollout.
                if step not in timestamps_by_step:
                    if not ran_out_of_chunks:
                        console.print(
                            "[yellow]Warning: No timestamp entry for this step; using last action[/yellow]"
                        )
                        ran_out_of_chunks = True
                    action = last_action
                    saved_action = None
                else:
                    chunk_idx, action_idx = timestamps_by_step[step]

                    # Get saved actions for comparison (if available)
                    saved_action = None
                    if 0 <= chunk_idx < len(all_saved_actions) and action_idx >= 0:
                        saved_action = all_saved_actions[chunk_idx][action_idx]

                    # Lazily infer actions for the referenced chunk index (or use saved actions).
                    if config.use_saved_actions:
                        if not (0 <= chunk_idx < len(all_saved_actions)):
                            action = last_action
                        else:
                            chunk_actions = np.asarray(all_saved_actions[chunk_idx])
                            if action_idx < 0 or action_idx >= len(chunk_actions):
                                # Mirror broker null-action semantics
                                action = _null_action_from_chunk(chunk_actions)
                            else:
                                action = chunk_actions[action_idx]
                    else:
                        if policy is None:
                            raise ValueError(
                                "Policy is required when not using saved actions"
                            )
                        if chunk_idx not in inferred_actions_by_chunk:
                            if not (0 <= chunk_idx < len(chunks)):
                                inferred_actions_by_chunk[chunk_idx] = np.asarray([])
                            else:
                                chunk_data = unflatten_debug_data(chunks[chunk_idx])
                                noise = get_noise_from_debug(chunk_data)
                                # Use the exact observation saved at inference time.
                                debug_obs = create_observation_from_debug(
                                    chunk_data, task_description, step_override=None
                                )
                                if config.use_rtc:
                                    prev_action, s_param, d_param = get_rtc_params_from_debug(
                                        chunk_data
                                    )
                                    response = policy.infer(
                                        debug_obs,
                                        use_rtc=True,
                                        prev_action=prev_action,
                                        s_param=s_param,
                                        d_param=d_param,
                                        noise=noise,
                                        return_debug_data=config.return_debug_data,
                                    )
                                else:
                                    response = policy.infer(
                                        debug_obs,
                                        noise=noise,
                                        return_debug_data=config.return_debug_data,
                                    )
                                inferred_actions_by_chunk[chunk_idx] = np.asarray(
                                    response["actions"]
                                )

                        chunk_actions = inferred_actions_by_chunk[chunk_idx]
                        if chunk_actions.size == 0:
                            action = last_action
                        elif action_idx < 0 or action_idx >= len(chunk_actions):
                            action = _null_action_from_chunk(chunk_actions)
                        else:
                            action = chunk_actions[action_idx]

                    progress.update(
                        task_progress,
                        description=f"[cyan]Step {step}, Chunk {chunk_idx + 1}/{len(chunks)}",
                    )
            else:
                # SYNC path: consume chunks sequentially.
                if ran_out_of_chunks:
                    action = last_action
                    saved_action = None
                elif current_actions is None or action_idx >= len(current_actions):
                    if chunk_idx >= len(chunks):
                        if not ran_out_of_chunks:
                            console.print(
                                "[yellow]Warning: Ran out of debug chunks, using last action[/yellow]"
                            )
                            ran_out_of_chunks = True
                        action = last_action
                        saved_action = None
                    else:
                        chunk_data = unflatten_debug_data(chunks[chunk_idx])
                        current_saved_actions = all_saved_actions[chunk_idx]

                        if config.use_saved_actions:
                            current_actions = current_saved_actions.copy()
                        else:
                            if policy is None:
                                raise ValueError(
                                    "Policy is required when not using saved actions"
                                )
                            noise = get_noise_from_debug(chunk_data)
                            debug_obs = create_observation_from_debug(
                                chunk_data, task_description, step_override=step
                            )
                            response = policy.infer(
                                debug_obs,
                                noise=noise,
                                return_debug_data=config.return_debug_data,
                            )
                            current_actions = response["actions"]

                        chunk_idx += 1
                        action_idx = 0

                        progress.update(
                            task_progress,
                            description=f"[cyan]Step {step}, Chunk {chunk_idx}/{len(chunks)}",
                        )

                        action = current_actions[action_idx]
                        saved_action = current_saved_actions[action_idx]
                        action_idx += 1
                else:
                    action = current_actions[action_idx]
                    saved_action = current_saved_actions[action_idx]  # type: ignore[index]
                    action_idx += 1

            # Track actions for comparison (only when we have valid saved actions)
            if not ran_out_of_chunks and saved_action is not None:
                replay_actions_list.append(
                    action.copy() if hasattr(action, "copy") else np.array(action)
                )
                saved_actions_list.append(
                    saved_action.copy()
                    if hasattr(saved_action, "copy")
                    else np.array(saved_action)
                )

            # Remember last action for when we run out of chunks
            last_action = action

            # Apply action
            env.apply_action({"actions": action})
            step += 1

    # Capture final frame
    if not env.is_episode_complete():
        obs = env.get_observation()
        frames.append(obs["observation/image"])

    success = env.current_success

    # Save video
    output_path = config.output_video
    if output_path is None:
        output_path = config.debug_data_dir / "replay.mp4"
    else:
        output_path = pathlib.Path(output_path)

    console.print(f"\n[bold]Saving replay video to {output_path}[/bold]")
    imageio.mimwrite(
        str(output_path),
        [np.asarray(f) for f in frames],
        fps=config.control_hz,
    )

    # Cleanup
    env.close()

    # Convert action lists to arrays
    replay_actions = (
        np.array(replay_actions_list) if replay_actions_list else np.array([])
    )
    saved_actions = np.array(saved_actions_list) if saved_actions_list else np.array([])

    # Build horse tail lists after replay (predictions are only available after inference).
    gt_horse_tails: List[dict] = []
    pred_horse_tails: List[dict] = []
    if timestamps_by_step:
        sorted_steps = sorted(timestamps_by_step.keys())
        prev_chunk_idx, prev_action_idx = timestamps_by_step[sorted_steps[0]]
        for step_idx in range(1, len(sorted_steps)):
            step = sorted_steps[step_idx]
            chunk_idx, action_idx = timestamps_by_step[step]
            if chunk_idx != prev_chunk_idx and prev_action_idx >= 0:
                start_idx = prev_action_idx + 1
                if config.plot_gt_horse_tails and 0 <= prev_chunk_idx < len(all_saved_actions):
                    prev_actions = all_saved_actions[prev_chunk_idx]
                    if start_idx < len(prev_actions):
                        gt_horse_tails.append(
                            {
                                "start_step": step,
                                "actions": np.asarray(prev_actions[start_idx:]),
                            }
                        )
                if config.plot_pred_horse_tails and prev_chunk_idx in inferred_actions_by_chunk:
                    pred_actions = inferred_actions_by_chunk[prev_chunk_idx]
                    if pred_actions.size > 0 and start_idx < len(pred_actions):
                        pred_horse_tails.append(
                            {
                                "start_step": step,
                                "actions": np.asarray(pred_actions[start_idx:]),
                            }
                        )
            prev_chunk_idx, prev_action_idx = chunk_idx, action_idx

    return (
        success,
        replay_actions,
        saved_actions,
        gt_horse_tails,
        pred_horse_tails,
        chunk_boundaries,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Replay debug data from a saved episode"
    )
    parser.add_argument(
        "--debug_data_dir",
        type=str,
        required=True,
        help="Path to the episode directory containing metadata.json and debug_data/",
    )
    parser.add_argument(
        "--use_saved_actions",
        action="store_true",
        help="Use saved output_actions directly instead of re-inferring from policy",
    )
    parser.add_argument(
        "--use_rtc",
        action="store_true",
        help="If set, use RTC inference mode when re-inferring from policy (requires saved rtc params in debug data).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Policy server host (only needed if not using --use_saved_actions)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Policy server port (only needed if not using --use_saved_actions)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for environment",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Output video path (default: <debug_data_dir>/replay.mp4)",
    )
    parser.add_argument(
        "--return_debug_data",
        action="store_true",
        help="If set, request debug payloads from the policy server and write a per-chunk comparison report.",
    )
    parser.add_argument(
        "--debug_report_path",
        type=str,
        default=None,
        help="Where to write the per-chunk debug comparison report (jsonl). Default: <debug_data_dir>/triton_debug_compare.jsonl",
    )
    parser.add_argument(
        "--plot_gt_horse_tails",
        action="store_true",
        help="Plot ground-truth discarded action tails from debug data.",
    )
    parser.add_argument(
        "--plot_pred_horse_tails",
        action="store_true",
        help="Plot predicted discarded action tails from replay inference.",
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default=None,
        help="Path for interactive HTML plot (default: <debug_data_dir>/action_comparison.html).",
    )
    parser.add_argument(
        "--no_html",
        action="store_true",
        help="Skip generating the interactive HTML plot.",
    )
    parser.add_argument(
        "--analyze_overlap",
        action="store_true",
        default=True,
        help="Compute action overlap/reuse statistics (default: True).",
    )
    parser.add_argument(
        "--no_analyze_overlap",
        action="store_false",
        dest="analyze_overlap",
        help="Skip computing action overlap/reuse statistics.",
    )

    args = parser.parse_args()

    console = Console()

    config = ReplayConfig(
        debug_data_dir=pathlib.Path(args.debug_data_dir),
        host=args.host,
        port=args.port,
        seed=args.seed,
        max_steps=args.max_steps,
        output_video=args.output_video,
        use_saved_actions=args.use_saved_actions,
        use_rtc=args.use_rtc,
        return_debug_data=args.return_debug_data,
        debug_report_path=args.debug_report_path,
        plot_gt_horse_tails=args.plot_gt_horse_tails,
        plot_pred_horse_tails=args.plot_pred_horse_tails,
        output_html=args.output_html,
        no_html=args.no_html,
        analyze_overlap=args.analyze_overlap,
    )

    console.print(
        "[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]"
    )
    console.print(
        "[bold magenta]                    Debug Data Replay                       [/bold magenta]"
    )
    console.print(
        "[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]"
    )
    console.print()

    policy = None
    if not config.use_saved_actions:
        if config.use_rtc and not config.return_debug_data:
            console.print(
                "[yellow]Note: --use_rtc relies on rtc params saved inside debug_data; "
                "make sure the episode was generated with --save_debug_data.[/yellow]"
            )
        # Connect to policy server
        console.print(
            f"[bold]Connecting to policy server at {config.host}:{config.port}...[/bold]"
        )
        policy = _websocket_client_policy.WebsocketClientPolicy(
            host=config.host,
            port=config.port,
        )
        console.print("[green]Connected![/green]")
        console.print()
    else:
        console.print("[bold]Using saved actions (no policy server needed)[/bold]")
        console.print()

    # Run replay
    try:
        (
            success,
            replay_actions,
            saved_actions,
            gt_horse_tails,
            pred_horse_tails,
            chunk_boundaries,
        ) = replay_episode(config, policy, console)

        # Generate action comparison plot
        if len(replay_actions) > 0 and len(saved_actions) > 0:
            plot_path = config.debug_data_dir / "action_comparison.png"
            console.print(
                f"\n[bold]Generating action comparison plot: {plot_path}[/bold]"
            )
            plot_action_comparison(
                replay_actions,
                saved_actions,
                plot_path,
                action_horizon=config.action_horizon,
                gt_horse_tails=gt_horse_tails,
                pred_horse_tails=pred_horse_tails,
                chunk_boundaries=chunk_boundaries,
            )
            if not config.no_html:
                html_path = (
                    pathlib.Path(config.output_html)
                    if config.output_html
                    else config.debug_data_dir / "action_comparison.html"
                )
                try:
                    plot_action_comparison_html(
                        replay_actions,
                        saved_actions,
                        html_path,
                        action_horizon=config.action_horizon,
                        gt_horse_tails=gt_horse_tails,
                        pred_horse_tails=pred_horse_tails,
                        chunk_boundaries=chunk_boundaries,
                    )
                    console.print(
                        f"[bold]Generated interactive HTML: {html_path}[/bold]"
                    )
                except RuntimeError as exc:
                    console.print(f"[yellow]Skipping HTML plot: {exc}[/yellow]")

            # Print summary statistics
            differences = np.abs(replay_actions - saved_actions)
            max_diff = np.max(differences)
            mean_diff = np.mean(differences)
            is_deterministic = max_diff < 1e-5

            console.print()
            console.print("[bold cyan]Action Comparison Summary:[/bold cyan]")
            console.print(f"  Total timesteps compared: {len(replay_actions)}")
            console.print(f"  Max absolute difference: {max_diff:.10f}")
            console.print(f"  Mean absolute difference: {mean_diff:.10f}")
        
        # Compute and visualize action overlap/reuse analysis
        if config.analyze_overlap and gt_horse_tails and chunk_boundaries:
            console.print()
            console.print("[bold cyan]Computing action overlap/reuse statistics...[/bold cyan]")
            
            # Load metadata and chunks to get all_saved_actions
            metadata = load_metadata(config.debug_data_dir)
            chunks = load_debug_chunks(config.debug_data_dir)
            timestamps_by_step = load_timestamps(config.debug_data_dir)
            
            all_saved_actions = []
            for chunk in chunks:
                chunk_data = unflatten_debug_data(chunk)
                saved_chunk_actions = get_saved_actions_from_debug(
                    chunk_data, config.action_dim
                )
                all_saved_actions.append(saved_chunk_actions)
            
            overlap_stats = compute_action_overlap_stats(
                gt_horse_tails,
                all_saved_actions,
                chunk_boundaries,
                timestamps_by_step,
            )
            
            # Print summary table
            console.print()
            table = Table(title="Action Reuse Potential Analysis", show_header=True, header_style="bold magenta")
            table.add_column("Distance Threshold", style="cyan", justify="right")
            table.add_column("Reusable Actions", style="green", justify="right")
            table.add_column("Total Comparisons", style="blue", justify="right")
            table.add_column("Reuse %", style="yellow", justify="right")
            
            for thresh in sorted(overlap_stats["per_threshold"].keys()):
                stats = overlap_stats["per_threshold"][thresh]
                table.add_row(
                    f"{thresh:.4f}",
                    str(stats["reusable_count"]),
                    str(stats["comparisons"]),
                    f"{stats['percentage']:.2f}%"
                )
            
            console.print(table)
            
            # Generate overlap visualization
            overlap_plot_path = config.debug_data_dir / "action_overlap_analysis.png"
            console.print(f"\n[bold]Generating overlap analysis plot: {overlap_plot_path}[/bold]")
            plot_overlap_analysis(overlap_stats, overlap_plot_path)
            
            # Generate action comparison plots with reusability highlighting for each threshold
            console.print(f"\n[bold cyan]Generating reusability-highlighted plots for each threshold...[/bold cyan]")
            for thresh in sorted(overlap_stats["per_threshold"].keys()):
                thresh_plot_path = config.debug_data_dir / f"action_comparison_reusable_thresh_{thresh:.4f}.png"
                console.print(f"  - Threshold {thresh:.4f}: {thresh_plot_path}")
                plot_action_comparison_with_reusability(
                    replay_actions,
                    saved_actions,
                    thresh_plot_path,
                    action_horizon=config.action_horizon,
                    gt_horse_tails=gt_horse_tails,
                    overlap_stats=overlap_stats,
                    distance_threshold=thresh,
                    chunk_boundaries=chunk_boundaries,
                )
            
            # Save detailed stats to JSON
            stats_json_path = config.debug_data_dir / "action_overlap_stats.json"
            with open(stats_json_path, "w") as f:
                json.dump(overlap_stats, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            console.print(f"[bold]Saved detailed stats to: {stats_json_path}[/bold]")

        console.print()
        console.print(
            "[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]"
        )
        if success:
            console.print(
                "[bold green]                    REPLAY SUCCESS ✓                        [/bold green]"
            )
        else:
            console.print(
                "[bold red]                    REPLAY FAILURE ✗                        [/bold red]"
            )
        console.print(
            "[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]"
        )

    except Exception as e:
        console.print(f"[bold red]Error during replay: {e}[/bold red]")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
