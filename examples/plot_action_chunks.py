"""
Plot action chunks saved by the server-side ActionChunkDebugRecorder.

Usage:
    python examples/plot_action_chunks.py action_chunks.parquet
    python examples/plot_action_chunks.py action_chunks.parquet --out debug_plot.png
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from openpi_client.schemas import ActionChunk
import pandas as pd

ACTION_DIM_NAMES = ["X", "Y", "Z", "RX", "RY", "RZ", "Gripper"]


def load_chunks(path: pathlib.Path) -> list[ActionChunk]:
    return ActionChunk.from_parquet(path)


def _to_array(val) -> np.ndarray:
    """Parquet round-trips numpy arrays as lists-of-lists; restore them."""
    return np.array([np.array(x) for x in val])


def plot_action_chunks(parquet_path: pathlib.Path, out_path: pathlib.Path) -> None:
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("action_start_step").reset_index(drop=True)

    n_chunks = len(df)
    colors = cm.tab20(np.linspace(0, 1, n_chunks))

    # Reconstruct per-chunk action arrays: list of (horizon, action_dim)
    chunk_actions: list[np.ndarray] = [_to_array(df.iloc[i]["actions"]) for i in range(n_chunks)]
    action_dim = chunk_actions[0].shape[1]
    dim_names = ACTION_DIM_NAMES[:action_dim] + [f"Dim {i}" for i in range(len(ACTION_DIM_NAMES), action_dim)]

    latencies = (df["response_timestamp"] - df["request_timestamp"]).values * 1000  # ms

    # ── 1. Action trajectories ─────────────────────────────────────────────
    fig_actions, axes = plt.subplots(action_dim, 1, figsize=(16, 2.8 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]

    for i, (row, actions, color) in enumerate(zip(df.itertuples(), chunk_actions, colors)):
        start = row.action_start_step
        steps = np.arange(start, start + len(actions))
        for dim_idx, ax in enumerate(axes):
            ax.plot(steps, actions[:, dim_idx], color=color, linewidth=1.5, alpha=0.85)
            ax.axvline(start, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    for dim_idx, ax in enumerate(axes):
        ax.set_ylabel(dim_names[dim_idx], fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Action step")
    fig_actions.suptitle(
        f"Action chunks  ({n_chunks} chunks, horizon={df['execution_horizon'].iloc[0]})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    actions_out = out_path.with_name(out_path.stem + "_actions" + out_path.suffix)
    fig_actions.savefig(actions_out, dpi=150, bbox_inches="tight")
    print(f"Saved: {actions_out}")
    plt.close(fig_actions)

    # ── 2. Latency per chunk ───────────────────────────────────────────────
    fig_lat, ax_lat = plt.subplots(figsize=(12, 3.5))
    ax_lat.bar(range(n_chunks), latencies, color=colors, edgecolor="none")
    ax_lat.axhline(
        np.mean(latencies), color="red", linestyle="--", linewidth=1.2, label=f"Mean {np.mean(latencies):.1f} ms"
    )
    ax_lat.set_xlabel("Chunk index")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Server inference latency per chunk (request → response)")
    ax_lat.legend()
    ax_lat.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    latency_out = out_path.with_name(out_path.stem + "_latency" + out_path.suffix)
    fig_lat.savefig(latency_out, dpi=150, bbox_inches="tight")
    print(f"Saved: {latency_out}")
    plt.close(fig_lat)

    # ── 3. Chunk timeline (action_start_step vs observation_step) ─────────
    fig_timeline, ax_t = plt.subplots(figsize=(12, 3))
    ax_t.scatter(range(n_chunks), df["observation_step"], s=20, label="observation_step", zorder=3)
    ax_t.scatter(range(n_chunks), df["action_start_step"], s=20, label="action_start_step", marker="x", zorder=3)
    skew = (df["action_start_step"] - df["observation_step"]).values
    ax_t.fill_between(
        range(n_chunks),
        df["observation_step"],
        df["action_start_step"],
        alpha=0.2,
        label=f"skew (mean={np.mean(skew):.1f})",
    )
    ax_t.set_xlabel("Chunk index")
    ax_t.set_ylabel("Step")
    ax_t.set_title("Observation step vs action_start_step per chunk")
    ax_t.legend(fontsize=9)
    ax_t.grid(True, alpha=0.3)
    plt.tight_layout()

    timeline_out = out_path.with_name(out_path.stem + "_timeline" + out_path.suffix)
    fig_timeline.savefig(timeline_out, dpi=150, bbox_inches="tight")
    print(f"Saved: {timeline_out}")
    plt.close(fig_timeline)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot action chunks from a debug parquet file")
    parser.add_argument("parquet", type=str, help="Path to action_chunks.parquet")
    parser.add_argument(
        "--out", type=str, default=None, help="Output image base path (default: <parquet_dir>/action_chunks_plot.png)"
    )
    args = parser.parse_args()

    parquet_path = pathlib.Path(args.parquet)
    out_path = pathlib.Path(args.out) if args.out else parquet_path.with_name("action_chunks_plot.png")

    plot_action_chunks(parquet_path, out_path)


if __name__ == "__main__":
    main()
