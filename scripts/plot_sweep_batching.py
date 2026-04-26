"""
Plot sweep_batching results: success rate, starvation rate, and throughput
vs. max batch size, with one line per scheduler, faceted by number of robots.

Usage:
    uv run python scripts/plot_sweep_batching.py [--data-dir DATA_DIR] [--out-dir OUT_DIR]
"""

import argparse
from collections import defaultdict
import json
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCHEDULER_STYLES = {
    "greedy-action": {"color": "#555555", "marker": "o", "linestyle": "-"},
    "greedy-deadline": {"color": "#e05c5c", "marker": "s", "linestyle": "-"},
    "fixed-size-greedy": {"color": "#1f77b4", "marker": "^", "linestyle": "-"},
    "fixed-size": {"color": "#1f77b4", "marker": "^", "linestyle": "-"},
    "true-max-batch": {"color": "#1f77b4", "marker": "^", "linestyle": "-"},
    "max-batch": {"color": "#1f77b4", "marker": "^", "linestyle": "-"},
}

SCHEDULER_LABELS = {
    "greedy-action": "Greedy (action)",
    "greedy-deadline": "Greedy (deadline)",
    "fixed-size-greedy": "Fixed-size greedy",
    "fixed-size": "Fixed-size",
    "true-max-batch": "True max batch",
    "max-batch": "True max batch",
}

# ---------------------------------------------------------------------------
# Data loading  (identical helpers to plot_sweep_schedulers.py)
# ---------------------------------------------------------------------------


def _load_experiment_duration(run_dir: str) -> float | None:
    """Compute total experiment wall-clock duration from timestamps.csv files."""
    t_min = float("inf")
    t_max = float("-inf")
    found = False
    for dirpath, _, filenames in os.walk(run_dir):
        if "timestamps.csv" in filenames:
            ts = pd.read_csv(os.path.join(dirpath, "timestamps.csv"), usecols=["timestamp"])
            if ts.empty:
                continue
            t_min = min(t_min, float(ts["timestamp"].iloc[0]))
            t_max = max(t_max, float(ts["timestamp"].iloc[-1]))
            found = True
    return (t_max - t_min) if found else None


def _load_server_duration(run_dir: str) -> float | None:
    """Compute wall-clock duration from server_metrics_history.json as a fallback."""
    history_path = os.path.join(run_dir, "server_metrics_history.json")
    if not os.path.exists(history_path):
        return None

    try:
        with open(history_path) as f:
            history = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    start = history.get("start_time")
    end = history.get("end_time")
    if start is None or end is None or end <= start:
        return None
    return end - start


def _canonical_scheduler(run_dir: str, folder_scheduler: str) -> str:
    """Prefer the scheduler reported by the server when available."""
    metadata_path = os.path.join(run_dir, "server_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path) as f:
                scheduler = json.load(f).get("scheduling_algorithm")
            if scheduler:
                return scheduler
        except (OSError, json.JSONDecodeError):
            pass

    if folder_scheduler == "fixed-size":
        return "true-max-batch"
    return folder_scheduler


def load_run(run_dir: str) -> dict | None:
    """Load metrics from a single run directory.

    A partially completed sweep may have only some artifacts. Load each metric
    independently so existing points still appear in plots.
    """
    results_path = os.path.join(run_dir, "results.csv")
    summary_path = os.path.join(run_dir, "summary.csv")

    if not os.path.exists(results_path) and not os.path.exists(summary_path):
        return None

    results = pd.read_csv(results_path) if os.path.exists(results_path) else None
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

    result = {}

    if summary is not None and not summary.empty:
        if "success" in summary:
            result["success_rate"] = summary["success"].mean()
        if "planner_starvation_rate" in summary:
            result["starvation_rate"] = summary["planner_starvation_rate"].mean()
        if "post_first_starvation_rate" in summary:
            result["post_first_starvation_rate"] = summary["post_first_starvation_rate"].mean()

    if results is not None and not results.empty:
        if "success_rate" not in result and "success" in results:
            result["success_rate"] = results["success"].mean()
        if "starvation_rate" not in result and {"starvation_steps", "observed_steps"} <= set(results.columns):
            observed_steps = results["observed_steps"].sum()
            if observed_steps > 0:
                result["starvation_rate"] = results["starvation_steps"].sum() / observed_steps
        if "post_first_starvation_rate" not in result and {
            "post_first_starvation_steps",
            "post_first_observed_steps",
        } <= set(results.columns):
            observed_steps = results["post_first_observed_steps"].sum()
            if observed_steps > 0:
                result["post_first_starvation_rate"] = results["post_first_starvation_steps"].sum() / observed_steps

        total_time_s = _load_experiment_duration(run_dir)
        if total_time_s is None:
            total_time_s = _load_server_duration(run_dir)
        if total_time_s is not None and total_time_s > 0 and "success" in results:
            result["throughput"] = results["success"].sum() / total_time_s * 60  # successes per minute

    return result or None


def load_all(data_dir: str) -> dict:
    """
    Returns nested dict: data[scheduler][num_robots][max_batch] = list of metric dicts.
    """
    pattern = re.compile(
        r"scheduler_(?P<scheduler>[\w-]+)_max_batch_(?P<max_batch>\d+)_num_robots_(?P<num_robots>\d+)$"
    )
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for name in sorted(os.listdir(data_dir)):
        m = pattern.match(name)
        if not m:
            continue
        max_batch = int(m.group("max_batch"))
        num_robots = int(m.group("num_robots"))
        run_dir = os.path.join(data_dir, name)
        scheduler = _canonical_scheduler(run_dir, m.group("scheduler"))
        metrics = load_run(run_dir)
        if metrics is None:
            print(f"  [skip] {name} (incomplete)")
            continue
        data[scheduler][num_robots][max_batch].append(metrics)

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _style_ax(ax, ylabel: str):
    ax.tick_params(labelsize=9)
    ax.grid(visible=True, linestyle=":", linewidth=0.6, alpha=0.6, color="gray")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Max batch size", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)


def plot_metric(data: dict, metric: str, ylabel: str, axes) -> bool:
    """
    axes: array of Axes, one per num_robots value (sorted ascending).
    Each subplot shows metric vs max_batch with one line per scheduler.
    """
    all_num_robots = sorted({nr for sched_data in data.values() for nr in sched_data})
    plotted = False

    for ax, num_robots in zip(axes, all_num_robots, strict=True):
        for scheduler, sched_data in sorted(data.items()):
            if scheduler == "greedy-action":
                continue
            batch_data = sched_data.get(num_robots, {})
            if not batch_data:
                continue
            xs = []
            ys = []
            for x in sorted(batch_data.keys()):
                vals = [r[metric] for r in batch_data[x] if metric in r]
                if not vals:
                    continue
                xs.append(x)
                ys.append(sum(vals) / len(vals))
            if not xs:
                continue
            style = {
                "linewidth": 1.8,
                "markersize": 6,
                "label": SCHEDULER_LABELS.get(scheduler, scheduler),
            }
            style.update(SCHEDULER_STYLES.get(scheduler, {}))
            ax.plot(xs, ys, **style)
            plotted = True

        ax.set_title(f"{num_robots} robots", fontsize=9)
        _style_ax(ax, ylabel if ax is axes[0] else "")
        # Only show y-label on leftmost subplot
        if ax is not axes[0]:
            ax.set_ylabel("")

    return plotted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/libero/batching5",
        help="Directory containing sweep_batching run folders",
    )
    parser.add_argument("--out-dir", default="./plots", help="Directory to save output figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data from {args.data_dir} ...")
    data = load_all(args.data_dir)

    all_num_robots = sorted({nr for sched_data in data.values() for nr in sched_data})
    n_cols = len(all_num_robots)
    if n_cols == 0:
        raise SystemExit(f"No plottable runs found in {args.data_dir}")

    metrics = [
        ("success_rate", "Success rate", "success_rate"),
        ("starvation_rate", "Starvation rate", "starvation_rate"),
        (
            "post_first_starvation_rate",
            "Starvation rate (excl. pre-first-action)",
            "post_first_starvation_rate",
        ),
        ("throughput", "Throughput (successes / min)", "throughput"),
    ]

    for metric_key, ylabel, fname_suffix in metrics:
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5), sharey=True)
        if n_cols == 1:
            axes = [axes]

        if not plot_metric(data, metric_key, ylabel, axes):
            print(f"  [skip] {metric_key} (no available values)")
            plt.close(fig)
            continue

        if metric_key in {"success_rate", "starvation_rate", "post_first_starvation_rate"}:
            # axes[0].set_ylim(0, 1.05)
            axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

        # Shared legend on the last subplot
        handles, labels = axes[-1].get_legend_handles_labels()
        axes[-1].legend(
            handles,
            labels,
            fontsize=7,
            frameon=False,
            handlelength=3.0,
            labelspacing=0.25,
            borderaxespad=0.3,
        )

        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"sweep_batching_{fname_suffix}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"  Saved {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
