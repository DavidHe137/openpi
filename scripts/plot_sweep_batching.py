"""
Plot sweep_batching results: success rate, starvation rate, and throughput
vs. max batch size, with one line per scheduler, faceted by number of robots.

Usage:
    uv run python scripts/plot_sweep_batching.py [--data-dir DATA_DIR] [--out-dir OUT_DIR]
"""

import argparse
from collections import defaultdict
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
}

SCHEDULER_LABELS = {
    "greedy-action": "Greedy (action)",
    "greedy-deadline": "Greedy (deadline)",
    "fixed-size-greedy": "Fixed-size greedy",
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


def load_run(run_dir: str) -> dict | None:
    """Load metrics from a single run directory. Returns None if incomplete."""
    results_path = os.path.join(run_dir, "results.csv")
    summary_path = os.path.join(run_dir, "summary.csv")

    for p in (results_path, summary_path):
        if not os.path.exists(p):
            return None

    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path)

    total_time_s = _load_experiment_duration(run_dir)
    if total_time_s is None:
        return None

    n_successes = results["success"].sum()
    throughput = n_successes / total_time_s * 60  # successes per minute

    result = {
        "success_rate": summary["success"].mean(),
        "starvation_rate": summary["planner_starvation_rate"].mean(),
        "throughput": throughput,
    }
    if "post_first_starvation_rate" in summary.columns:
        result["post_first_starvation_rate"] = summary["post_first_starvation_rate"].mean()
    return result


def load_all(data_dir: str) -> dict:
    """
    Returns nested dict: data[scheduler][num_robots][max_batch] = list of metric dicts.
    """
    pattern = re.compile(
        r"scheduler_(?P<scheduler>[\w-]+)_max_batch_(?P<max_batch>\d+)_num_robots_(?P<num_robots>\d+)_run_(?P<run>\d+)$"
    )
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for name in sorted(os.listdir(data_dir)):
        m = pattern.match(name)
        if not m:
            continue
        scheduler = m.group("scheduler")
        max_batch = int(m.group("max_batch"))
        num_robots = int(m.group("num_robots"))
        run_dir = os.path.join(data_dir, name)
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


def plot_metric(data: dict, metric: str, ylabel: str, axes):
    """
    axes: array of Axes, one per num_robots value (sorted ascending).
    Each subplot shows metric vs max_batch with one line per scheduler.
    """
    all_num_robots = sorted({nr for sched_data in data.values() for nr in sched_data})

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

        ax.set_title(f"{num_robots} robots", fontsize=9)
        _style_ax(ax, ylabel if ax is axes[0] else "")
        # Only show y-label on leftmost subplot
        if ax is not axes[0]:
            ax.set_ylabel("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/libero/batching",
        help="Directory containing sweep_batching run folders",
    )
    parser.add_argument("--out-dir", default="./plots", help="Directory to save output figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data from {args.data_dir} ...")
    data = load_all(args.data_dir)

    all_num_robots = sorted({nr for sched_data in data.values() for nr in sched_data})
    n_cols = len(all_num_robots)

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

        plot_metric(data, metric_key, ylabel, axes)

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
