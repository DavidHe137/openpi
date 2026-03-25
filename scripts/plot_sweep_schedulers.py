"""
Plot sweep_schedulers results: success rate, starvation rate, and throughput
vs. number of robots, with one line per scheduler.

Usage:
    uv run python plot_sweep_schedulers.py [--data-dir DATA_DIR] [--out-dir OUT_DIR]
    uv run python plot_sweep_schedulers.py --h0-data-dir DIR0 --h1-data-dir DIR1 [--out-dir OUT_DIR]
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
    "greedy": {"color": "#555555", "marker": "o", "linestyle": "-"},
    "greedy_plus": {"color": "#000000", "marker": "D", "linestyle": "-"},
    "round_robin": {"color": "#e05c5c", "marker": "s", "linestyle": "-"},
    "wdrr": {"color": "#d18f1b", "marker": "P", "linestyle": "-"},
    "lookahead": {"color": "#1f77b4", "marker": "^", "linestyle": "-"},
}

SCHEDULER_LABELS = {
    "greedy": "Greedy",
    "greedy_plus": "Greedy+",
    "round_robin": "Round Robin",
    "wdrr": "WDRR",
    "lookahead": "Lookahead",
}

# ---------------------------------------------------------------------------
# Data loading
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

    return {
        "success_rate": summary["success"].mean(),
        "starvation_rate": summary["planner_starvation_rate"].mean(),
        "throughput": throughput,
    }


def load_all(data_dir: str) -> dict:
    """
    Returns nested dict: data[scheduler][num_robots] = list of metric dicts.
    """
    pattern = re.compile(r"scheduler_(?P<scheduler>\w+)_num_robots_(?P<num_robots>\d+)_run_(?P<run>\d+)$")
    data = defaultdict(lambda: defaultdict(list))

    for name in sorted(os.listdir(data_dir)):
        m = pattern.match(name)
        if not m:
            continue
        scheduler = m.group("scheduler")
        num_robots = int(m.group("num_robots"))
        run_dir = os.path.join(data_dir, name)
        metrics = load_run(run_dir)
        if metrics is None:
            print(f"  [skip] {name} (incomplete)")
            continue
        data[scheduler][num_robots].append(metrics)

    return data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _style_ax(ax, ylabel: str):
    ax.tick_params(labelsize=9)
    ax.grid(visible=True, linestyle=":", linewidth=0.6, alpha=0.6, color="gray")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Number of robots", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)


def plot_metric(data: dict, metric: str, ylabel: str, ax: plt.Axes):
    for scheduler, robot_data in sorted(data.items()):
        xs = sorted(robot_data.keys())
        # average over runs
        ys = [sum(r[metric] for r in robot_data[x]) / len(robot_data[x]) for x in xs]
        style = {"linewidth": 1.8, "markersize": 6, "label": SCHEDULER_LABELS.get(scheduler, scheduler)}
        style.update(SCHEDULER_STYLES.get(scheduler, {}))
        ax.plot(xs, ys, **style)

    _style_ax(ax, ylabel)


def plot_metric_compare(
    h0_data: dict,
    h1_data: dict,
    metric: str,
    ylabel: str,
    ax: plt.Axes,
    *,
    h0_label: str,
    h1_label: str,
):
    h0_linestyle = "-"
    # Use an explicit dotted pattern so the h1 style is clearly distinguishable.
    h1_linestyle = (0, (1.2, 2.0))

    schedulers = sorted(set(h0_data.keys()) | set(h1_data.keys()))
    for scheduler in schedulers:
        base_style = {
            "linewidth": 1.8,
            "markersize": 6,
            "color": SCHEDULER_STYLES.get(scheduler, {}).get("color", "#444444"),
            "marker": SCHEDULER_STYLES.get(scheduler, {}).get("marker", "o"),
        }
        pretty_name = SCHEDULER_LABELS.get(scheduler, scheduler)

        for data, horizon_label, linestyle in (
            (h0_data, h0_label, h0_linestyle),
            (h1_data, h1_label, h1_linestyle),
        ):
            robot_data = data.get(scheduler, {})
            if not robot_data:
                continue
            xs = sorted(robot_data.keys())
            ys = [sum(r[metric] for r in robot_data[x]) / len(robot_data[x]) for x in xs]
            style = dict(base_style)
            style.update({"linestyle": linestyle, "label": f"{pretty_name} ({horizon_label})"})
            ax.plot(xs, ys, **style)

    _style_ax(ax, ylabel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Single-run mode: directory containing sweep_schedulers run folders",
    )
    parser.add_argument("--h0-data-dir", default=None, help="Compare mode: first sweep directory")
    parser.add_argument("--h1-data-dir", default=None, help="Compare mode: second sweep directory")
    parser.add_argument("--h0-label", default="h0", help="Legend label for --h0-data-dir")
    parser.add_argument("--h1-label", default="h1", help="Legend label for --h1-data-dir")
    parser.add_argument("--out-dir", default="./plots", help="Directory to save output figures")
    args = parser.parse_args()

    compare_mode = (args.h0_data_dir is not None) or (args.h1_data_dir is not None)
    if compare_mode and (args.h0_data_dir is None or args.h1_data_dir is None):
        raise ValueError("When using compare mode, both --h0-data-dir and --h1-data-dir are required")

    if not compare_mode and args.data_dir is None:
        args.data_dir = "data/libero/sweep_schedulers"

    os.makedirs(args.out_dir, exist_ok=True)

    if compare_mode:
        print(f"Loading h0 data from {args.h0_data_dir} ...")
        h0_data = load_all(args.h0_data_dir)
        print(f"Loading h1 data from {args.h1_data_dir} ...")
        h1_data = load_all(args.h1_data_dir)
    else:
        print(f"Loading data from {args.data_dir} ...")
        data = load_all(args.data_dir)

    metrics = [
        ("success_rate", "Success rate", "success_rate"),
        ("starvation_rate", "Starvation rate", "starvation_rate"),
        ("throughput", "Throughput (successes / min)", "throughput"),
    ]

    for metric_key, ylabel, fname_suffix in metrics:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        if compare_mode:
            plot_metric_compare(
                h0_data,
                h1_data,
                metric_key,
                ylabel,
                ax,
                h0_label=args.h0_label,
                h1_label=args.h1_label,
            )
        else:
            plot_metric(data, metric_key, ylabel, ax)

        if metric_key in {"success_rate", "starvation_rate"}:
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

        ax.legend(
            fontsize=7,
            frameon=False,
            handlelength=3.0,
            labelspacing=0.25,
            borderaxespad=0.3,
        )
        fig.tight_layout()

        if compare_mode:
            out_path = os.path.join(
                args.out_dir,
                f"sweep_schedulers_{fname_suffix}_{args.h0_label}_vs_{args.h1_label}.png",
            )
        else:
            out_path = os.path.join(args.out_dir, f"sweep_schedulers_{fname_suffix}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"  Saved {out_path}")

        plt.close(fig)


if __name__ == "__main__":
    main()
