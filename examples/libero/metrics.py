"""Metrics and plotting utilities for LIBERO experiments."""

from typing import List, Dict, Callable, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dataclasses import asdict
from rich.console import Console
from rich.table import Table
from examples.libero.subscribers.saver import Result
from openpi_client.schemas import pathlib, ActionChunk


# =============================================================================
# Data Loading
# =============================================================================


def load_episodes(output_path: pathlib.Path) -> pd.DataFrame:
    """Load all metadata.json files into a DataFrame."""
    metadata_files = list(output_path.glob("**/metadata.json"))
    if not metadata_files:
        return pd.DataFrame()

    results: List[Result] = [Result.from_json(f) for f in metadata_files]
    return pd.DataFrame([asdict(result) for result in results])


def load_action_chunks(output_path: pathlib.Path) -> pd.DataFrame:
    """Load all action_chunks.parquet files with task metadata."""
    action_chunk_files = list(output_path.glob("**/action_chunks.parquet"))

    rows = []
    for action_chunk_file in action_chunk_files:
        episode_dir = action_chunk_file.parent
        metadata_file = episode_dir / "metadata.json"

        if not metadata_file.exists():
            print(f"Warning: metadata.json not found in {episode_dir}, skipping")
            continue

        result = Result.from_json(metadata_file)
        chunks = ActionChunk.from_parquet(action_chunk_file)

        for chunk in chunks:
            rows.append(
                {
                    "task_suite_name": result.task_suite_name,
                    "task_id": result.task_id,
                    "task_language": result.task_language,
                    "latency": chunk.latency,
                    "execution_horizon": chunk.execution_horizon,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Plot Primitives
# =============================================================================


def plot_histogram(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    xlabel: str,
    color: str = "steelblue",
    show_stats: bool = True,
) -> None:
    """Plot histogram with optional percentile markers."""
    ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor="black")

    if show_stats:
        stats = {
            "mean": np.mean(data),
            "median": np.median(data),
            "p90": np.percentile(data, 90),
            "p95": np.percentile(data, 95),
            "p99": np.percentile(data, 99),
        }
        ax.axvline(
            stats["mean"],
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {stats['mean']:.3f}",
        )
        ax.axvline(
            stats["median"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {stats['median']:.3f}",
        )
        ax.axvline(
            stats["p90"],
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"P90: {stats['p90']:.3f}",
        )
        ax.axvline(
            stats["p95"],
            color="purple",
            linestyle="-.",
            linewidth=2,
            label=f"P95: {stats['p95']:.3f}",
        )
        ax.axvline(
            stats["p99"],
            color="brown",
            linestyle="-",
            linewidth=1,
            label=f"P99: {stats['p99']:.3f}",
        )
        ax.legend(fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")


def plot_bar_chart(
    ax: plt.Axes,
    labels: List[str],
    values: np.ndarray,
    ylabel: str = "Value",
    title: str = "",
    counts: Optional[np.ndarray] = None,
    overall_line: Optional[Tuple[float, str]] = None,
    color_fn: Optional[Callable[[float], str]] = None,
) -> None:
    """Plot bar chart with optional annotations.

    Args:
        ax: Matplotlib axes
        labels: Bar labels
        values: Bar values
        ylabel: Y-axis label
        title: Plot title
        counts: Optional counts to show on bars as (n=X)
        overall_line: Optional (value, label) for horizontal line
        color_fn: Optional function(value) -> color for conditional coloring
    """
    bars = ax.bar(
        range(len(values)), values, color="steelblue", edgecolor="black", alpha=0.7
    )

    if color_fn:
        for bar, val in zip(bars, values):
            bar.set_color(color_fn(val))

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    if overall_line:
        value, label = overall_line
        ax.axhline(y=value, color="red", linestyle="--", linewidth=2, label=label)
        ax.legend()

    if counts is not None:
        for bar, val, count in zip(bars, values, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02,
                f"{val:.1%}\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def plot_grouped_violin(
    ax: plt.Axes,
    groups: Dict[str, Dict[str, np.ndarray]],
    ylabel: str = "Value",
    title: str = "",
    group_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Plot violin plots comparing multiple groups per category.

    Args:
        ax: Matplotlib axes
        groups: {category: {group_name: values}}
                e.g. {"Task 0": {"success": [...], "failure": [...]}}
        ylabel: Y-axis label
        title: Plot title
        group_colors: Optional {group_name: color} mapping
    """
    if group_colors is None:
        group_colors = {"success": "lightgreen", "failure": "lightcoral"}

    # Collect all group names for consistent ordering
    all_group_names = set()
    for category_groups in groups.values():
        all_group_names.update(category_groups.keys())
    group_names = sorted(all_group_names)

    positions = []
    labels = []
    all_data = []
    all_colors = []

    for i, (category, category_groups) in enumerate(groups.items()):
        base_pos = i * (len(group_names) + 1)
        for j, group_name in enumerate(group_names):
            if group_name in category_groups and len(category_groups[group_name]) > 0:
                all_data.append(category_groups[group_name])
                all_colors.append(group_colors.get(group_name, "lightblue"))
                positions.append(base_pos + j)
                labels.append(f"{category}\n({group_name})")

    if not all_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    # Plot violins
    parts = ax.violinplot(
        all_data, positions=positions, widths=0.8, showmeans=True, showmedians=True
    )

    # Color the violins
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(all_colors[i])
        pc.set_alpha(0.7)

    # Style the lines
    for partname in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if partname in parts:
            parts[partname].set_edgecolor("black")
            parts[partname].set_linewidth(1)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Legend
    legend_elements = [
        Patch(
            facecolor=group_colors.get(name, "lightblue"), edgecolor="black", label=name
        )
        for name in group_names
        if any(name in g for g in groups.values())
    ]
    if legend_elements:
        ax.legend(handles=legend_elements)


# =============================================================================
# Layout Helper
# =============================================================================


def plot_task_breakdown(
    df: pd.DataFrame,
    column: str,
    plot_fn: Callable[[plt.Axes, np.ndarray, str], None],
    title: str,
    filename: pathlib.Path,
) -> None:
    """Create grid: 'All Tasks' in first cell, then one cell per task.

    Args:
        df: DataFrame with 'task_suite_name' and 'task_id' columns
        column: Which column to extract values from
        plot_fn: Function(ax, data, subtitle) that plots on a single axes
        title: Overall figure title
        filename: Where to save
    """
    if df.empty:
        print(f"No data for {column}")
        return

    # Create task labels and group
    df = df.copy()
    if "task_language" in df.columns:
        df["task_label"] = (
            "Task " + df["task_id"].astype(str) + "\n" + df["task_language"].str[:30]
        )
    else:
        df["task_label"] = (
            df["task_suite_name"] + " - Task " + df["task_id"].astype(str)
        )
    grouped = df.groupby(["task_id", "task_label"], sort=True)

    n_tasks = len(grouped)
    n_plots = 1 + n_tasks  # overall + per-task

    # Grid dimensions
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))

    # Normalize axes to 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Plot overall
    plot_fn(axes.flat[0], df[column].values, "All Tasks Combined")

    # Plot per task
    for idx, ((task_id, task_label), group) in enumerate(grouped, start=1):
        plot_fn(axes.flat[idx], group[column].values, task_label)

    # Hide unused subplots
    for idx in range(n_plots, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=150)
    plt.close(fig)

    print(f"Saved {filename}")


# =============================================================================
# Plot Generators
# =============================================================================


def generate_latency_plot(output_path: pathlib.Path) -> None:
    """Latency distribution: overall + per-task."""
    df = load_action_chunks(output_path)
    plot_task_breakdown(
        df,
        column="latency",
        plot_fn=lambda ax, data, title: plot_histogram(
            ax, data, title, "Latency (seconds)"
        ),
        title="Action Chunk Latency Distribution",
        filename=output_path / "plots" / "action_chunk_latency.png",
    )


def generate_execution_horizon_plot(output_path: pathlib.Path) -> None:
    """Execution horizon distribution: overall + per-task."""
    df = load_action_chunks(output_path)
    plot_task_breakdown(
        df,
        column="execution_horizon",
        plot_fn=lambda ax, data, title: plot_histogram(
            ax, data, title, "Steps", color="coral", show_stats=False
        ),
        title="Execution Horizon Distribution",
        filename=output_path / "plots" / "execution_horizon.png",
    )


def generate_success_rate_plot(output_path: pathlib.Path) -> None:
    """Success rate bar chart by task."""
    df = load_episodes(output_path)
    if df.empty:
        print("No episode data for success rate plot")
        return

    # Aggregate by task
    summary = (
        df.groupby(["task_suite_name", "task_id", "task_language"])["success"]
        .agg(["mean", "count"])
        .reset_index()
    )
    summary["task_label"] = (
        "Task "
        + summary["task_id"].astype(str)
        + "\n"
        + summary["task_language"].str[:30]
    )

    overall_rate = df["success"].mean()

    def success_color(rate: float) -> str:
        if rate >= 0.8:
            return "green"
        elif rate >= 0.5:
            return "orange"
        return "red"

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_bar_chart(
        ax,
        labels=summary["task_label"].tolist(),
        values=summary["mean"].values,
        ylabel="Success Rate",
        title="Success Rate by Task",
        counts=summary["count"].values,
        overall_line=(overall_rate, f"Overall: {overall_rate:.2%}"),
        color_fn=success_color,
    )
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "success_rate.png", dpi=150)
    plt.close(fig)

    print(f"Saved {plots_dir / 'success_rate.png'}")


def generate_steps_plot(output_path: pathlib.Path) -> None:
    """Steps analysis: overall histograms + per-task violin plot."""
    df = load_episodes(output_path)
    if df.empty:
        print("No episode data for steps plot")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.3)
    fig.suptitle("Steps Taken Analysis", fontsize=16, fontweight="bold")

    # Overall distributions
    success_steps = df[df["success"]]["steps_taken"].values
    failure_steps = df[~df["success"]]["steps_taken"].values

    ax_success = fig.add_subplot(gs[0, 0])
    if len(success_steps) > 0:
        plot_histogram(
            ax_success,
            success_steps,
            "Successful Episodes",
            "Steps",
            color="green",
            show_stats=False,
        )
    else:
        ax_success.text(
            0.5,
            0.5,
            "No successful episodes",
            ha="center",
            va="center",
            transform=ax_success.transAxes,
        )
        ax_success.set_title("Successful Episodes")

    ax_failure = fig.add_subplot(gs[0, 1])
    if len(failure_steps) > 0:
        plot_histogram(
            ax_failure,
            failure_steps,
            "Failed Episodes",
            "Steps",
            color="red",
            show_stats=False,
        )
    else:
        ax_failure.text(
            0.5,
            0.5,
            "No failed episodes",
            ha="center",
            va="center",
            transform=ax_failure.transAxes,
        )
        ax_failure.set_title("Failed Episodes")

    # Per-task violin plot
    df["task_label"] = (
        "Task " + df["task_id"].astype(str) + "\n" + df["task_language"].str[:30]
    )
    groups = {}
    for (task_id, task_label), group in df.groupby(
        ["task_id", "task_label"], sort=True
    ):
        groups[task_label] = {
            "success": group[group["success"]]["steps_taken"].values,
            "failure": group[~group["success"]]["steps_taken"].values,
        }

    ax_violin = fig.add_subplot(gs[1, :])
    plot_grouped_violin(
        ax_violin, groups, ylabel="Steps", title="Steps by Task (Success vs Failure)"
    )

    plt.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "steps_taken.png", dpi=150)
    plt.close(fig)

    print(f"Saved {plots_dir / 'steps_taken.png'}")


def generate_per_robot_success_rate_plot(output_path: pathlib.Path) -> None:
    """Success rate bar chart broken down by robot."""
    df = load_episodes(output_path)
    if df.empty:
        print("No episode data for per-robot success rate plot")
        return

    robot_summary = (
        df.groupby("robot_idx")["success"].agg(["mean", "count"]).reset_index()
    )
    robot_summary = robot_summary.sort_values("robot_idx")

    overall_rate = df["success"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        robot_summary["robot_idx"].astype(str),
        robot_summary["mean"],
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    ax.axhline(
        y=overall_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall: {overall_rate:.2%}",
    )

    for bar, rate, count in zip(bars, robot_summary["mean"], robot_summary["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{rate:.0%}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Robot Index", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Per-Robot Success Rate", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "per_robot_success_rate.png", dpi=150)
    plt.close(fig)
    print(f"Saved {plots_dir / 'per_robot_success_rate.png'}")


def generate_per_robot_completion_speed_plot(output_path: pathlib.Path) -> None:
    """Box plot of steps taken per robot (lower = faster completion)."""
    df = load_episodes(output_path)
    if df.empty:
        print("No episode data for per-robot completion speed plot")
        return

    robots = sorted(df["robot_idx"].unique())
    data_by_robot = [df[df["robot_idx"] == r]["steps_taken"].values for r in robots]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data_by_robot, labels=[str(r) for r in robots], patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.set_xlabel("Robot Index", fontsize=12)
    ax.set_ylabel("Steps Taken", fontsize=12)
    ax.set_title("Per-Robot Task Completion Speed", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "per_robot_completion_speed.png", dpi=150)
    plt.close(fig)
    print(f"Saved {plots_dir / 'per_robot_completion_speed.png'}")


def generate_all_plots(output_path: pathlib.Path) -> None:
    """Generate all plots."""
    print("Generating plots...")
    generate_latency_plot(output_path)
    generate_execution_horizon_plot(output_path)
    generate_success_rate_plot(output_path)
    generate_steps_plot(output_path)
    generate_per_robot_success_rate_plot(output_path)
    generate_per_robot_completion_speed_plot(output_path)
    print("Done!")


# =============================================================================
# Metrics Calculation (console output + CSV)
# =============================================================================


def calculate_metrics(output_path: pathlib.Path) -> None:
    """Aggregate results and display summary table."""
    df = load_episodes(output_path)
    if df.empty:
        print("No results found")
        return

    df.to_csv(output_path / "results.csv", index=False)

    summary = df.groupby(["task_suite_name", "task_id"]).agg({"success": "mean"})
    summary.reset_index().to_csv(output_path / "summary.csv", index=False)

    # Display with rich
    console = Console()
    table = Table(title="Task Success Summary")
    table.add_column("Task Suite", style="cyan")
    table.add_column("Task ID", style="magenta")
    table.add_column("Success Rate", style="green")

    for _, row in summary.reset_index().iterrows():
        table.add_row(
            str(row["task_suite_name"]),
            str(row["task_id"]),
            f"{row['success']:.2%}",
        )

    console.print(table)
    console.print(
        f"\n[bold green]Total success rate: {summary['success'].mean():.2%}[/bold green]"
    )
