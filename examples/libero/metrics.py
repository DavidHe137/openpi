from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass
from rich.console import Console
from rich.table import Table
from examples.libero.subscribers.saver import Result
from openpi_client.schemas import pathlib, ActionChunk


@dataclass
class Metrics:
    # individual task performance degradation
    success_rate: float
    success_rate_by_task: Dict[str, float]  # task.language -> success_rate
    average_steps_taken_for_success: float
    average_steps_taken_for_failure: float
    average_steps_taken_for_success_by_task: Dict[
        str, float
    ]  # task.language -> average_steps_taken_for_success
    average_steps_taken_for_failure_by_task: Dict[
        str, float
    ]  # task.language -> average_steps_taken_for_failure

    # overall throughput
    n_robots: int
    total_successes: int
    total_failures: int
    total_time: float  # NOTE: not super accurate, as env startup takes time, can refactor this in the future
    successful_tasks_per_second: float


def calculate_metrics(output_path: pathlib.Path) -> None:
    """Aggregate results from all metadata files in the output path."""
    metadata_files = list(output_path.glob("**/metadata.json"))

    results: List[Result] = []
    for metadata_file in metadata_files:
        result = Result.from_json(metadata_file)
        results.append(result)

    results_df = pd.DataFrame([asdict(result) for result in results])
    results_df.to_csv(output_path / "results.csv", index=False)
    summary = results_df.groupby(["task_suite_name", "task_id"]).agg(
        {
            "success": "mean",
        }
    )
    summary.reset_index().to_csv(output_path / "summary.csv", index=False)

    # Display results using rich
    console = Console()
    table = Table(title="Task Success Summary")
    table.add_column("Task Suite", style="cyan")
    table.add_column("Task ID", style="magenta")
    table.add_column("Success Rate", style="green")

    for _, row in summary.reset_index().iterrows():
        table.add_row(
            str(row["task_suite_name"]), str(row["task_id"]), f"{row['success']:.2%}"
        )

    console.print(table)
    console.print(
        f"\n[bold green]Total success rate: {summary['success'].mean():.2%}[/bold green]"
    )


def load_action_chunks(output_path: pathlib.Path) -> pd.DataFrame:
    """Load all action chunk data and associate with task metadata."""
    action_chunk_files = list(output_path.glob("**/action_chunks.parquet"))

    rows = []
    for action_chunk_file in action_chunk_files:
        # Parse task info from directory structure: robot_idx/episode_idx_suite_taskid_status
        episode_dir = action_chunk_file.parent
        parts = episode_dir.name.split("_")
        task_suite_name = parts[1] if len(parts) > 1 else "unknown"
        task_id = int(parts[2]) if len(parts) > 2 else -1

        chunks = ActionChunk.from_parquet(action_chunk_file)
        for chunk in chunks:
            rows.append(
                {
                    "task_suite_name": task_suite_name,
                    "task_id": task_id,
                    "latency": chunk.latency,
                    "execution_horizon": chunk.execution_horizon,
                }
            )

    return pd.DataFrame(rows)


def compute_percentiles(data: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a distribution."""
    return {
        "mean": np.mean(data).item(),
        "median": np.median(data).item(),
        "p90": np.percentile(data, 90).item(),
        "p95": np.percentile(data, 95).item(),
        "p99": np.percentile(data, 99).item(),
    }


def plot_distribution(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    xlabel: str,
    color: str = "steelblue",
) -> None:
    """Plot a histogram with percentile annotations."""
    stats = compute_percentiles(data)

    ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor="black")
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

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)


def generate_distribution_plots(output_path: pathlib.Path) -> None:
    """Generate PNG plots for latency and execution horizon distributions."""
    df = load_action_chunks(output_path)

    if df.empty:
        print("No action chunk data found.")
        return

    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Combined distributions for all tasks
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("All Tasks Combined", fontsize=14, fontweight="bold")

    plot_distribution(
        axes[0],
        df["latency"].values,
        "Action Chunk Latency Distribution",
        "Latency (seconds)",
    )
    plot_distribution(
        axes[1],
        df["execution_horizon"].values,
        "Execution Horizon Distribution",
        "Execution Horizon (steps)",
        color="coral",
    )

    plt.tight_layout()
    fig.savefig(plots_dir / "all_tasks_distributions.png", dpi=150)
    plt.close(fig)

    # Per task suite and task id
    grouped = df.groupby(["task_suite_name", "task_id"])
    for (suite_name, task_id), group in grouped:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{suite_name} - Task {task_id}", fontsize=14, fontweight="bold")

        plot_distribution(
            axes[0],
            group["latency"].values,
            "Action Chunk Latency Distribution",
            "Latency (seconds)",
        )
        plot_distribution(
            axes[1],
            group["execution_horizon"].values,
            "Execution Horizon Distribution",
            "Execution Horizon (steps)",
            color="coral",
        )

        plt.tight_layout()
        fig.savefig(
            plots_dir / f"{suite_name}_task{task_id}_distributions.png", dpi=150
        )
        plt.close(fig)

    print(f"Distribution plots saved to {plots_dir}")
