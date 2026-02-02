from typing import List, Dict
import pandas as pd
from dataclasses import asdict
from rich.console import Console
from rich.table import Table
from examples.libero.subscribers.saver import Result
from openpi_client.schemas import pathlib
from dataclasses import dataclass


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


# TODO: more aggregated metrics, grouped by task suite and task id on single pdf with subplots. also have a single plot for all tasks combined.
# distribution of action chunk latencies with mean, median, 90th percentile, 95th percentile, 99th percentile
# distribution of action chunk execution horizons with mean, median, 90th percentile, 95th percentile, 99th percentile

# NOTE: need to do more thinking on this. how to plot schedule of events (inferences, action chunks, etc.) over time?
