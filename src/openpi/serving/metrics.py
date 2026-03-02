"""Metrics collection and visualization for the websocket policy server."""

from collections import defaultdict
from collections import deque
from collections.abc import Sequence
import csv
from dataclasses import dataclass
from dataclasses import field
import logging
from pathlib import Path
import time
from typing import Any

import matplotlib  # noqa: ICN001
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Tracks timing information for a single request."""

    request_id: int
    arrival_time: float  # When received by websocket (time.perf_counter())
    queued_time: float  # When sent to worker via ZeroMQ
    processing_start_time: float | None = None  # When batch processing started
    finished_time: float | None = None  # When response sent back

    @property
    def queue_wait_time(self) -> float | None:
        """Time spent waiting in queue before processing."""
        if self.processing_start_time is None:
            return None
        return self.processing_start_time - self.queued_time

    @property
    def end_to_end_latency(self) -> float | None:
        """Total latency from arrival to finished."""
        if self.finished_time is None:
            return None
        return self.finished_time - self.arrival_time


@dataclass
class BatchMetrics:
    """Tracks metrics for a single batch."""

    batch_id: int
    processing_start_time: float
    processing_end_time: float
    num_real_requests: int
    total_batch_size: int
    request_ids: list[int]
    robot_ids: list[str] = field(default_factory=list)
    start_steps: list[int] = field(default_factory=list)
    execution_horizons: list[int] = field(default_factory=list)

    @property
    def batch_processing_time(self) -> float:
        """Total time to process this batch."""
        return self.processing_end_time - self.processing_start_time

    @property
    def batch_utilization(self) -> float:
        """Ratio of real requests to total batch size."""
        return self.num_real_requests / self.total_batch_size if self.total_batch_size > 0 else 0.0


@dataclass
class RobotSchedulingState:
    """Per-robot tracking for scheduling visualizations."""

    last_start_step: int = 0
    last_execution_horizon: int = 0
    last_response_time: float = 0.0
    total_starvations: int = 0
    total_wasted_actions: int = 0


@dataclass
class MetricsCollector:
    """Aggregates and manages metrics collection."""

    # Per-request tracking
    request_metrics: dict[int, RequestMetrics] = field(default_factory=dict)

    # Per-batch tracking
    batch_metrics: list[BatchMetrics] = field(default_factory=list)

    # Per-robot scheduling state
    robot_states: dict[str, RobotSchedulingState] = field(default_factory=dict)

    # Rolling window of recent latencies for logging (max 10)
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=10))

    # Start time for throughput calculations
    start_time: float = field(default_factory=time.perf_counter)
    first_arrival_time: float = field(default_factory=lambda: float("inf"))
    last_arrival_time: float = field(default_factory=lambda: float("-inf"))

    def add_request_arrival(self, request_id: int, arrival_time: float) -> None:
        """Record when a request arrived at the websocket."""
        normalized_arrival_time = arrival_time - self.start_time
        self.request_metrics[request_id] = RequestMetrics(
            request_id=request_id,
            arrival_time=normalized_arrival_time,
            queued_time=0.0,  # Will be set when queued
        )
        self.first_arrival_time = min(self.first_arrival_time, normalized_arrival_time)
        self.last_arrival_time = max(self.last_arrival_time, normalized_arrival_time)

    def add_request_queued(self, request_id: int, queued_time: float) -> None:
        """Record when a request was sent to worker queue."""
        if request_id in self.request_metrics:
            self.request_metrics[request_id].queued_time = queued_time - self.start_time

    def add_batch_start(self, request_ids: Sequence[int], start_time: float) -> None:
        """Record when batch processing started."""
        for req_id in request_ids:
            if req_id in self.request_metrics:
                self.request_metrics[req_id].processing_start_time = start_time - self.start_time

    def add_request_finished(self, request_id: int, finished_time: float) -> None:
        """Record when a request response was sent."""
        if request_id in self.request_metrics:
            metrics = self.request_metrics[request_id]
            metrics.finished_time = finished_time - self.start_time

            # Add to recent latencies for logging
            if metrics.end_to_end_latency is not None:
                self.recent_latencies.append(metrics.end_to_end_latency)

    def add_batch_metrics(self, batch_metric: BatchMetrics) -> None:
        """Record batch-level metrics and update per-robot scheduling state."""
        normalized = BatchMetrics(
            batch_id=batch_metric.batch_id,
            processing_start_time=batch_metric.processing_start_time - self.start_time,
            processing_end_time=batch_metric.processing_end_time - self.start_time,
            num_real_requests=batch_metric.num_real_requests,
            total_batch_size=batch_metric.total_batch_size,
            request_ids=batch_metric.request_ids,
            robot_ids=batch_metric.robot_ids,
            start_steps=batch_metric.start_steps,
            execution_horizons=batch_metric.execution_horizons,
        )
        self.batch_metrics.append(normalized)

        response_time = normalized.processing_end_time
        for robot_id, start_step, exec_horizon in zip(
            normalized.robot_ids, normalized.start_steps, normalized.execution_horizons, strict=True
        ):
            state = self.robot_states.setdefault(robot_id, RobotSchedulingState())

            if state.last_execution_horizon > 0:
                actions_consumed = start_step - state.last_start_step
                actions_remaining = state.last_execution_horizon - actions_consumed

                if actions_remaining < 0:
                    state.total_starvations += abs(actions_remaining)
                elif actions_remaining > 0:
                    state.total_wasted_actions += actions_remaining

            state.last_start_step = start_step
            state.last_execution_horizon = exec_horizon
            state.last_response_time = response_time

    def get_recent_latency_stats(self) -> dict[str, float]:
        """Get statistics for recent latencies (1, 5, 10 samples)."""
        if not self.recent_latencies:
            return {"avg_1": 0.0, "avg_5": 0.0, "avg_10": 0.0}

        latencies = list(self.recent_latencies)
        return {
            "avg_1": float(np.mean(latencies[-1:])) if len(latencies) >= 1 else 0.0,
            "avg_5": float(np.mean(latencies[-5:])) if len(latencies) >= 5 else float(np.mean(latencies)),
            "avg_10": float(np.mean(latencies)),
        }

    def compute_aggregated_metrics(self) -> dict[str, Any]:
        """Compute aggregated metrics for plotting."""
        if not self.batch_metrics:
            return {}

        batches_per_second = defaultdict(list)
        for batch in self.batch_metrics:
            time_bucket = int(batch.processing_start_time)
            batches_per_second[time_bucket].append(batch)

        min_second_bucket = int(self.first_arrival_time)
        max_second_bucket = int(self.last_arrival_time)

        completed_requests = [m for m in self.request_metrics.values() if m.end_to_end_latency is not None]

        return {
            "batch_times": [b.batch_processing_time for b in self.batch_metrics],
            "batch_utilizations": [b.batch_utilization for b in self.batch_metrics],
            "real_throughputs": [
                sum(b.num_real_requests for b in batches_per_second[time_bucket])
                for time_bucket in range(min_second_bucket, max_second_bucket + 1)
            ],
            "total_throughputs": [
                sum(b.total_batch_size for b in batches_per_second[time_bucket])
                for time_bucket in range(min_second_bucket, max_second_bucket + 1)
            ],
            "timestamps": [b.processing_start_time for b in self.batch_metrics],
            "latencies": [m.end_to_end_latency for m in completed_requests],
            "queue_waits": [m.queue_wait_time for m in completed_requests if m.queue_wait_time is not None],
            "completed_requests": len(completed_requests),
            "total_batches": len(self.batch_metrics),
        }


def plot_metrics(metrics: MetricsCollector, output_dir: str) -> None:
    """Generate and save metrics plots."""
    sns.set_style("darkgrid")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = metrics.compute_aggregated_metrics()

    if not data["timestamps"]:
        logger.warning("No metrics data to plot")
        return

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle("Websocket Policy Server Metrics", fontsize=16, fontweight="bold")
    xlim = (metrics.first_arrival_time - 1, metrics.last_arrival_time + 1)

    # Plot 0: Arrival Time Rug Plot
    ax = axes[0]
    sns.rugplot(data["timestamps"], ax=ax)
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Arrival Time Distribution", fontweight="bold")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()

    # Plot 1: Batch Size Over Time (number of requests being processed)
    ax = axes[1]
    # Create a timeline showing batch size at each moment
    # Build a step function: start time -> num_requests, end time -> 0
    time_points = []
    batch_sizes = []

    # Add all batch start/end events
    for batch in metrics.batch_metrics:
        time_points.append(batch.processing_start_time)
        batch_sizes.append(batch.num_real_requests)
        time_points.append(batch.processing_end_time)
        batch_sizes.append(0)

    # Sort by time
    if time_points:
        sorted_data = sorted(zip(time_points, batch_sizes, strict=True))
        time_points, batch_sizes = zip(*sorted_data, strict=True)

        # Add initial point at 0 if needed
        if time_points[0] > metrics.first_arrival_time:
            time_points = [metrics.first_arrival_time, *list(time_points)]
            batch_sizes = [0, *list(batch_sizes)]

        ax.step(time_points, batch_sizes, where="post", linewidth=2, color="b", label="Batch size")
        ax.fill_between(time_points, batch_sizes, step="post", alpha=0.3, color="b")

    ax.set_xlim(xlim)
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_ylabel("Number of Requests", fontweight="bold")
    ax.set_title("Batch Size Over Time", fontweight="bold")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()

    # Plot 2: Idle Time Percentage per Second
    ax = axes[2]
    min_second = int(metrics.first_arrival_time)
    max_second = int(metrics.last_arrival_time)
    seconds_range = list(range(min_second, max_second + 1))
    idle_percentages = []

    for second in seconds_range:
        second_start = float(second)
        second_end = float(second + 1)

        # Calculate total time spent processing batches during this second
        batch_time = 0.0
        for batch in metrics.batch_metrics:
            # Find overlap between batch interval and this second
            overlap_start = max(second_start, batch.processing_start_time)
            overlap_end = min(second_end, batch.processing_end_time)
            if overlap_start < overlap_end:
                batch_time += overlap_end - overlap_start

        # Idle time = total time - batch time
        idle_time = 1.0 - batch_time
        idle_percentage = idle_time * 100.0
        idle_percentages.append(max(0.0, min(100.0, idle_percentage)))  # Clamp to [0, 100]

    ax.bar(seconds_range, idle_percentages, width=1.0, color="orange", alpha=0.7, edgecolor="darkorange")
    ax.set_xlim(xlim)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_ylabel("Idle Time (%)", fontweight="bold")
    ax.set_title("Worker Idle Time per Second", fontweight="bold")
    ax.grid(visible=True, alpha=0.3, axis="y")

    # Plot 3: Latency Statistics over Time (scatter + rolling window overlay)
    ax = axes[3]
    if data["latencies"]:
        latencies_ms = [latency * 1000 for latency in data["latencies"]]

        # Approximate timestamp for each latency (distribute across batch timestamps)
        # This is approximate since multiple requests complete per batch
        latency_timestamps_individual = []
        requests_per_batch = len(latencies_ms) // max(len(data["timestamps"]), 1)
        for i in range(len(latencies_ms)):
            batch_idx = min(i // max(requests_per_batch, 1), len(data["timestamps"]) - 1)
            latency_timestamps_individual.append(data["timestamps"][batch_idx] if data["timestamps"] else 0)

        # Scatter plot of individual latencies
        ax.scatter(
            latency_timestamps_individual,
            latencies_ms,
            alpha=0.3,
            s=10,
            color="gray",
            label="Individual requests",
            zorder=1,
        )

        # Compute rolling statistics (window of 50 requests)
        window = 50
        if len(latencies_ms) >= window:
            avg_latencies = []
            p50_latencies = []
            p99_latencies = []
            latency_timestamps = []

            for i in range(window, len(latencies_ms) + 1, window // 2):  # 50% overlap
                window_latencies = latencies_ms[max(0, i - window) : i]
                avg_latencies.append(np.mean(window_latencies))
                p50_latencies.append(np.percentile(window_latencies, 50))
                p99_latencies.append(np.percentile(window_latencies, 99))
                # Use timestamp of middle request in window
                mid_idx = max(0, i - window // 2)
                latency_timestamps.append(
                    latency_timestamps_individual[min(mid_idx, len(latency_timestamps_individual) - 1)]
                )

            # Overlay rolling window statistics
            if latency_timestamps:
                ax.plot(latency_timestamps, avg_latencies, "b-", linewidth=2, label="Average", alpha=0.9, zorder=2)
                ax.plot(latency_timestamps, p50_latencies, "g-", linewidth=2, label="P50 (Median)", alpha=0.9, zorder=2)
                ax.plot(latency_timestamps, p99_latencies, "r-", linewidth=2, label="P99", alpha=0.9, zorder=2)

        ax.set_xlabel("Time (seconds)", fontweight="bold")
        ax.set_ylabel("Latency (ms)", fontweight="bold")
        ax.set_title("End-to-End Latency Over Time (Scatter + Rolling Window)", fontweight="bold")
        ax.grid(visible=True, alpha=0.3)
        ax.set_xlim(xlim)
        ax.legend()

    plt.tight_layout()

    # Save plots
    pdf_path = output_path / "metrics.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Metrics plots saved to {pdf_path}")

    # Also save raw metrics as CSV for further analysis
    csv_path = output_path / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "batch_id",
                "batch_time_ms",
                "num_real",
                "batch_size",
                "utilization",
                "real_throughput",
                "total_throughput",
            ]
        )
        for i, batch in enumerate(metrics.batch_metrics):
            writer.writerow(
                [
                    data["timestamps"][i] if i < len(data["timestamps"]) else 0,
                    batch.batch_id,
                    batch.batch_processing_time * 1000,
                    batch.num_real_requests,
                    batch.total_batch_size,
                    batch.batch_utilization,
                    data["real_throughputs"][i] if i < len(data["real_throughputs"]) else 0,
                    data["total_throughputs"][i] if i < len(data["total_throughputs"]) else 0,
                ]
            )

    logger.info(f"Raw metrics saved to {csv_path}")

    # Scheduling-specific plots (only if robot_ids are available)
    has_scheduling_data = any(b.robot_ids for b in metrics.batch_metrics)
    if has_scheduling_data:
        _plot_scheduling_metrics(metrics, output_path)


def _plot_scheduling_metrics(metrics: MetricsCollector, output_path: Path) -> None:
    """Generate the 4 scheduling-specific plots."""
    sns.set_style("darkgrid")
    xlim = (metrics.first_arrival_time - 1, metrics.last_arrival_time + 1)

    # Collect per-robot event timelines: list of (response_time, start_step, execution_horizon)
    robot_events: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
    for batch in metrics.batch_metrics:
        for robot_id, ss, eh in zip(batch.robot_ids, batch.start_steps, batch.execution_horizons, strict=True):
            robot_events[robot_id].append((batch.processing_end_time, ss, eh))

    robots_sorted = sorted(robot_events.keys())
    robot_to_y = {r: i for i, r in enumerate(robots_sorted)}
    n_robots = len(robots_sorted)

    if n_robots == 0:
        return

    colors = plt.cm.tab20.colors  # type: ignore[attr-defined]

    # ---- Plot 1: GPU Processing Timeline (Gantt chart) ----
    fig, ax = plt.subplots(figsize=(16, max(4, n_robots * 0.4)))
    for batch in metrics.batch_metrics:
        t0, t1 = batch.processing_start_time, batch.processing_end_time
        for robot_id in batch.robot_ids:
            y = robot_to_y[robot_id]
            ax.barh(y, t1 - t0, left=t0, height=0.7, color=colors[y % len(colors)], edgecolor="black", linewidth=0.3)

    ax.set_yticks(range(n_robots))
    ax.set_yticklabels(robots_sorted, fontsize=8)
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_ylabel("Robot", fontweight="bold")
    ax.set_title("GPU Processing Timeline", fontsize=14, fontweight="bold")
    ax.set_xlim(xlim)
    ax.grid(visible=True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(output_path / "gpu_timeline.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"GPU timeline saved to {output_path / 'gpu_timeline.pdf'}")

    # ---- Plot 2: Actions Left Per Robot Over Time (gradient from fresh → depleted) ----
    from matplotlib.colors import LinearSegmentedColormap

    fresh_cmap = LinearSegmentedColormap.from_list("fresh", ["#d73027", "#fee08b", "#1a9850"])
    N_SLICES = 40  # thin slices per chunk to approximate a smooth gradient

    fig, ax = plt.subplots(figsize=(16, max(4, n_robots * 0.4)))
    bar_height = 0.7

    for robot_id in robots_sorted:
        events = robot_events[robot_id]
        y_base = robot_to_y[robot_id]

        for i, (t_resp, ss, eh) in enumerate(events):
            if i + 1 < len(events):
                t_next = events[i + 1][0]
                ss_next = events[i + 1][1]
                consumed = ss_next - ss
                t_empty = t_resp + (t_next - t_resp) * min(eh / max(consumed, 1), 1.0) if consumed > 0 else t_next
            else:
                if len(events) > 1:
                    avg_dt = (events[-1][0] - events[0][0]) / (len(events) - 1)
                    t_empty = t_resp + avg_dt
                else:
                    t_empty = t_resp + 1.0

            chunk_duration = t_empty - t_resp
            if chunk_duration <= 0:
                continue

            # Draw gradient slices: green (fresh, fraction=1) → red (depleted, fraction=0)
            slice_w = chunk_duration / N_SLICES
            for s in range(N_SLICES):
                frac = 1.0 - s / N_SLICES  # 1 = fresh, 0 = depleted
                ax.barh(
                    y_base, slice_w, left=t_resp + s * slice_w,
                    height=bar_height, color=fresh_cmap(frac), linewidth=0,
                )

            # Starvation gap (solid red)
            if i + 1 < len(events):
                t_next_resp = events[i + 1][0]
                if t_empty < t_next_resp:
                    ax.barh(y_base, t_next_resp - t_empty, left=t_empty, height=bar_height, color="#d73027", alpha=0.5, linewidth=0)

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap=fresh_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, aspect=30)
    cbar.set_label("Actions remaining (fraction)", fontweight="bold")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["depleted", "half", "fresh"])

    ax.set_yticks(range(n_robots))
    ax.set_yticklabels(robots_sorted, fontsize=8)
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_ylabel("Robot", fontweight="bold")
    ax.set_title("Actions Left Per Robot Over Time (green=fresh chunk, red=depleted/starved)", fontsize=14, fontweight="bold")
    ax.set_xlim(xlim)
    ax.grid(visible=True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(output_path / "actions_left_timeline.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Actions-left timeline saved to {output_path / 'actions_left_timeline.pdf'}")

    # ---- Plot 3: Total Robot Starvations (bar chart) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    starvations = [metrics.robot_states.get(r, RobotSchedulingState()).total_starvations for r in robots_sorted]
    bar_colors = ["red" if s > 0 else "steelblue" for s in starvations]
    ax.bar(robots_sorted, starvations, color=bar_colors, edgecolor="black", alpha=0.8)
    for i, (r, s) in enumerate(zip(robots_sorted, starvations)):
        if s > 0:
            ax.text(i, s + 0.2, str(s), ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Robot", fontweight="bold")
    ax.set_ylabel("Starvation Count", fontweight="bold")
    ax.set_title("Total Robot Starvations (ran out of actions before new chunk)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path / "starvations.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Starvations plot saved to {output_path / 'starvations.pdf'}")

    # ---- Plot 4: Wasted Actions From Chunk Overlap (bar chart) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    wasted = [metrics.robot_states.get(r, RobotSchedulingState()).total_wasted_actions for r in robots_sorted]
    bar_colors = ["orange" if w > 0 else "steelblue" for w in wasted]
    ax.bar(robots_sorted, wasted, color=bar_colors, edgecolor="black", alpha=0.8)
    for i, (r, w) in enumerate(zip(robots_sorted, wasted)):
        if w > 0:
            ax.text(i, w + 0.2, str(w), ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Robot", fontweight="bold")
    ax.set_ylabel("Wasted Actions", fontweight="bold")
    ax.set_title("Wasted Actions From Chunk Overlap (new chunk arrived before old finished)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path / "wasted_actions.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wasted actions plot saved to {output_path / 'wasted_actions.pdf'}")
