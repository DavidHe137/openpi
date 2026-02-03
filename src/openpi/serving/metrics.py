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

    @property
    def batch_processing_time(self) -> float:
        """Total time to process this batch."""
        return self.processing_end_time - self.processing_start_time

    @property
    def batch_utilization(self) -> float:
        """Ratio of real requests to total batch size."""
        return self.num_real_requests / self.total_batch_size if self.total_batch_size > 0 else 0.0


@dataclass
class MetricsCollector:
    """Aggregates and manages metrics collection."""

    # Per-request tracking
    request_metrics: dict[int, RequestMetrics] = field(default_factory=dict)

    # Per-batch tracking
    batch_metrics: list[BatchMetrics] = field(default_factory=list)

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
        """Record batch-level metrics."""
        self.batch_metrics.append(
            BatchMetrics(
                batch_id=batch_metric.batch_id,
                processing_start_time=batch_metric.processing_start_time - self.start_time,
                processing_end_time=batch_metric.processing_end_time - self.start_time,
                num_real_requests=batch_metric.num_real_requests,
                total_batch_size=batch_metric.total_batch_size,
                request_ids=batch_metric.request_ids,
            )
        )

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

    # Create figure with 4 subplots (2x2)
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

    # Plot 1: Total Throughput (with padding) over Time
    ax = axes[1]
    # FIXME: can definitely clean up these metrics
    seconds_range = range(int(metrics.first_arrival_time), int(metrics.last_arrival_time) + 1)
    ax.plot(
        seconds_range,
        data["total_throughputs"],
        "r-",
        linewidth=2,
        label="Total (with padding)",
    )
    ax.set_xlim(xlim)
    ax.plot(seconds_range, data["real_throughputs"], "b--", linewidth=1.5, alpha=0.7, label="Real")
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_ylabel("Throughput (requests/sec)", fontweight="bold")
    ax.set_title("Total vs Real Throughput Over Time", fontweight="bold")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()

    # Plot 2: Batch Utilization over Time
    ax = axes[2]
    ax.plot(data["timestamps"], [u * 100 for u in data["batch_utilizations"]], "g-", linewidth=2)
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_ylabel("Batch Utilization (%)", fontweight="bold")
    ax.set_title("Batch Utilization Over Time", fontweight="bold")
    ax.set_xlim(xlim)
    ax.set_ylim(0, 105)
    ax.grid(visible=True, alpha=0.3)

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
    png_path = output_path / "metrics.png"
    pdf_path = output_path / "metrics.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Metrics plots saved to {png_path} and {pdf_path}")

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
