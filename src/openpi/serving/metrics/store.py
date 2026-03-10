from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
import itertools
import time
from typing import Any

import numpy as np
from openpi_client.messages import InferResponse
from openpi_client.schemas import JSONDataclass

from openpi.serving.schemas import SchedulerTimingSample


@dataclass
class RequestRecord:
    """Full lifecycle record for one inference request."""

    robot_id: str
    batch_id: int
    request_id: int
    observation_step: int
    action_start_step: int
    execution_horizon: int
    # Lifecycle timestamps
    request_timestamp: float  # client: when request was created
    server_arrival_time: float  # server: when observation arrived
    inference_start_time: float  # gpu: before infer_batch
    inference_end_time: float  # gpu: after infer_batch
    server_send_time: float = 0.0  # server: before websocket.send_bytes()
    receive_time: float = 0.0  # client: ResponseAck.receive_time
    execution_start_step: int = 0  # client: ResponseAck.execution_start_step
    first_executed_index: int = 0  # client: index within chunk where execution started

    @property
    def queue_delay_ms(self) -> float:
        return (self.inference_start_time - self.server_arrival_time) * 1000

    @property
    def gpu_time_ms(self) -> float:
        return (self.inference_end_time - self.inference_start_time) * 1000

    @property
    def total_latency_ms(self) -> float:
        return (self.inference_end_time - self.request_timestamp) * 1000

    @property
    def outbound_ms(self) -> float:
        """Only valid when receive_time > 0."""
        return (self.receive_time - self.server_send_time) * 1000


@dataclass
class BatchSummary:
    """Stored once per completed GPU batch."""

    batch_id: int
    inference_start_time: float
    inference_end_time: float

    @property
    def gpu_time_ms(self) -> float:
        return (self.inference_end_time - self.inference_start_time) * 1000


@dataclass
class MetricsStore(JSONDataclass):
    """Single-call-site metrics store. All updates go through record_batch / record_ack."""

    batches: list[BatchSummary] = field(default_factory=list)
    requests: list[RequestRecord] = field(default_factory=list)
    scheduler_timings: list[SchedulerTimingSample] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    batch_counter: int = field(default=0, init=False)

    def __post_init__(self):
        # Transient index: in-flight requests waiting for ack; not serialized
        self._pending: dict[int, RequestRecord] = {r.request_id: r for r in self.requests if r.receive_time == 0.0}

    def record_batch(self, responses: list[InferResponse]) -> None:
        """Called once per batch by _router_task."""
        if not responses:
            return

        self.batch_counter += 1
        batch = BatchSummary(
            batch_id=self.batch_counter,
            inference_start_time=responses[0].inference_start_time,
            inference_end_time=responses[0].inference_end_time,
        )
        self.batches.append(batch)

        for response in responses:
            record = RequestRecord(
                robot_id=response.robot_id,
                batch_id=self.batch_counter,
                request_id=response.request_id,
                observation_step=response.observation_step,
                action_start_step=response.action_start_step,
                execution_horizon=response.execution_horizon,
                request_timestamp=response.request_timestamp,
                server_arrival_time=response.server_arrival_time,
                inference_start_time=response.inference_start_time,
                inference_end_time=response.inference_end_time,
            )
            self.requests.append(record)
            self._pending[record.request_id] = record

    def record_ack(
        self,
        robot_id: str,
        request_id: int,
        server_send_time: float | None,
        receive_time: float,
        execution_start_step: int,
        first_executed_index: int = 0,
    ) -> None:
        """Called when client sends ResponseAck."""
        if record := self._pending.pop(request_id, None):
            record.server_send_time = server_send_time or 0.0
            record.receive_time = receive_time
            record.execution_start_step = execution_start_step
            record.first_executed_index = first_executed_index

    def record_scheduler_timings(self, samples: list[SchedulerTimingSample]) -> None:
        """Called from the server process when the scheduler publishes timing samples."""
        self.scheduler_timings.extend(samples)

    def reset(self) -> None:
        """Reset all metrics."""
        self.batches.clear()
        self.requests.clear()
        self.scheduler_timings.clear()
        self.start_time = time.time()
        self.batch_counter = 0
        self._pending.clear()

    # --- windowed views ---

    def _window(self, window_s: float | None) -> tuple[list[BatchSummary], list[RequestRecord]]:
        """Return (batches, requests) filtered to the given time window."""
        if window_s is None:
            return self.batches, self.requests
        cutoff = time.time() - window_s
        batches = [b for b in self.batches if b.inference_end_time >= cutoff]
        batch_ids = {b.batch_id for b in batches}
        requests = [r for r in self.requests if r.batch_id in batch_ids]
        return batches, requests

    # --- snapshot helpers ---

    def _gpu_stats(self, batches: list[BatchSummary], window_s: float | None) -> tuple[float, float]:
        """Returns (avg_gpu_time_ms, gpu_busy_pct)."""
        gpu_times = [b.gpu_time_ms for b in batches]
        avg_gpu_time_ms = float(np.mean(gpu_times)) if gpu_times else 0.0
        if len(batches) >= 2:
            wall_s = window_s or (batches[-1].inference_end_time - batches[0].inference_start_time)
            gpu_busy_pct = min(100.0, sum(gpu_times) / (wall_s * 1000) * 100) if wall_s > 0 else 0.0
        else:
            gpu_busy_pct = 0.0
        return avg_gpu_time_ms, gpu_busy_pct

    def _latency_percentiles(self, requests: list[RequestRecord]) -> tuple[float, float]:
        """Returns (p50_latency_ms, p99_latency_ms)."""
        latencies_ms = [r.total_latency_ms for r in requests]
        if not latencies_ms:
            return 0.0, 0.0
        return float(np.percentile(latencies_ms, 50)), float(np.percentile(latencies_ms, 99))

    def _per_robot_stats(self, requests: list[RequestRecord]) -> dict[str, Any]:
        """Returns per-robot starvation and network delay stats."""
        per_robot: dict[str, Any] = {}
        for robot_id in {r.robot_id for r in requests}:
            acked = sorted(
                [r for r in requests if r.robot_id == robot_id and r.receive_time > 0.0],
                key=lambda r: r.execution_start_step,
            )
            starvations = sum(
                max(
                    0,
                    (curr.action_start_step + curr.first_executed_index)
                    - (prev.action_start_step + prev.execution_horizon),
                )
                for prev, curr in itertools.pairwise(acked)
            )
            network_delays_ms = [r.outbound_ms for r in acked]
            per_robot[robot_id] = {
                "total_starvations": starvations,
                "avg_network_delay_ms": float(np.mean(network_delays_ms)) if network_delays_ms else 0.0,
            }
        return per_robot

    def snapshot(self, window_s: float | None = None) -> dict[str, Any]:
        """JSON-serializable summary of current metrics."""
        now = time.time()
        uptime_s = now - self.start_time
        batches, requests = self._window(window_s)

        avg_gpu_time_ms, gpu_busy_pct = self._gpu_stats(batches, window_s)
        p50_latency_ms, p99_latency_ms = self._latency_percentiles(requests)

        queue_delays = [r.queue_delay_ms for r in requests]
        avg_queue_delay_ms = float(np.mean(queue_delays)) if queue_delays else 0.0

        span_s = window_s if window_s is not None else uptime_s
        requests_per_second = len(requests) / span_s if span_s > 0 else 0.0

        return {
            "uptime_s": uptime_s,
            "total_batches": self.batch_counter,
            "total_requests": len(requests),
            "avg_gpu_time_ms": avg_gpu_time_ms,
            "gpu_busy_pct": round(gpu_busy_pct, 1),
            "p50_latency_ms": p50_latency_ms,
            "p99_latency_ms": p99_latency_ms,
            "avg_queue_delay_ms": avg_queue_delay_ms,
            "requests_per_second": requests_per_second,
            "per_robot": self._per_robot_stats(requests),
        }

    # --- history helpers ---

    def _batch_series(
        self, batches: list[BatchSummary], requests: list[RequestRecord], t0: float
    ) -> list[dict[str, Any]]:
        """Returns per-batch time-series entries."""
        requests_by_batch: dict[int, list[RequestRecord]] = defaultdict(list)
        for r in requests:
            requests_by_batch[r.batch_id].append(r)

        batch_data = []
        for i, b in enumerate(batches):
            batch_reqs = requests_by_batch.get(b.batch_id, [])
            per_req = [
                {
                    "robot_id": r.robot_id,
                    "inbound_ms": round((r.server_arrival_time - r.request_timestamp) * 1000, 2),
                    "queue_ms": round(r.queue_delay_ms, 2),
                    "infer_ms": round(b.gpu_time_ms, 2),
                }
                for r in batch_reqs
            ]
            idle_before_ms = (
                round((b.inference_start_time - batches[i - 1].inference_end_time) * 1000, 2) if i > 0 else 0.0
            )
            batch_data.append(
                {
                    "t": round(b.inference_end_time - t0, 3),
                    "batch_size": len(batch_reqs),
                    "gpu_time_ms": round(b.gpu_time_ms, 2),
                    "idle_before_ms": idle_before_ms,
                    "inference_start_t": round(b.inference_start_time - t0, 3),
                    "inference_end_t": round(b.inference_end_time - t0, 3),
                    "robot_ids": [r.robot_id for r in batch_reqs],
                    "per_request": per_req,
                }
            )
        return batch_data

    def _outbound_delays(self, requests: list[RequestRecord]) -> dict[str, list[float]]:
        """Returns per-robot lists of outbound round-trip delays in ms."""
        outbound: dict[str, list[float]] = {}
        for r in requests:
            if r.receive_time > 0.0:
                outbound.setdefault(r.robot_id, []).append(round(r.outbound_ms, 2))
        return outbound

    def _scheduler_timing_series(self) -> dict[str, list[float]]:
        """Returns per-metric lists of scheduler timing samples in ms."""
        result: dict[str, list[float]] = {}
        for sample in self.scheduler_timings:
            key = f"{sample.scheduler_name}.{sample.metric_name}"
            result.setdefault(key, []).append(round(sample.duration_ms, 3))
        return result

    def history(self, window_s: float | None = None) -> dict[str, Any]:
        """Per-batch time-series data for Plotly charts in the dashboard."""
        batches, requests = self._window(window_s)
        return {
            "server_start_time": self.start_time,
            "batches": self._batch_series(batches, requests, self.start_time),
            "outbound_delays_ms": self._outbound_delays(requests),
            "scheduler_timings_ms": self._scheduler_timing_series(),
        }
