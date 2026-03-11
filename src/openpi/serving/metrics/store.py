from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
import time
from typing import Any, NamedTuple

import numpy as np
from openpi_client.messages import InferResponse
from openpi_client.schemas import JSONDataclass

from openpi.serving.schemas import SchedulerTimingSample


class Episode(NamedTuple):
    """A single contiguous episode for one robot."""

    start_time: float
    end_time: float  # exclusive; float("inf") for the current episode
    num_steps: int  # obs steps run 0 .. num_steps-1


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


class BatchSummary(NamedTuple):
    batch_id: int
    inference_start_time: float
    inference_end_time: float

    @property
    def gpu_time_ms(self) -> float:
        return (self.inference_end_time - self.inference_start_time) * 1000


@dataclass
class MetricsStore(JSONDataclass):
    batches: list[BatchSummary] = field(default_factory=list)
    requests: list[RequestRecord] = field(default_factory=list)
    scheduler_timings: list[SchedulerTimingSample] = field(default_factory=list)
    arrivals: list[tuple[str, int, float]] = field(default_factory=list)
    # (robot_id, observation_step, arrival_timestamp)
    start_time: float = field(default_factory=time.time)
    batch_counter: int = field(default=0)

    def __post_init__(self):
        # Coerce from JSON: dicts (old format) or lists (NamedTuple → JSON array)
        self.batches = [
            BatchSummary(**b) if isinstance(b, dict) else BatchSummary(*b) if not isinstance(b, BatchSummary) else b
            for b in self.batches
        ]
        self.requests = [RequestRecord(**r) if isinstance(r, dict) else r for r in self.requests]
        self.scheduler_timings = [
            SchedulerTimingSample(**s) if isinstance(s, dict) else s for s in self.scheduler_timings
        ]
        # JSON round-trips tuples as lists; coerce back
        self.arrivals = [(r, s, t) for r, s, t in self.arrivals]
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
        request_id: int,
        server_send_time: float | None,
        receive_time: float,
        execution_start_step: int,
        first_executed_index: int = 0,
    ) -> None:
        record = self._pending.pop(request_id)
        record.server_send_time = server_send_time or 0.0
        record.receive_time = receive_time
        record.execution_start_step = execution_start_step
        record.first_executed_index = first_executed_index

    def record_arrival(self, robot_id: str, observation_step: int, arrival_timestamp: float) -> None:
        self.arrivals.append((robot_id, observation_step, arrival_timestamp))

    def record_scheduler_timings(self, samples: list[SchedulerTimingSample]) -> None:
        self.scheduler_timings.extend(samples)

    def actions_left_series(self) -> dict[str, list[list[int]]]:
        """Returns per-robot, per-episode actions_left arrays for dashboard plotting.

        Result shape: {robot_id: [[actions_left per step], ...]}
        Uses full history (not windowed) so episode boundaries are correct.
        """
        result: dict[str, list[list[int]]] = {}
        for robot_id in {r for r, _, _ in self.arrivals}:
            episodes = self._episodes(robot_id)
            acked_all = [r for r in self.requests if r.robot_id == robot_id and r.receive_time > 0.0]
            robot_eps = []
            for ep in episodes:
                if ep.num_steps == 0:
                    continue
                ep_chunks = [r for r in acked_all if ep.start_time <= r.server_arrival_time < ep.end_time]
                robot_eps.append(self._compute_actions_left(ep, ep_chunks).tolist())
            result[robot_id] = robot_eps
        return result

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

    def _episodes(self, robot_id: str) -> list[Episode]:
        """Returns one Episode per contiguous run for this robot."""
        robot_arr = [
            (observation_step, arrival_timestamp)
            for arrival_robot_id, observation_step, arrival_timestamp in self.arrivals
            if arrival_robot_id == robot_id
        ]
        if not robot_arr:
            return []
        episodes: list[Episode] = []
        current_start = -1.0
        num_steps = 0
        last_step = -1
        for obs_step, timestamp in robot_arr:
            if last_step >= 0 and obs_step < last_step:  # episode reset
                episodes.append(Episode(current_start, timestamp, num_steps))
                num_steps = 0
            if num_steps == 0:
                current_start = timestamp
            num_steps = max(num_steps, obs_step + 1)
            last_step = obs_step
        if num_steps > 0:
            episodes.append(Episode(current_start, float("inf"), num_steps))
        return episodes

    def _compute_actions_left(self, ep: Episode, ep_chunks: list[RequestRecord]) -> np.ndarray:
        """Returns actions_left[0..num_steps-1] for one episode."""
        actions_left = np.zeros(ep.num_steps, dtype=np.int32)
        for r in ep_chunks:
            start = r.action_start_step + r.first_executed_index
            end = min(r.action_start_step + r.execution_horizon, ep.num_steps)
            if start < end:
                steps = np.arange(start, end)
                avail = r.action_start_step + r.execution_horizon - steps
                actions_left[start:end] = np.maximum(actions_left[start:end], avail)
        return actions_left

    def _per_robot_stats(self, requests: list[RequestRecord]) -> dict[str, Any]:
        """Returns per-robot starvation and network delay stats."""
        per_robot: dict[str, Any] = {}
        for robot_id in {r.robot_id for r in requests}:
            episodes = self._episodes(robot_id)
            acked_all = [r for r in self.requests if r.robot_id == robot_id and r.receive_time > 0.0]

            total_starvations = 0
            for ep in episodes:
                if ep.num_steps == 0:
                    continue
                ep_chunks = [r for r in acked_all if ep.start_time <= r.server_arrival_time < ep.end_time]
                actions_left = self._compute_actions_left(ep, ep_chunks)
                total_starvations += int(np.sum(actions_left == 0))

            acked_windowed = [r for r in requests if r.robot_id == robot_id and r.receive_time > 0.0]
            network_delays_ms = [r.outbound_ms for r in acked_windowed]
            per_robot[robot_id] = {
                "total_starvations": total_starvations,
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
                round(
                    (b.inference_start_time - batches[i - 1].inference_end_time) * 1000,
                    2,
                )
                if i > 0
                else 0.0
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

    def reset(self) -> None:
        self.batches.clear()
        self.requests.clear()
        self.scheduler_timings.clear()
        self.arrivals.clear()
        self.start_time = time.time()
        self.batch_counter = 0
        self._pending.clear()
