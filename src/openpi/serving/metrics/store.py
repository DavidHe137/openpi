from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
import time
from typing import Any, NamedTuple, TypeAlias

import numpy as np
from openpi_client.messages import InferResponse
from openpi_client.messages import ResponseAck
from openpi_client.schemas import JSONDataclass

from openpi.serving.schemas import SchedulerTimingSample
from openpi.serving.schemas import SlotRequest

RobotID: TypeAlias = str


@dataclass
class RequestRecord:
    """Full lifecycle record for one inference request."""

    robot_id: RobotID
    request_id: int
    observation_step: int
    action_start_step: int
    execution_horizon: int
    request_timestamp: float  # client: when request was created
    server_arrival_time: float  # server: when observation arrived


@dataclass
class ResponseRecord:
    request: RequestRecord
    batch_id: int

    inference_start_time: float  # gpu: before infer_batch
    inference_end_time: float  # gpu: after infer_batch
    server_send_time: float = 0.0  # server: before websocket.send_bytes()
    receive_time: float = 0.0  # client: ResponseAck.receive_time
    execution_start_step: int = 0  # client: ResponseAck.execution_start_step
    first_executed_index: int = 0  # client: index within chunk where execution started

    @property
    def queue_delay_ms(self) -> float:
        return (self.inference_start_time - self.request.server_arrival_time) * 1000

    @property
    def gpu_time_ms(self) -> float:
        return (self.inference_end_time - self.inference_start_time) * 1000

    @property
    def total_latency_ms(self) -> float:
        return (self.inference_end_time - self.request.request_timestamp) * 1000

    @property
    def outbound_ms(self) -> float:
        """Only valid when receive_time > 0."""
        return (self.receive_time - self.server_send_time) * 1000


@dataclass
class Episode:
    """A single contiguous episode for one robot."""

    robot_id: RobotID
    requests: list[RequestRecord]
    responses: list[ResponseRecord]

    def __post_init__(self) -> None:
        assert len(self.requests) > 0
        assert all(r.observation_step == i for i, r in enumerate(self.requests))
        assert all(
            next_request.action_start_step > prev_request.action_start_step
            for prev_request, next_request in zip(self.requests[:-1], self.requests[1:], strict=True)
        )
        assert all(r.robot_id == self.robot_id for r in self.requests)
        assert all(r.request in set(self.requests) for r in self.responses)

    @property
    def start_time(self) -> float:
        return self.requests[0].request_timestamp

    @property
    def num_steps(self) -> int:
        return len(self.requests)

    @property
    def actions_left_history(self) -> np.ndarray:
        actions_left_history = np.zeros(self.num_steps, dtype=np.int32)
        for response in self.responses:
            execution_end_step = min(
                response.request.action_start_step + response.request.execution_horizon,
                self.num_steps,
            )
            actions_left = [
                response.request.execution_horizon - response.first_executed_index
                for _ in range(
                    response.request.action_start_step,
                    response.request.action_start_step + response.request.execution_horizon,
                )
            ]

            actions_left_history[response.execution_start_step : execution_end_step] = np.maximum(
                actions_left_history[response.execution_start_step : execution_end_step],
                actions_left,
            )

        return actions_left_history


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
    responses: list[ResponseRecord] = field(default_factory=list)
    scheduler_timings: list[SchedulerTimingSample] = field(default_factory=list)
    arrivals: list[tuple[RobotID, int, float]] = field(default_factory=list)
    # (robot_id, observation_step, arrival_timestamp)
    start_time: float = field(default_factory=time.time)

    def __post_init__(self):
        # Coerce from JSON: dicts (old format) or lists (NamedTuple → JSON array)
        self.batches = [
            BatchSummary(**b) if isinstance(b, dict) else BatchSummary(*b) if not isinstance(b, BatchSummary) else b
            for b in self.batches
        ]
        self.requests = [RequestRecord(**r) if isinstance(r, dict) else r for r in self.requests]
        self.responses = [
            ResponseRecord(
                request=RequestRecord(**r["request"]) if isinstance(r.get("request"), dict) else r["request"],
                **{k: v for k, v in r.items() if k != "request"},
            )
            if isinstance(r, dict)
            else r
            for r in self.responses
        ]
        self.scheduler_timings = [
            SchedulerTimingSample(**s) if isinstance(s, dict) else s for s in self.scheduler_timings
        ]
        # JSON round-trips tuples as lists; coerce back
        self.arrivals = [(r, s, t) for r, s, t in self.arrivals]

        # Transient indices: not serialized
        self._requests_by_id: dict[int, RequestRecord] = {r.request_id: r for r in self.requests}
        self._pending: dict[int, tuple[int, InferResponse]] = {}  # request_id → (batch_id, InferResponse)
        self._batch_counter: int = max((b.batch_id for b in self.batches), default=0)

    def record_batch(self, responses: list[InferResponse]) -> None:
        """Called once per batch by _router_task."""
        if not responses:
            return

        self._batch_counter += 1
        batch_id = self._batch_counter

        self.batches.append(
            BatchSummary(
                batch_id=batch_id,
                inference_start_time=responses[0].inference_start_time,
                inference_end_time=responses[0].inference_end_time,
            )
        )

        for response in responses:
            self._pending[response.request_id] = (batch_id, response)

    def record_response(
        self,
        response: ResponseAck,
        server_send_time: float = 0.0,
    ) -> None:
        entry = self._pending.pop(response.request_id, None)
        if entry is None:
            return
        batch_id, infer_response = entry
        request = self._requests_by_id.get(infer_response.request_id)
        if request is None:
            return
        # Update execution_horizon with the actual value from inference
        request.execution_horizon = infer_response.execution_horizon
        self.responses.append(
            ResponseRecord(
                request=request,
                batch_id=batch_id,
                inference_start_time=infer_response.inference_start_time,
                inference_end_time=infer_response.inference_end_time,
                server_send_time=server_send_time,
                receive_time=response.receive_time,
                execution_start_step=response.execution_start_step,
                first_executed_index=response.first_executed_index,
            )
        )

    def record_request(self, slot_request: SlotRequest) -> None:
        record = RequestRecord(
            robot_id=slot_request.robot_id,
            request_id=slot_request.request_id,
            observation_step=slot_request.observation_step,
            action_start_step=slot_request.action_start_step,
            execution_horizon=slot_request.min_execution_horizon,
            request_timestamp=slot_request.request_timestamp,
            server_arrival_time=slot_request.arrival_timestamp,
        )
        self.requests.append(record)
        self._requests_by_id[record.request_id] = record

    def record_scheduler_timings(self, samples: list[SchedulerTimingSample]) -> None:
        self.scheduler_timings.extend(samples)

    # --- windowed views ---

    def _window(self, window_s: float | None) -> tuple[list[BatchSummary], list[ResponseRecord]]:
        """Return (batches, responses) filtered to the given time window."""
        if window_s is None:
            return self.batches, self.responses
        cutoff = time.time() - window_s
        batches = [b for b in self.batches if b.inference_end_time >= cutoff]
        responses = [r for r in self.responses if r.inference_end_time >= cutoff]
        return batches, responses

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

    def _latency_percentiles(self, responses: list[ResponseRecord]) -> tuple[float, float]:
        """Returns (p50_latency_ms, p99_latency_ms)."""
        latencies_ms = [r.total_latency_ms for r in responses]
        if not latencies_ms:
            return 0.0, 0.0
        return float(np.percentile(latencies_ms, 50)), float(np.percentile(latencies_ms, 99))

    def _per_robot_stats(self, responses: list[ResponseRecord]) -> dict[RobotID, Any]:
        """Returns per-robot latency stats."""
        by_robot: dict[RobotID, list[ResponseRecord]] = defaultdict(list)
        for r in responses:
            by_robot[r.request.robot_id].append(r)
        return {
            robot_id: {
                "count": len(rs),
                "p50_latency_ms": round(float(np.percentile([r.total_latency_ms for r in rs], 50)), 2),
                "p99_latency_ms": round(float(np.percentile([r.total_latency_ms for r in rs], 99)), 2),
                "avg_queue_delay_ms": round(float(np.mean([r.queue_delay_ms for r in rs])), 2),
            }
            for robot_id, rs in by_robot.items()
        }

    def snapshot(self, window_s: float | None = None) -> dict[str, Any]:
        """JSON-serializable summary of current metrics."""
        now = time.time()
        uptime_s = now - self.start_time
        batches, responses = self._window(window_s)

        avg_gpu_time_ms, gpu_busy_pct = self._gpu_stats(batches, window_s)
        p50_latency_ms, p99_latency_ms = self._latency_percentiles(responses)

        queue_delays = [r.queue_delay_ms for r in responses]
        avg_queue_delay_ms = float(np.mean(queue_delays)) if queue_delays else 0.0

        span_s = window_s if window_s is not None else uptime_s
        requests_per_second = len(responses) / span_s if span_s > 0 else 0.0

        return {
            "uptime_s": uptime_s,
            "total_batches": len(self.batches),
            "total_requests": len(self.requests),
            "avg_gpu_time_ms": avg_gpu_time_ms,
            "gpu_busy_pct": round(gpu_busy_pct, 1),
            "p50_latency_ms": p50_latency_ms,
            "p99_latency_ms": p99_latency_ms,
            "avg_queue_delay_ms": avg_queue_delay_ms,
            "requests_per_second": requests_per_second,
            "per_robot": self._per_robot_stats(responses),
        }

    # --- history helpers ---

    def _batch_series(
        self, batches: list[BatchSummary], responses: list[ResponseRecord], t0: float
    ) -> list[dict[str, Any]]:
        """Returns per-batch time-series entries."""
        responses_by_batch: dict[int, list[ResponseRecord]] = defaultdict(list)
        for r in responses:
            responses_by_batch[r.batch_id].append(r)

        batch_data = []
        for i, b in enumerate(batches):
            batch_responses = responses_by_batch.get(b.batch_id, [])
            per_req = [
                {
                    "robot_id": r.request.robot_id,
                    "inbound_ms": round(
                        (r.request.server_arrival_time - r.request.request_timestamp) * 1000,
                        2,
                    ),
                    "queue_ms": round(r.queue_delay_ms, 2),
                    "infer_ms": round(b.gpu_time_ms, 2),
                }
                for r in batch_responses
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
                    "batch_size": len(batch_responses),
                    "gpu_time_ms": round(b.gpu_time_ms, 2),
                    "idle_before_ms": idle_before_ms,
                    "inference_start_t": round(b.inference_start_time - t0, 3),
                    "inference_end_t": round(b.inference_end_time - t0, 3),
                    "robot_ids": [r.request.robot_id for r in batch_responses],
                    "per_request": per_req,
                }
            )
        return batch_data

    def _outbound_delays(self, responses: list[ResponseRecord]) -> dict[RobotID, list[float]]:
        """Returns per-robot lists of outbound round-trip delays in ms."""
        outbound: dict[RobotID, list[float]] = {}
        for r in responses:
            if r.receive_time > 0.0:
                outbound.setdefault(r.request.robot_id, []).append(round(r.outbound_ms, 2))
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
        batches, responses = self._window(window_s)
        return {
            "server_start_time": self.start_time,
            "batches": self._batch_series(batches, responses, self.start_time),
            "outbound_delays_ms": self._outbound_delays(responses),
            "scheduler_timings_ms": self._scheduler_timing_series(),
        }

    def reset(self) -> None:
        self.batches.clear()
        self.requests.clear()
        self.responses.clear()
        self.scheduler_timings.clear()
        self.arrivals.clear()
        self.start_time = time.time()
        self._batch_counter = 0
        self._pending.clear()
        self._requests_by_id.clear()
