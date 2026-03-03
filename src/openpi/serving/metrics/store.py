"""MetricsStore: in-memory metrics state for the websocket policy server."""

from collections import deque
from dataclasses import dataclass
from dataclasses import field
import time
from typing import Any

import numpy as np
from openpi_client.messages import InferResponse


@dataclass
class BatchSummary:
    """Stored once per completed GPU batch."""

    batch_id: int
    robot_ids: list[str]
    request_ids: list[int]
    request_timestamps: list[float]
    server_arrival_times: list[float]
    inference_start_time: float  # same for all requests in the batch
    inference_end_time: float  # same for all requests in the batch
    execution_horizons: list[int]
    start_steps: list[int]

    @property
    def gpu_time_ms(self) -> float:
        return (self.inference_end_time - self.inference_start_time) * 1000

    @property
    def queue_delays_ms(self) -> list[float]:
        return [(self.inference_start_time - t) * 1000 for t in self.server_arrival_times]


@dataclass
class RobotState:
    """Per-robot tracking for scheduling metrics and network delay."""

    last_start_step: int = 0
    last_execution_horizon: int = 0
    last_server_send_times: dict = field(default_factory=dict)  # request_id → server_send_time
    total_starvations: int = 0
    recent_network_delays_ms: deque = field(default_factory=lambda: deque(maxlen=500))


@dataclass
class MetricsStore:
    """Single-call-site metrics store. All updates go through record_batch / record_ack."""

    recent_batches: deque = field(default_factory=lambda: deque(maxlen=500))
    robot_states: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    _batch_counter: int = field(default=0, init=False)

    def record_batch(self, responses: list[InferResponse]) -> None:
        """Called once per batch by _router_task."""
        if not responses:
            return

        self._batch_counter += 1
        batch = BatchSummary(
            batch_id=self._batch_counter,
            robot_ids=[r.robot_id for r in responses],
            request_ids=[r.request_id for r in responses],
            request_timestamps=[r.request_timestamp for r in responses],
            server_arrival_times=[r.server_arrival_time for r in responses],
            inference_start_time=responses[0].inference_start_time,
            inference_end_time=responses[0].inference_end_time,
            execution_horizons=[r.execution_horizon for r in responses],
            start_steps=[r.start_step for r in responses],
        )
        self.recent_batches.append(batch)

        for response in responses:
            state = self.robot_states.setdefault(response.robot_id, RobotState())

            if state.last_execution_horizon > 0:
                actions_consumed = response.start_step - state.last_start_step
                actions_remaining = state.last_execution_horizon - actions_consumed
                if actions_remaining < 0:
                    state.total_starvations += abs(actions_remaining)

            state.last_start_step = response.start_step
            state.last_execution_horizon = response.execution_horizon

    def record_send(self, robot_id: str, request_id: int, server_send_time: float) -> None:
        """Called from send() just before websocket.send_bytes()."""
        state = self.robot_states.get(robot_id)
        if state is not None:
            state.last_server_send_times[request_id] = server_send_time

    def record_ack(self, robot_id: str, request_id: int, receive_time: float) -> None:
        """Called when client sends ResponseAck."""
        state = self.robot_states.get(robot_id)
        if state is None:
            return
        server_send_time = state.last_server_send_times.pop(request_id, None)
        if server_send_time is not None:
            delay_ms = (receive_time - server_send_time) * 1000
            state.recent_network_delays_ms.append(delay_ms)

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable summary of current metrics."""
        now = time.time()
        batches = list(self.recent_batches)
        uptime_s = now - self.start_time
        total_requests = sum(len(b.robot_ids) for b in batches)

        gpu_times = [b.gpu_time_ms for b in batches]
        avg_gpu_time_ms = float(np.mean(gpu_times)) if gpu_times else 0.0

        if len(batches) >= 2:
            total_wall_ms = (batches[-1].inference_end_time - batches[0].inference_start_time) * 1000
            total_busy_ms = sum(gpu_times)
            gpu_busy_pct = min(100.0, total_busy_ms / total_wall_ms * 100) if total_wall_ms > 0 else 0.0
        else:
            gpu_busy_pct = 0.0

        latencies_ms = [(b.inference_end_time - req_ts) * 1000 for b in batches for req_ts in b.request_timestamps]
        p50_latency_ms = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
        p99_latency_ms = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

        queue_delays: list[float] = [d for b in batches for d in b.queue_delays_ms]
        avg_queue_delay_ms = float(np.mean(queue_delays)) if queue_delays else 0.0

        requests_per_second = total_requests / uptime_s if uptime_s > 0 else 0.0

        per_robot: dict[str, Any] = {}
        for robot_id, state in self.robot_states.items():
            delays = list(state.recent_network_delays_ms)
            per_robot[robot_id] = {
                "total_starvations": state.total_starvations,
                "avg_network_delay_ms": float(np.mean(delays)) if delays else 0.0,
            }

        return {
            "uptime_s": uptime_s,
            "total_batches": self._batch_counter,
            "total_requests": total_requests,
            "avg_gpu_time_ms": avg_gpu_time_ms,
            "gpu_busy_pct": round(gpu_busy_pct, 1),
            "p50_latency_ms": p50_latency_ms,
            "p99_latency_ms": p99_latency_ms,
            "avg_queue_delay_ms": avg_queue_delay_ms,
            "requests_per_second": requests_per_second,
            "per_robot": per_robot,
        }

    def history(self) -> dict[str, Any]:
        """Per-batch time-series data for Plotly charts in the dashboard."""
        batches = list(self.recent_batches)
        t0 = self.start_time

        batch_data = []
        for i, b in enumerate(batches):
            per_req = []
            for j, rid in enumerate(b.robot_ids):
                per_req.append(
                    {
                        "robot_id": rid,
                        "inbound_ms": round((b.server_arrival_times[j] - b.request_timestamps[j]) * 1000, 2),
                        "queue_ms": round((b.inference_start_time - b.server_arrival_times[j]) * 1000, 2),
                        "infer_ms": round(b.gpu_time_ms, 2),
                    }
                )
            idle_before_ms = (
                round((b.inference_start_time - batches[i - 1].inference_end_time) * 1000, 2) if i > 0 else 0.0
            )
            batch_data.append(
                {
                    "t": round(b.inference_end_time - t0, 3),
                    "batch_size": len(b.robot_ids),
                    "gpu_time_ms": round(b.gpu_time_ms, 2),
                    "idle_before_ms": idle_before_ms,
                    "inference_start_t": round(b.inference_start_time - t0, 3),
                    "inference_end_t": round(b.inference_end_time - t0, 3),
                    "robot_ids": b.robot_ids,
                    "per_request": per_req,
                }
            )

        outbound: dict[str, list[float]] = {
            robot_id: [round(d, 2) for d in state.recent_network_delays_ms]
            for robot_id, state in self.robot_states.items()
            if state.recent_network_delays_ms
        }

        return {
            "server_start_time": t0,
            "batches": batch_data,
            "outbound_delays_ms": outbound,
        }

    def reset(self) -> None:
        """Clear all accumulated metrics and reset counters."""
        self.recent_batches.clear()
        self.robot_states.clear()
        self.start_time = time.time()
        self._batch_counter = 0
