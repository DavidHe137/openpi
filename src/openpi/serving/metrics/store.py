from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import threading
import time
from typing import Any, NamedTuple, TypeAlias, TypeVar

import numpy as np
from openpi_client.messages import EpisodeEnd
from openpi_client.messages import EpisodeStart
from openpi_client.messages import InferResponse
from openpi_client.messages import ResponseAck
from openpi_client.schemas import JSONDataclass

from openpi.serving.schemas import SchedulerTimingSample
from openpi.serving.schemas import SlotRequest

RobotID: TypeAlias = str
T = TypeVar("T")


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
    task_suite_name: str
    task_id: int
    max_episode_steps: int
    task_language: str

    requests: list[RequestRecord]
    responses: list[ResponseRecord]
    success: bool | None = None

    def __post_init__(self) -> None:
        assert all(r.observation_step == i for i, r in enumerate(self.requests))
        assert all(
            next_request.action_start_step > prev_request.action_start_step
            for prev_request, next_request in zip(self.requests[:-1], self.requests[1:], strict=True)
        )

    @property
    def start_time(self) -> float:
        return self.requests[0].request_timestamp

    @property
    def num_steps(self) -> int:
        return len(self.requests)

    @property
    def actions_left_history(self) -> np.ndarray[int, " num_steps"]:
        actions_left_history = np.zeros(self.num_steps, dtype=np.int32)
        for response in self.responses:
            execution_end_step = min(
                response.request.action_start_step + response.request.execution_horizon,
                self.num_steps,
            )
            # TODO: fix this calculation
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

    def add_request(self, request: RequestRecord) -> None:
        assert request.observation_step == len(self.requests)
        self.requests.append(request)

    def add_response(self, response: ResponseRecord) -> None:
        self.responses.append(response)


@dataclass
class Robot:
    """Per-robot mutable state tracked during inference."""

    robot_id: str
    episodes: list[Episode]

    def __post_init__(self) -> None:
        # TODO: how to make these not saved in asdict but just in memory?
        self.requests: dict[int, RequestRecord] = {}
        self.responses: dict[int, ResponseRecord] = {}

    @property
    def current_episode(self) -> Episode:
        assert len(self.episodes) > 0
        return self.episodes[-1]

    def start_episode(self, episode_start: EpisodeStart) -> None:
        assert len(self.episodes) == episode_start.episode_idx
        self.episodes.append(
            Episode(
                task_suite_name=episode_start.task_suite_name,
                task_id=episode_start.task_id,
                max_episode_steps=episode_start.max_episode_steps,
                task_language=episode_start.task_language,
                requests=[],
                responses=[],
            )
        )

    def end_episode(self, episode_end: EpisodeEnd) -> None:
        episode = self.current_episode
        assert episode.task_suite_name == episode_end.task_suite_name
        assert episode.task_id == episode_end.task_id
        assert episode.num_steps == episode_end.steps_taken
        assert len(self.episodes) - 1 == episode_end.episode_idx
        episode.success = episode_end.success

    def add_request(self, request: RequestRecord) -> None:
        self.current_episode.requests.append(request)

    def add_response(self, response: ResponseRecord) -> None:
        self.current_episode.responses.append(response)

    def get_request(self, request_id: int) -> RequestRecord:
        # NOTE: can only be called when store is live
        # search backward on current request
        return next(r for r in reversed(self.current_episode.requests) if r.request_id == request_id)


class BatchSummary(NamedTuple):
    batch_id: int
    robot_ids: list[RobotID]
    request_ids: list[int]
    inference_start_time: float
    inference_end_time: float

    @property
    def gpu_time_ms(self) -> float:
        return (self.inference_end_time - self.inference_start_time) * 1000


@dataclass
class MetricsStore(JSONDataclass):
    """Single-call-site metrics store. All updates go through record_batch / record_ack."""

    robots: dict[RobotID, Robot] = field(default_factory=dict)
    batches: list[BatchSummary] = field(default_factory=list)
    scheduler_timings: list[SchedulerTimingSample] = field(default_factory=list)

    # TODO: figure out if locking is necessary, if we can get away without it
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self):
        # FIXME: have dataclasses implement their own coercion
        # Coerce from JSON: dicts (old format) or lists (NamedTuple → JSON array)
        self.batches = [
            BatchSummary(**b) if isinstance(b, dict) else BatchSummary(*b) if not isinstance(b, BatchSummary) else b
            for b in self.batches
        ]
        # TODO: doesn't work right now
        self.robots = {robot_id: Robot(robot_id=robot_id, episodes=[]) for robot_id in self.robots}
        self.scheduler_timings = [
            SchedulerTimingSample(**s) if isinstance(s, dict) else s for s in self.scheduler_timings
        ]

    # scheduler updates

    def record_batch(self, responses: list[InferResponse]) -> None:
        """Called once per batch by _router_task."""
        with self._lock:
            self.batches.append(
                BatchSummary(
                    batch_id=len(self.batches),
                    robot_ids=[r.robot_id for r in responses],
                    request_ids=[r.request_id for r in responses],
                    inference_start_time=responses[0].inference_start_time,
                    inference_end_time=responses[0].inference_end_time,
                )
            )

    def record_scheduler_timings(self, samples: list[SchedulerTimingSample]) -> None:
        """Called from the server process when the scheduler publishes timing samples."""
        with self._lock:
            self.scheduler_timings.extend(samples)

    # request/response lifecycle
    # TODO: look in server, think about how to map requests to responses elegantly
    def record_request(self, robot_id: str, request: SlotRequest) -> None:
        """Called when client sends InferRequest."""
        with self._lock:
            record = RequestRecord(
                robot_id=robot_id,
                request_id=request.request_id,
                observation_step=request.observation_step,
                action_start_step=request.action_start_step,
                execution_horizon=request.min_execution_horizon,
                request_timestamp=request.request_timestamp,
                server_arrival_time=request.arrival_timestamp,
            )
            self.robots[robot_id].add_request(record)

    def record_response(
        self,
        robot_id: str,
        ack: ResponseAck,
    ) -> None:
        """Called when client sends ResponseAck."""
        with self._lock:
            request_record = self.robots[robot_id].get_request(ack.request_id)
            batch = next(b for b in reversed(self.batches) if ack.request_id in b.request_ids)
            self.robots[robot_id].add_response(
                ResponseRecord(
                    request=request_record,
                    batch_id=batch.batch_id,
                    inference_start_time=batch.inference_start_time if batch else 0.0,
                    inference_end_time=batch.inference_end_time if batch else 0.0,
                    server_send_time=ack.server_send_time,
                    receive_time=ack.receive_time,
                    execution_start_step=ack.execution_start_step,
                    first_executed_index=ack.first_executed_index,
                )
            )

    # episode lifecycle

    def record_episode_start(
        self,
        robot_id: str,
        episode_start: EpisodeStart,
    ) -> None:
        """Called when client streams an in-progress task step count."""
        with self._lock:
            self.robots[robot_id].start_episode(episode_start)

    def record_episode_end(
        self,
        robot_id: str,
        episode_end: EpisodeEnd,
    ) -> None:
        """Called when client streams an in-progress task step count."""
        with self._lock:
            self.robots[robot_id].end_episode(episode_end)

    # metrics
    # FIXME: everything after here is still a work in progress

    @staticmethod
    def _window_filter(items: list[T], event_time_getter: Callable[[T], float], cutoff: float | None) -> list[T]:
        if cutoff is None:
            return items
        return [item for item in items if event_time_getter(item) >= cutoff]

    def _build_robot_sla_rollup(
        self,
        robot_ids: set[str],
        intervals: list[StarvationIntervalEvent],
        sla_pct: float,
    ) -> tuple[dict[str, dict[str, Any]], int, int, float]:
        per_robot: dict[str, dict[str, Any]] = {
            robot_id: {"observed_steps": 0, "starved_steps": 0} for robot_id in robot_ids
        }
        for interval in intervals:
            row = per_robot.setdefault(interval.robot_id, {"observed_steps": 0, "starved_steps": 0})
            row["observed_steps"] += interval.observed_steps
            row["starved_steps"] += interval.starved_steps

        active_robot_count = 0
        healthy_robot_count = 0
        total_observed_steps = 0
        total_starved_steps = 0
        for row in per_robot.values():
            observed_steps = row["observed_steps"]
            starved_steps = row["starved_steps"]
            total_observed_steps += observed_steps
            total_starved_steps += starved_steps
            starvation_rate_pct = (starved_steps / observed_steps * 100) if observed_steps > 0 else 0.0
            active = observed_steps > 0
            healthy = active and starvation_rate_pct <= sla_pct
            row["starvation_rate_pct"] = starvation_rate_pct
            row["active"] = active
            row["healthy"] = healthy
            if active:
                active_robot_count += 1
            if healthy:
                healthy_robot_count += 1

        global_starvation_rate_pct = (
            (total_starved_steps / total_observed_steps * 100) if total_observed_steps > 0 else 0.0
        )
        return (
            per_robot,
            active_robot_count,
            healthy_robot_count,
            global_starvation_rate_pct,
        )

    def _build_sla_capacity_curve(self, per_robot_rollup: dict[str, dict[str, Any]]) -> list[dict[str, float | int]]:
        active_rates = [
            float(row["starvation_rate_pct"]) for row in per_robot_rollup.values() if int(row["observed_steps"]) > 0
        ]
        active_robot_count = len(active_rates)
        curve: list[dict[str, float | int]] = []
        for sla_pct in range(21):
            healthy_robot_count = sum(1 for rate in active_rates if rate <= sla_pct)
            curve.append(
                {
                    "sla_pct": float(sla_pct),
                    "healthy_robot_count": healthy_robot_count,
                    "active_robot_count": active_robot_count,
                    "healthy_robot_ratio_pct": (healthy_robot_count / active_robot_count * 100)
                    if active_robot_count > 0
                    else 0.0,
                }
            )
        return curve

    def _build_healthy_robots_over_time(
        self,
        intervals: list[StarvationIntervalEvent],
        *,
        sla_pct: float,
        t0: float,
    ) -> list[dict[str, float | int]]:
        points: list[dict[str, float | int]] = []
        robot_totals: dict[str, dict[str, int]] = {}
        for interval in sorted(intervals, key=lambda item: item.event_time):
            row = robot_totals.setdefault(interval.robot_id, {"observed_steps": 0, "starved_steps": 0})
            row["observed_steps"] += interval.observed_steps
            row["starved_steps"] += interval.starved_steps

            active_robot_count = 0
            healthy_robot_count = 0
            for robot_row in robot_totals.values():
                observed_steps = robot_row["observed_steps"]
                if observed_steps <= 0:
                    continue
                active_robot_count += 1
                starvation_rate_pct = robot_row["starved_steps"] / observed_steps * 100
                if starvation_rate_pct <= sla_pct:
                    healthy_robot_count += 1

            points.append(
                {
                    "t": round(interval.event_time - t0, 3),
                    "healthy_robot_count": healthy_robot_count,
                    "active_robot_count": active_robot_count,
                }
            )
        return points

    def snapshot(self, window_s: float | None = None, *, sla_pct: float = 10.0) -> dict[str, Any]:
        """JSON-serializable summary of current metrics."""
        with self._lock:
            now = time.time()
            uptime_s = now - self.start_time
            cutoff = now - window_s if window_s is not None else None

            batches = self._window_filter(self.batches, lambda b: b.inference_end_time, cutoff)
            task_events = self._window_filter(self.task_events, lambda e: e.event_time, cutoff)
            starvation_intervals = self._window_filter(self.starvation_intervals, lambda i: i.event_time, cutoff)

            total_requests = sum(len(b.robot_ids) for b in batches)

            gpu_times = [b.gpu_time_ms for b in batches]
            avg_gpu_time_ms = float(np.mean(gpu_times)) if gpu_times else 0.0

            if len(batches) >= 2:
                wall_s = (
                    window_s
                    if window_s is not None
                    else (batches[-1].inference_end_time - batches[0].inference_start_time)
                )
                total_busy_ms = sum(gpu_times)
                gpu_busy_pct = min(100.0, total_busy_ms / (wall_s * 1000) * 100) if wall_s > 0 else 0.0
            else:
                gpu_busy_pct = 0.0

            latencies_ms = [(b.inference_end_time - req_ts) * 1000 for b in batches for req_ts in b.request_timestamps]
            p50_latency_ms = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
            p99_latency_ms = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

            queue_delays: list[float] = [d for b in batches for d in b.queue_delays_ms]
            avg_queue_delay_ms = float(np.mean(queue_delays)) if queue_delays else 0.0

            span_s = window_s if window_s is not None else uptime_s
            requests_per_second = total_requests / span_s if span_s > 0 else 0.0

            (
                robot_rollup,
                active_robot_count,
                healthy_robot_count,
                global_starvation_rate_pct,
            ) = self._build_robot_sla_rollup(
                set(self.robot_states.keys()),
                starvation_intervals,
                sla_pct,
            )
            task_successes_by_robot: dict[str, int] = {}
            for event in task_events:
                if event.success:
                    task_successes_by_robot[event.robot_id] = task_successes_by_robot.get(event.robot_id, 0) + 1

            total_task_episodes = len(task_events)
            total_task_successes = sum(task_successes_by_robot.values())
            tp_suc_per_sec_all = (total_task_successes / span_s) if span_s > 0 else 0.0

            per_robot: dict[str, Any] = {}
            for robot_id in sorted(
                set(self.robot_states.keys()) | set(robot_rollup.keys()) | set(task_successes_by_robot)
            ):
                state = self.robot_states.get(robot_id)
                rollup = robot_rollup.get(
                    robot_id,
                    {
                        "observed_steps": 0,
                        "starved_steps": 0,
                        "starvation_rate_pct": 0.0,
                        "active": False,
                        "healthy": False,
                    },
                )
                successes = task_successes_by_robot.get(robot_id, 0)
                per_robot[robot_id] = {
                    "total_starvations": state.total_starvations if state is not None else 0,
                    "avg_network_delay_ms": float(np.mean(state.network_delays_ms))
                    if state is not None and state.network_delays_ms
                    else 0.0,
                    "tp_suc_per_sec_robot": (successes / span_s) if span_s > 0 else 0.0,
                    **rollup,
                }

            durations_s = [e.duration_s for e in task_events]
            steps = [e.steps_taken for e in task_events]

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
                "sla_pct": float(sla_pct),
                "healthy_robot_count": healthy_robot_count,
                "active_robot_count": active_robot_count,
                "healthy_robot_ratio_pct": (healthy_robot_count / active_robot_count * 100)
                if active_robot_count > 0
                else 0.0,
                "global_starvation_rate_pct": global_starvation_rate_pct,
                "per_robot": per_robot,
                "total_task_episodes": total_task_episodes,
                "task_success_rate_pct": (total_task_successes / total_task_episodes * 100)
                if total_task_episodes
                else 0.0,
                "tp_suc_per_sec_all": tp_suc_per_sec_all,
                "avg_task_duration_s": float(np.mean(durations_s)) if durations_s else 0.0,
                "avg_task_steps": float(np.mean(steps)) if steps else 0.0,
            }

    def history(self, window_s: float | None = None, *, sla_pct: float = 10.0) -> dict[str, Any]:
        """Per-batch time-series data for Plotly charts in the dashboard."""
        with self._lock:
            now = time.time()
            cutoff = now - window_s if window_s is not None else None
            batches = self._window_filter(self.batches, lambda b: b.inference_end_time, cutoff)
            task_events = self._window_filter(self.task_events, lambda e: e.event_time, cutoff)
            task_progress = self._window_filter(list(self.task_progress.values()), lambda p: p.update_time, cutoff)
            starvation_intervals = self._window_filter(self.starvation_intervals, lambda i: i.event_time, cutoff)
            t0 = self.start_time

            batch_data = []
            for i, b in enumerate(batches):
                per_req = []
                for j, rid in enumerate(b.robot_ids):
                    per_req.append(
                        {
                            "robot_id": rid,
                            "inbound_ms": round(
                                (b.server_arrival_times[j] - b.request_timestamps[j]) * 1000,
                                2,
                            ),
                            "queue_ms": round(
                                (b.inference_start_time - b.server_arrival_times[j]) * 1000,
                                2,
                            ),
                            "infer_ms": round(b.gpu_time_ms, 2),
                        }
                    )
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
                robot_id: [round(d, 2) for d in state.network_delays_ms]
                for robot_id, state in self.robot_states.items()
                if state.network_delays_ms
            }
            scheduler_timings: dict[str, list[float]] = {}
            for sample in self.scheduler_timings:
                metric_key = f"{sample.scheduler_name}.{sample.metric_name}"
                scheduler_timings.setdefault(metric_key, []).append(round(sample.duration_ms, 3))

            robot_rollup, _, _, _ = self._build_robot_sla_rollup(
                set(self.robot_states.keys()), starvation_intervals, sla_pct
            )
            sla_capacity_curve = self._build_sla_capacity_curve(robot_rollup)
            healthy_robots_over_time = self._build_healthy_robots_over_time(
                starvation_intervals,
                sla_pct=sla_pct,
                t0=t0,
            )
            task_event_data = [
                {
                    "t": round(event.event_time - t0, 3),
                    "robot_id": event.robot_id,
                    "task_key": f"{event.task_suite_name}/{event.task_id}",
                    "task_suite_name": event.task_suite_name,
                    "task_id": event.task_id,
                    "task_language": event.task_language,
                    "episode_idx": event.episode_idx,
                    "success": event.success,
                    "duration_s": round(event.duration_s, 3),
                    "steps_taken": event.steps_taken,
                    "total_episodes": event.total_episodes,
                    "max_episode_steps": event.max_episode_steps,
                    "max_duration_s": event.max_duration_s,
                }
                for event in task_events
            ]
            task_progress_data = [
                {
                    "t": round(prog.update_time - t0, 3),
                    "robot_id": prog.robot_id,
                    "task_key": f"{prog.task_suite_name}/{prog.task_id}",
                    "task_suite_name": prog.task_suite_name,
                    "task_id": prog.task_id,
                    "task_language": prog.task_language,
                    "episode_idx": prog.episode_idx,
                    "current_step": prog.current_step,
                    "max_episode_steps": prog.max_episode_steps,
                    "total_episodes": prog.total_episodes,
                }
                for prog in task_progress
            ]
            per_robot_starvation = [{"robot_id": robot_id, **row} for robot_id, row in sorted(robot_rollup.items())]

            return {
                "server_start_time": t0,
                "sla_pct": float(sla_pct),
                "batches": batch_data,
                "outbound_delays_ms": outbound,
                "scheduler_timings_ms": scheduler_timings,
                "task_events": task_event_data,
                "task_progress": task_progress_data,
                "sla_capacity_curve": sla_capacity_curve,
                "healthy_robots_over_time": healthy_robots_over_time,
                "per_robot_starvation": per_robot_starvation,
            }

    def reset(self) -> None:
        """Clear all accumulated metrics and reset counters."""
        with self._lock:
            self.batches.clear()
            self.scheduler_timings.clear()
            self.robot_states.clear()
            self.starvation_intervals.clear()
            self.task_events.clear()
            self.task_progress.clear()
            self.start_time = time.time()
            self._batch_counter = 0
