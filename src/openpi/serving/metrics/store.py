"""MetricsStore: in-memory metrics state for the websocket policy server."""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import threading
import time
from typing import Any, TypeVar

import numpy as np
from openpi_client.messages import InferResponse

from openpi.serving.schemas import SchedulerTimingSample

T = TypeVar("T")


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
    last_server_send_times: dict[int, float] = field(default_factory=dict)  # request_id → server_send_time
    total_starvations: int = 0
    network_delays_ms: list[float] = field(default_factory=list)  # (delay_ms,) unbounded


@dataclass
class StarvationIntervalEvent:
    """One observed interval for starvation-rate accounting."""

    robot_id: str
    observed_steps: int
    starved_steps: int
    event_time: float


@dataclass
class TaskEpisodeEvent:
    """One downstream task completion event from a client runtime."""

    robot_id: str
    task_suite_name: str
    task_id: int
    episode_idx: int
    success: bool
    duration_s: float
    steps_taken: int
    event_time: float
    task_language: str | None = None
    total_episodes: int | None = None
    max_episode_steps: int | None = None
    max_duration_s: float | None = None


@dataclass
class TaskEpisodeProgress:
    """Latest in-progress step count for a downstream task episode."""

    robot_id: str
    task_suite_name: str
    task_id: int
    episode_idx: int
    current_step: int
    max_episode_steps: int
    update_time: float
    task_language: str | None = None
    total_episodes: int | None = None


@dataclass
class MetricsStore:
    """Single-call-site metrics store. All updates go through record_batch / record_ack."""

    batches: list[BatchSummary] = field(default_factory=list)
    scheduler_timings: list[SchedulerTimingSample] = field(default_factory=list)
    robot_states: dict[str, RobotState] = field(default_factory=dict)
    starvation_intervals: list[StarvationIntervalEvent] = field(default_factory=list)
    task_events: list[TaskEpisodeEvent] = field(default_factory=list)
    task_progress: dict[tuple[str, str, int, int], TaskEpisodeProgress] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    _batch_counter: int = field(default=0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def record_batch(self, responses: list[InferResponse]) -> None:
        """Called once per batch by _router_task."""
        if not responses:
            return
        with self._lock:
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
                start_steps=[r.action_start_step for r in responses],
            )
            self.batches.append(batch)

            for response in responses:
                state = self.robot_states.setdefault(response.robot_id, RobotState())

                if state.last_execution_horizon > 0:
                    delta = response.action_start_step - state.last_start_step
                    if delta > 0:
                        starved_steps = max(0, delta - state.last_execution_horizon)
                        self.starvation_intervals.append(
                            StarvationIntervalEvent(
                                robot_id=response.robot_id,
                                observed_steps=delta,
                                starved_steps=starved_steps,
                                event_time=response.inference_end_time,
                            )
                        )
                        state.total_starvations += starved_steps

                state.last_start_step = response.action_start_step
                state.last_execution_horizon = response.execution_horizon

    def record_send(self, robot_id: str, request_id: int, server_send_time: float) -> None:
        """Called from send() just before websocket.send_bytes()."""
        with self._lock:
            state = self.robot_states.get(robot_id)
            if state is not None:
                state.last_server_send_times[request_id] = server_send_time

    def record_ack(self, robot_id: str, request_id: int, receive_time: float, execution_start_step: int) -> None:
        """Called when client sends ResponseAck."""
        with self._lock:
            state = self.robot_states.get(robot_id)
            if state is None:
                return
            server_send_time = state.last_server_send_times.pop(request_id, None)
            if server_send_time is not None:
                delay_ms = (receive_time - server_send_time) * 1000
                state.network_delays_ms.append(delay_ms)

    def record_scheduler_timings(self, samples: list[SchedulerTimingSample]) -> None:
        """Called from the server process when the scheduler publishes timing samples."""
        with self._lock:
            self.scheduler_timings.extend(samples)

    def record_task_progress(
        self,
        robot_id: str,
        task_suite_name: str,
        task_id: int,
        episode_idx: int,
        *,
        current_step: int,
        max_episode_steps: int,
        task_language: str | None = None,
        total_episodes: int | None = None,
        update_time: float | None = None,
    ) -> None:
        """Called when client streams an in-progress task step count."""
        with self._lock:
            key = (robot_id, task_suite_name, task_id, episode_idx)
            existing = self.task_progress.get(key)
            step = max(current_step, existing.current_step if existing is not None else 0)
            self.task_progress[key] = TaskEpisodeProgress(
                robot_id=robot_id,
                task_suite_name=task_suite_name,
                task_id=task_id,
                episode_idx=episode_idx,
                current_step=step,
                max_episode_steps=max_episode_steps,
                task_language=task_language or (existing.task_language if existing is not None else None),
                total_episodes=total_episodes or (existing.total_episodes if existing is not None else None),
                update_time=time.time() if update_time is None else update_time,
            )

    def record_task_result(
        self,
        robot_id: str,
        task_suite_name: str,
        task_id: int,
        episode_idx: int,
        *,
        success: bool,
        duration_s: float,
        steps_taken: int,
        task_language: str | None = None,
        total_episodes: int | None = None,
        max_episode_steps: int | None = None,
        max_duration_s: float | None = None,
        event_time: float | None = None,
    ) -> None:
        """Called when client sends a downstream task completion event."""
        with self._lock:
            key = (robot_id, task_suite_name, task_id, episode_idx)
            self.task_progress.pop(key, None)
            self.task_events.append(
                TaskEpisodeEvent(
                    robot_id=robot_id,
                    task_suite_name=task_suite_name,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    success=success,
                    duration_s=duration_s,
                    steps_taken=steps_taken,
                    task_language=task_language,
                    total_episodes=total_episodes,
                    max_episode_steps=max_episode_steps,
                    max_duration_s=max_duration_s,
                    event_time=time.time() if event_time is None else event_time,
                )
            )

    def record_task_update(
        self,
        robot_id: str,
        task_suite_name: str,
        task_id: int,
        episode_idx: int,
        *,
        current_step: int,
        max_episode_steps: int,
        phase: str,
        task_language: str | None = None,
        total_episodes: int | None = None,
        success: bool | None = None,
        duration_s: float | None = None,
        steps_taken: int | None = None,
        max_duration_s: float | None = None,
        event_time: float | None = None,
    ) -> None:
        """Record a downstream task update (in-progress or terminal result)."""
        if phase == "progress":
            self.record_task_progress(
                robot_id=robot_id,
                task_suite_name=task_suite_name,
                task_id=task_id,
                episode_idx=episode_idx,
                current_step=current_step,
                max_episode_steps=max_episode_steps,
                task_language=task_language,
                total_episodes=total_episodes,
                update_time=event_time,
            )
            return

        if phase == "result":
            if success is None or duration_s is None:
                return
            self.record_task_result(
                robot_id=robot_id,
                task_suite_name=task_suite_name,
                task_id=task_id,
                episode_idx=episode_idx,
                success=success,
                duration_s=duration_s,
                steps_taken=current_step if steps_taken is None else steps_taken,
                task_language=task_language,
                total_episodes=total_episodes,
                max_episode_steps=max_episode_steps,
                max_duration_s=max_duration_s,
                event_time=event_time,
            )

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
        return per_robot, active_robot_count, healthy_robot_count, global_starvation_rate_pct

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

            robot_rollup, active_robot_count, healthy_robot_count, global_starvation_rate_pct = (
                self._build_robot_sla_rollup(
                    set(self.robot_states.keys()),
                    starvation_intervals,
                    sla_pct,
                )
            )
            per_robot: dict[str, Any] = {}
            for robot_id in sorted(set(self.robot_states.keys()) | set(robot_rollup.keys())):
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
                per_robot[robot_id] = {
                    "total_starvations": state.total_starvations if state is not None else 0,
                    "avg_network_delay_ms": float(np.mean(state.network_delays_ms))
                    if state is not None and state.network_delays_ms
                    else 0.0,
                    **rollup,
                }

            durations_s = [e.duration_s for e in task_events]
            steps = [e.steps_taken for e in task_events]
            total_task_episodes = len(task_events)
            total_task_successes = sum(int(e.success) for e in task_events)

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
