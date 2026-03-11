from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import itertools
import threading
import time
from typing import Any

import numpy as np
from openpi_client.messages import EpisodeEnd
from openpi_client.messages import EpisodeStart
from openpi_client.messages import InferResponse
from openpi_client.messages import ResponseAck
from openpi_client.schemas import JSONDataclass

from openpi.serving.metrics.schemas import BatchSummary
from openpi.serving.metrics.schemas import Episode
from openpi.serving.metrics.schemas import RequestRecord
from openpi.serving.metrics.schemas import ResponseRecord
from openpi.serving.metrics.schemas import Robot
from openpi.serving.metrics.schemas import RobotID
from openpi.serving.metrics.schemas import window_filter
from openpi.serving.schemas import SchedulerTimingSample
from openpi.serving.schemas import SlotRequest

# TODO: make sure nans are nans and not 0s
# TODO: make sure s, ms, and ns are consistent
# TODO: figure out if locking is necessary, if we can get away without it
# temporary hack to not serialize the lock in metrics store
lock: threading.RLock = threading.RLock()


@dataclass
class Snapshot:
    """Windowed metrics snapshot. All stats, SLA, and starvation computed here."""

    server_start_time: float
    start_timestamp: float
    end_timestamp: float
    sla_pct: float
    # Per-robot windowed actions_left concatenated with nan separators between episodes.
    robot_actions_left: dict[RobotID, np.ndarray]
    per_robot: dict[RobotID, dict]
    requests: list[RequestRecord]
    responses: list[ResponseRecord]
    batches: list[BatchSummary]
    completed_episodes: list[tuple[RobotID, Episode]]
    sla_capacity_curve: list[dict]
    healthy_robots_over_time: list[dict]

    @property
    def uptime_s(self) -> float:
        return self.end_timestamp - self.server_start_time

    @property
    def duration_s(self) -> float:
        return self.end_timestamp - self.start_timestamp

    @property
    def total_batches(self) -> int:
        return len(self.batches)

    @property
    def total_requests(self) -> int:
        return len(self.requests)

    @property
    def gpu_times_ms(self) -> list[float]:
        return [b.gpu_time_ms for b in self.batches]

    @property
    def queue_delays_ms(self) -> list[float]:
        return [r.queue_delay_ms for r in self.responses]

    @property
    def requests_per_second(self) -> float:
        return self.total_requests / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def avg_gpu_time_ms(self) -> float:
        t = self.gpu_times_ms
        return float(np.mean(t)) if t else 0.0

    @property
    def gpu_busy_pct(self) -> float:
        if not self.batches or self.duration_s <= 0:
            return 0.0
        return sum(b.gpu_time_ms for b in self.batches) / (self.duration_s * 1000) * 100

    @property
    def avg_queue_delay_ms(self) -> float:
        d = self.queue_delays_ms
        return float(np.mean(d)) if d else 0.0

    @property
    def p50_latency_ms(self) -> float:
        lats = [r.total_latency_ms for r in self.responses]
        return float(np.percentile(lats, 50)) if lats else 0.0

    @property
    def p99_latency_ms(self) -> float:
        lats = [r.total_latency_ms for r in self.responses]
        return float(np.percentile(lats, 99)) if lats else 0.0

    @property
    def task_success_rate_pct(self) -> float:
        if not self.completed_episodes:
            return 0.0
        return sum(1 for _, ep in self.completed_episodes if ep.success) / len(self.completed_episodes) * 100

    @property
    def tp_suc_per_sec_all(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return sum(1 for _, ep in self.completed_episodes if ep.success) / self.duration_s


@dataclass
class MetricsStore(JSONDataclass):
    """Single-call-site metrics store. All updates go through record_batch / record_ack."""

    robots: dict[RobotID, Robot] = field(default_factory=dict)
    batches: list[BatchSummary] = field(default_factory=list)
    scheduler_timings: list[SchedulerTimingSample] = field(default_factory=list)

    def __post_init__(self):
        self.batches = [BatchSummary.from_json(b) for b in self.batches]
        self.robots = {
            robot_id: v
            if isinstance(v, Robot)
            else Robot(**v)
            if isinstance(v, dict)
            else Robot(robot_id=robot_id, episodes=[])
            for robot_id, v in self.robots.items()
        }
        self.scheduler_timings = [SchedulerTimingSample.from_json(s) for s in self.scheduler_timings]

    def record_batch(self, responses: list[InferResponse]) -> None:
        """Called once per batch by _router_task."""
        with lock:
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
        with lock:
            self.scheduler_timings.extend(samples)

    # request/response lifecycle

    def record_request(self, robot_id: str, request: SlotRequest) -> None:
        """Called when client sends InferRequest."""
        with lock:
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
        with lock:
            request_record = self.robots[robot_id].get_request(ack.request_id)
            batch = next(b for b in reversed(self.batches) if ack.request_id in b.request_ids)
            self.robots[robot_id].add_response(
                ResponseRecord(
                    request=request_record,
                    batch_id=batch.batch_id,
                    inference_start_time=batch.inference_start_time,
                    inference_end_time=batch.inference_end_time,
                    server_send_time=ack.server_send_time,
                    receive_time=ack.receive_time,
                    execution_start_step=ack.execution_start_step,
                    first_executed_index=ack.first_executed_index,
                )
            )

    def record_episode_start(
        self,
        robot_id: str,
        episode_start: EpisodeStart,
    ) -> None:
        """Called when client streams an in-progress task step count."""
        with lock:
            if robot_id not in self.robots:
                self.robots[robot_id] = Robot(robot_id=robot_id, episodes=[])
            self.robots[robot_id].start_episode(episode_start)

    def record_episode_end(
        self,
        robot_id: str,
        episode_end: EpisodeEnd,
    ) -> None:
        """Called when client streams an in-progress task step count."""
        with lock:
            self.robots[robot_id].end_episode(episode_end)

    @property
    def start_time(self) -> float:
        if self.batches:
            return self.batches[0].inference_start_time
        return time.time()

    def snapshot(self, window_s: float | None = None, *, sla_pct: float = 10.0) -> Snapshot:
        """Windowed metrics snapshot with all stats, starvation, and SLA data."""
        with lock:
            end_timestamp = time.time()
            start_timestamp = end_timestamp - window_s if window_s is not None else self.start_time

            batches = window_filter(self.batches, lambda b: b.inference_end_time, (start_timestamp, end_timestamp))
            requests = list(
                itertools.chain.from_iterable(
                    robot.get_requests(start_timestamp, end_timestamp) for robot in self.robots.values()
                )
            )
            responses = list(
                itertools.chain.from_iterable(
                    robot.get_responses(start_timestamp, end_timestamp) for robot in self.robots.values()
                )
            )

            # Windowed actions_left: each robot gets episode slices within the window concatenated with nans.
            robot_actions_left = {
                robot_id: robot.get_actions_left_concatenated(start_timestamp, end_timestamp)
                for robot_id, robot in self.robots.items()
            }

            # Completed episodes whose last request falls within the window.
            completed_episodes: list[tuple[RobotID, Episode]] = [
                (robot_id, episode)
                for robot_id, robot in self.robots.items()
                for episode in robot.episodes
                if episode.success is not None
                and episode.requests
                and start_timestamp <= episode.requests[-1].request_timestamp < end_timestamp
            ]

            # Per-robot stats from windowed data.
            responses_by_robot: dict[str, list[ResponseRecord]] = {}
            for resp in responses:
                responses_by_robot.setdefault(resp.request.robot_id, []).append(resp)

            eps_by_robot: dict[str, list[Episode]] = {}
            for robot_id, ep in completed_episodes:
                eps_by_robot.setdefault(robot_id, []).append(ep)

            duration_s = end_timestamp - start_timestamp
            active_rates: list[float] = []
            per_robot: dict[str, dict] = {}

            for robot_id, alh in robot_actions_left.items():
                valid = alh[~np.isnan(alh)]
                observed_steps = len(valid)
                starved_steps = int(np.sum(valid == 0))
                starvation_rate_pct = starved_steps / observed_steps * 100 if observed_steps > 0 else 0.0

                robot_eps = eps_by_robot.get(robot_id, [])
                tp = sum(1 for ep in robot_eps if ep.success) / duration_s if duration_s > 0 else 0.0

                robot_resps = responses_by_robot.get(robot_id, [])
                net_delays = [r.outbound_ms for r in robot_resps if r.receive_time > 0]
                avg_net_delay = float(np.mean(net_delays)) if net_delays else 0.0

                if observed_steps > 0:
                    active_rates.append(starvation_rate_pct)

                per_robot[robot_id] = {
                    "observed_steps": observed_steps,
                    "starved_steps": starved_steps,
                    "starvation_rate_pct": starvation_rate_pct,
                    "healthy": starvation_rate_pct <= sla_pct,
                    "tp_suc_per_sec_robot": tp,
                    "avg_network_delay_ms": avg_net_delay,
                }

            # SLA capacity curve: how many robots stay healthy at each starvation threshold.
            n_active = len(active_rates)
            sla_capacity_curve = [
                {
                    "sla_pct": float(threshold),
                    "healthy_robot_count": sum(1 for r in active_rates if r <= threshold),
                    "active_robot_count": n_active,
                    "healthy_robot_ratio_pct": sum(1 for r in active_rates if r <= threshold) / n_active * 100
                    if n_active > 0
                    else 0.0,
                }
                for threshold in range(21)
            ]

            healthy_robots_over_time = self._build_healthy_robots_over_time(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                sla_pct=sla_pct,
                t0=self.start_time,
            )

            return Snapshot(
                server_start_time=self.start_time,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                sla_pct=sla_pct,
                robot_actions_left=robot_actions_left,
                per_robot=per_robot,
                requests=requests,
                responses=responses,
                batches=batches,
                completed_episodes=completed_episodes,
                sla_capacity_curve=sla_capacity_curve,
                healthy_robots_over_time=healthy_robots_over_time,
            )

    def _build_healthy_robots_over_time(
        self,
        *,
        start_timestamp: float,
        end_timestamp: float,
        sla_pct: float,
        t0: float,
    ) -> list[dict[str, float | int]]:
        """Healthy robot count at each episode boundary within the window.

        Starvation is accumulated from episode windowed slices, so the cumulative
        rates reset to only reflect data within [start_timestamp, end_timestamp).
        """
        # Collect (end_time, robot_id, observed, starved) for each episode with steps in window.
        events: list[tuple[float, str, int, int]] = []
        for robot_id, robot in self.robots.items():
            for episode in robot.episodes:
                if not episode.requests:
                    continue
                end_time = episode.requests[-1].request_timestamp
                if end_time < start_timestamp or end_time >= end_timestamp:
                    continue
                alh = episode.get_windowed_actions_left(start_timestamp, end_timestamp)
                if len(alh) == 0:
                    continue
                events.append((end_time, robot_id, len(alh), int(np.sum(alh == 0))))

        events.sort(key=lambda x: x[0])

        robot_totals: dict[str, dict[str, int]] = {}
        points: list[dict[str, float | int]] = []
        for end_time, robot_id, observed, starved in events:
            row = robot_totals.setdefault(robot_id, {"observed": 0, "starved": 0})
            row["observed"] += observed
            row["starved"] += starved

            active_count = sum(1 for r in robot_totals.values() if r["observed"] > 0)
            healthy_count = sum(
                1 for r in robot_totals.values() if r["observed"] > 0 and r["starved"] / r["observed"] * 100 <= sla_pct
            )
            points.append(
                {"t": round(end_time - t0, 3), "healthy_robot_count": healthy_count, "active_robot_count": active_count}
            )

        return points

    def history(self, window_s: float | None = None) -> dict[str, Any]:
        """Per-batch time-series data for Plotly charts in the dashboard."""
        with lock:
            now = time.time()
            t0 = self.start_time

            if window_s is not None:
                start_ts = now - window_s
                batches = window_filter(self.batches, lambda b: b.inference_end_time, (start_ts, now))
                cutoff: float | None = start_ts
            else:
                batches = list(self.batches)
                cutoff = None

            # request_id -> ResponseRecord lookup for per-request batch data
            response_by_id: dict[int, ResponseRecord] = {
                resp.request.request_id: resp
                for robot in self.robots.values()
                for episode in robot.episodes
                for resp in episode.responses
            }

            batch_data = []
            for i, b in enumerate(batches):
                per_req = []
                for rid, req_id in zip(b.robot_ids, b.request_ids, strict=True):
                    resp = response_by_id.get(req_id)
                    if resp is not None:
                        inbound_ms = round(
                            (resp.request.server_arrival_time - resp.request.request_timestamp) * 1000, 2
                        )
                        queue_ms = round((b.inference_start_time - resp.request.server_arrival_time) * 1000, 2)
                    else:
                        inbound_ms = 0.0
                        queue_ms = 0.0
                    per_req.append(
                        {
                            "robot_id": rid,
                            "inbound_ms": inbound_ms,
                            "queue_ms": queue_ms,
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

            outbound: dict[str, list[float]] = {}
            for robot_id, robot in self.robots.items():
                delays = [
                    round(resp.outbound_ms, 2)
                    for episode in robot.episodes
                    for resp in episode.responses
                    if resp.receive_time > 0 and (cutoff is None or resp.receive_time >= cutoff)
                ]
                if delays:
                    outbound[robot_id] = delays

            scheduler_timings: dict[str, list[float]] = {}
            for sample in self.scheduler_timings:
                metric_key = f"{sample.scheduler_name}.{sample.metric_name}"
                scheduler_timings.setdefault(metric_key, []).append(round(sample.duration_ms, 3))

            # Completed episodes in window
            task_event_data = []
            for robot_id, robot in self.robots.items():
                for ep_idx, episode in enumerate(robot.episodes):
                    if episode.success is None or not episode.requests:
                        continue
                    event_time = episode.requests[-1].request_timestamp
                    if cutoff is not None and event_time < cutoff:
                        continue
                    task_event_data.append(
                        {
                            "t": round(event_time - t0, 3),
                            "robot_id": robot_id,
                            "task_key": f"{episode.task_suite_name}/{episode.task_id}",
                            "task_suite_name": episode.task_suite_name,
                            "task_id": episode.task_id,
                            "task_language": episode.task_language,
                            "episode_idx": ep_idx,
                            "success": episode.success,
                            "duration_s": round(episode.requests[-1].request_timestamp - episode.start_time, 3),
                            "steps_taken": episode.num_steps,
                            "total_episodes": len(robot.episodes),
                            "max_episode_steps": episode.max_episode_steps,
                            "max_duration_s": 0.0,
                        }
                    )

            # In-progress episodes (last episode per robot where success is None)
            task_progress_data = []
            for robot_id, robot in self.robots.items():
                if not robot.episodes:
                    continue
                episode = robot.episodes[-1]
                if episode.success is not None or not episode.requests:
                    continue
                ep_idx = len(robot.episodes) - 1
                update_time = episode.requests[-1].request_timestamp
                if cutoff is not None and update_time < cutoff:
                    continue
                task_progress_data.append(
                    {
                        "t": round(update_time - t0, 3),
                        "robot_id": robot_id,
                        "task_key": f"{episode.task_suite_name}/{episode.task_id}",
                        "task_suite_name": episode.task_suite_name,
                        "task_id": episode.task_id,
                        "task_language": episode.task_language,
                        "episode_idx": ep_idx,
                        "current_step": episode.num_steps,
                        "max_episode_steps": episode.max_episode_steps,
                        "total_episodes": len(robot.episodes),
                    }
                )

            return {
                "server_start_time": t0,
                "batches": batch_data,
                "outbound_delays_ms": outbound,
                "scheduler_timings_ms": scheduler_timings,
                "task_events": task_event_data,
                "task_progress": task_progress_data,
            }

    def reset(self) -> None:
        """Clear all accumulated metrics and reset counters."""
        with lock:
            self.batches.clear()
            self.scheduler_timings.clear()
            self.robots.clear()
