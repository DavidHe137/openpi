from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import itertools
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

# TODO: make sure nans are nans and not 0s
# TODO: make sure s, ms, and ns are consistent
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

    def __post_init__(self) -> None:
        if isinstance(self.request, dict):
            self.request = RequestRecord(**self.request)

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


def window_filter(items: list[T], event_time_getter: Callable[[T], float], window_s: tuple[float, float]) -> list[T]:
    start_timestamp, end_timestamp = window_s
    return [item for item in items if start_timestamp <= event_time_getter(item) < end_timestamp]


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
        self.requests = [RequestRecord(**r) if isinstance(r, dict) else r for r in self.requests]
        self.responses = [ResponseRecord(**r) if isinstance(r, dict) else r for r in self.responses]
        assert all(r.observation_step == i for i, r in enumerate(self.requests))
        assert all(
            next_request.action_start_step >= prev_request.action_start_step
            for prev_request, next_request in zip(self.requests[:-1], self.requests[1:], strict=True)
        )

    @property
    def start_time(self) -> float:
        return self.requests[0].request_timestamp

    @property
    def end_time(self) -> float:
        # TODO: approximately correct
        return self.requests[-1].request_timestamp

    @property
    def num_steps(self) -> int:
        return len(self.requests)

    @property
    def actions_left_history(self) -> np.ndarray[int, " num_steps"]:
        actions_left_history = np.zeros(self.num_steps, dtype=np.int32)
        for response in self.responses:
            # At execution_start_step the robot is on action first_executed_index of the chunk,
            # so it has (execution_horizon - first_executed_index) actions remaining, counting
            # down by 1 each step until the chunk is exhausted or the episode ends.
            remaining = response.request.execution_horizon - response.first_executed_index
            execution_end_step = min(response.execution_start_step + remaining, self.num_steps)
            n = execution_end_step - response.execution_start_step
            actions_left = np.arange(remaining, remaining - n, -1)
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

    def get_requests(self, start_timestamp: float, end_timestamp: float) -> list[RequestRecord]:
        return window_filter(self.requests, lambda r: r.request_timestamp, (start_timestamp, end_timestamp))

    def get_responses(self, start_timestamp: float, end_timestamp: float) -> list[ResponseRecord]:
        return window_filter(self.responses, lambda r: r.receive_time, (start_timestamp, end_timestamp))


@dataclass
class Robot:
    """Per-robot mutable state tracked during inference."""

    robot_id: str
    episodes: list[Episode]

    def __post_init__(self) -> None:
        self.episodes = [Episode(**e) if isinstance(e, dict) else e for e in self.episodes]

    @property
    def current_episode(self) -> Episode:
        assert len(self.episodes) > 0
        return self.episodes[-1]

    def start_episode(self, episode_start: EpisodeStart) -> None:
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
        episode.success = episode_end.success

    def add_request(self, request: RequestRecord) -> None:
        self.current_episode.requests.append(request)

    def add_response(self, response: ResponseRecord) -> None:
        self.current_episode.responses.append(response)

    def get_request(self, request_id: int) -> RequestRecord:
        # NOTE: can only be called when store is live
        # search backward on current request
        return next(r for r in reversed(self.current_episode.requests) if r.request_id == request_id)

    @property
    def total_steps(self) -> int:
        return sum(e.num_steps for e in self.episodes)

    @property
    def total_starved_steps(self) -> int:
        return sum(np.sum(e.actions_left_history == 0) for e in self.episodes)

    def get_requests(self, start_timestamp: float, end_timestamp: float) -> list[RequestRecord]:
        return list(
            itertools.chain.from_iterable(e.get_requests(start_timestamp, end_timestamp) for e in self.episodes)
        )

    def get_responses(self, start_timestamp: float, end_timestamp: float) -> list[ResponseRecord]:
        return list(
            itertools.chain.from_iterable(e.get_responses(start_timestamp, end_timestamp) for e in self.episodes)
        )

    def get_actions_left_history(
        self, start_timestamp: float, end_timestamp: float
    ) -> dict[RobotID, np.ndarray[int, " steps_since_start_timestamp"]]:
        # FIXME: hardcoded hz for now
        total_steps = end_timestamp - start_timestamp / 20.0
        actions_left_history = np.full(total_steps, fill_value=np.nan, dtype=np.int32)
        # TODO: make this correct
        for episode in self.episodes:
            if episode.end_time < start_timestamp:
                continue
            actions_left_history_slice = actions_left_history[
                episode.start_time - start_timestamp : episode.end_time - start_timestamp
            ]

            actions_left_history[episode.start_time - start_timestamp : episode.end_time - start_timestamp] = (
                episode.actions_left_history
            )
        return actions_left_history


class StarvationIntervalEvent(NamedTuple):
    robot_id: str
    observed_steps: int
    starved_steps: int
    event_time: float  # last request timestamp of the episode


class BatchSummary(NamedTuple):
    batch_id: int
    robot_ids: list[RobotID]
    request_ids: list[int]
    inference_start_time: float
    inference_end_time: float

    @classmethod
    def from_json(cls, data: BatchSummary | dict | list) -> BatchSummary:
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(**data)
        return cls(*data)

    @property
    def gpu_time_ms(self) -> float:
        return (self.inference_end_time - self.inference_start_time) * 1000


# TODO: figure out if locking is necessary, if we can get away without it
# temporary hack to not serialize the lock in metrics store
lock: threading.RLock = threading.RLock()


@dataclass
class Snapshot:
    start_timestamp: float
    end_timestamp: float
    robot_actions_left: dict[RobotID, np.ndarray[int, " steps_since_start_timestamp"]]
    successes: dict[RobotID, int]  # TODO: should store additional metadata for success heatmap
    requests: list[RequestRecord]
    responses: list[ResponseRecord]
    batches: list[BatchSummary]

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
        return [batch.gpu_time_ms for batch in self.batches]

    @property
    def queue_delays_ms(self) -> list[float]:
        return [response.queue_delay_ms for response in self.responses]

    # TODO: task success rate stuff


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

    # FIXME: need to decide whether ot use first request, first inference, or even just startup
    @property
    def start_time(self) -> float:
        if self.batches:
            return self.batches[0].inference_start_time
        return time.time()

    def snapshot(self, window_s: float | None = None, *, sla_pct: float = 10.0) -> Snapshot:
        """JSON-serializable summary of current metrics."""
        with lock:
            end_timestamp = time.time()
            start_timestamp = time.time() - window_s if window_s is not None else self.start_time

            batches = window_filter(self.batches, lambda b: b.inference_end_time, (start_timestamp, end_timestamp))
            requests = itertools.chain.from_iterable(
                robot.get_requests(start_timestamp, end_timestamp) for robot in self.robots.values()
            )
            responses = itertools.chain.from_iterable(
                robot.get_responses(start_timestamp, end_timestamp) for robot in self.robots.values()
            )
            robot_actions_left = {
                robot_id: robot.get_actions_left_history(start_timestamp, end_timestamp)
                for robot_id, robot in self.robots.items()
            }

            successes = {
                robot_id: sum(1 for r in robot.responses if r.success) for robot_id, robot in self.robots.items()
            }

            return Snapshot(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                batches=batches,
                robot_actions_left=robot_actions_left,
                requests=requests,
                responses=responses,
                successes=successes,
            )

    def _build_robot_sla_rollup(
        self,
        sla_pct: float,
    ) -> tuple[dict[str, dict[str, Any]], int, int, float]:
        # TODO claude: make this concise/efficient by calling robot actions_left_history
        per_robot: dict[str, dict[str, Any]] = {
            robot_id: {
                "observed_steps": robot.total_steps,
                "starved_steps": robot.total_starved_steps,
                "starvation_rate_pct": (robot.total_starved_steps / robot.total_steps * 100)
                if robot.total_steps > 0
                else 0.0,
                "active": robot.total_steps > 0,
                "healthy": robot.total_starved_steps <= sla_pct * robot.total_steps,
            }
            for robot_id, robot in self.robots.items()
        }

        active_robot_count = sum(1 for robot in self.robots.values() if robot.total_steps > 0)
        healthy_robot_count = sum(
            1 for robot in self.robots.values() if robot.total_starved_steps <= sla_pct * robot.total_steps
        )
        total_starved_steps = sum(robot["starved_steps"] for robot in per_robot.values())
        total_observed_steps = sum(robot["observed_steps"] for robot in per_robot.values())
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

    def history(self, window_s: float | None = None, *, sla_pct: float = 10.0) -> dict[str, Any]:
        """Per-batch time-series data for Plotly charts in the dashboard."""
        with lock:
            now = time.time()
            cutoff = now - window_s if window_s is not None else None
            batches = window_filter(self.batches, lambda b: b.inference_end_time, cutoff)
            t0 = self.start_time

            # Build a lookup from request_id -> ResponseRecord for per-request batch data
            response_by_id: dict[int, ResponseRecord] = {}
            for robot in self.robots.values():
                for episode in robot.episodes:
                    for resp in episode.responses:
                        response_by_id[resp.request.request_id] = resp

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

            outbound: dict[str, list[float]] = {}
            for robot_id, robot in self.robots.items():
                delays = [
                    round(resp.outbound_ms, 2)
                    for episode in robot.episodes
                    for resp in episode.responses
                    if resp.receive_time > 0
                ]
                if delays:
                    outbound[robot_id] = delays

            scheduler_timings: dict[str, list[float]] = {}
            for sample in self.scheduler_timings:
                metric_key = f"{sample.scheduler_name}.{sample.metric_name}"
                scheduler_timings.setdefault(metric_key, []).append(round(sample.duration_ms, 3))

            robot_rollup, _, _, _ = self._build_robot_sla_rollup(sla_pct)
            sla_capacity_curve = self._build_sla_capacity_curve(robot_rollup)
            healthy_robots_over_time = self._build_healthy_robots_over_time(
                starvation_intervals,
                sla_pct=sla_pct,
                t0=t0,
            )

            # Completed episodes in window
            task_event_data = []
            for robot_id, robot in self.robots.items():
                for ep_idx, episode in enumerate(robot.episodes):
                    if episode.success is None:
                        continue
                    if not episode.requests:
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
                if episode.success is not None:
                    continue
                if not episode.requests:
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
        with lock:
            self.batches.clear()
            self.scheduler_timings.clear()
            self.robots.clear()
