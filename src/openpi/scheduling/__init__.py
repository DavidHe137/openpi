from abc import ABC
from abc import abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
import dataclasses
import itertools
import logging
import multiprocessing as mp
import time

from openpi.scheduling.latency import EMALatencyTracker
from openpi.serving.schemas import AckNotification
from openpi.serving.schemas import CompletionNotification
from openpi.serving.schemas import RequestBatch
from openpi.serving.schemas import SchedulerDecision
from openpi.serving.schemas import SlotRequest
from openpi.shared.clock import Clock
from openpi.shared.clock import default_clock

logger = logging.getLogger(__name__)


class RequestScheduler(ABC):
    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        clock: Clock | None = None,
    ):
        self._batch_queue = batch_queue  # TODO: try to move this out of this class
        self._max_batch_size = max_batch_size
        self._clock = clock if clock is not None else default_clock()

        self._latest_requests: dict[str, SlotRequest] = {}
        self._latest_scheduled_requests: dict[str, SlotRequest] = {}
        self._deadline_steps: dict[str, int] = {}  # observation step that the robot will be starved
        self._decisions: list[SchedulerDecision] = []
        self.latency_tracker = EMALatencyTracker()  # TODO: allow different latency trackers
        self.next_batch_id = itertools.count(1)
        self._in_flight = 0

    def update(self, request: SlotRequest) -> None:
        self._latest_requests[request.robot_id] = request
        if request.deadline_step is not None and request.deadline_step > self._deadline_steps.get(request.robot_id, 0):
            logger.warning(
                "Updated deadline step for robot %s from %d to %d",
                request.robot_id,
                self._deadline_steps.get(request.robot_id, 0),
                request.deadline_step,
            )
            self._deadline_steps[request.robot_id] = request.deadline_step
        self.latency_tracker.update_obs(request.robot_id, request.arrival_timestamp, request.request_timestamp)

    def update_completion(self, notification: CompletionNotification) -> None:
        self.latency_tracker.update_infer(notification.batch_size, notification.inference_duration)

    def update_ack(self, notification: AckNotification) -> None:
        self.latency_tracker.update_action_delivery(
            notification.robot_id,
            notification.receive_time,
            notification.server_send_time,
        )

    def collect_trace(self, batch: list[SlotRequest]) -> dict:
        all_requests = list(self._latest_requests.values())
        candidates = list(self.schedulable_requests)
        now = self._clock.time()

        return {
            "requests": sorted(
                (
                    {
                        "robot_id": r.robot_id,
                        "observation_step": r.observation_step,
                        "action_start_step": r.action_start_step,
                        "deadline": self.deadline(robot_id=r.robot_id) - now,
                    }
                    for r in all_requests
                ),
                key=lambda x: x["deadline"],
            ),
            "candidates": sorted(
                (
                    {
                        "robot_id": r.robot_id,
                        "deadline": self.deadline(robot_id=r.robot_id) - now,
                    }
                    for r in candidates
                ),
                key=lambda x: x["deadline"],
            ),
            "batch": sorted(
                (
                    {
                        "robot_id": r.robot_id,
                        "deadline": self.deadline(robot_id=r.robot_id) - now,
                    }
                    for r in batch
                ),
                key=lambda x: x["deadline"],
            ),
        }

    def schedule(self) -> None:
        """Return a list of batches of requests to be sent to the GPU."""
        with self.record_timing() as duration:
            batches = self.get_next_batches()

        # TODO: choose batches, collect trace, update deadliens
        now = self._clock.time()
        for batch in batches:
            batch_size = len(batch)
            # Capture deadlines before the loop overwrites them, sort earliest first.
            trace = self.collect_trace(batch)
            annotated = []
            for request in batch:
                # FIXME: this might monotonically increase if we end up serving a newer observation?
                self._deadline_steps[request.robot_id] = (
                    request.request_timestamp + request.execution_horizon / request.control_hz
                )
                self._latest_scheduled_requests[request.robot_id] = request
                total_latency_steps = (
                    self.latency_tracker.total_latency(request.robot_id, batch_size) / request.control_hz
                )
                # FIXME: only pass inference + action latency, can determine observation latency when processing
                annotated.append(dataclasses.replace(request, estimated_d_param=total_latency_steps))

            batch_id = next(self.next_batch_id)

            # FIXME: this branch only has single batch decisions for now, will need to refactor timing for multi batch decisions
            self._decisions.append(
                SchedulerDecision(batch_id=batch_id, recorded_at=now, duration=duration(), trace=trace)
            )
            self._batch_queue.put_nowait(RequestBatch(requests=annotated, batch_id=batch_id))
            self._in_flight += 1

    def notify_batch_complete(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @abstractmethod
    def get_next_batches(self) -> list[list[SlotRequest]]:
        pass

    def reset_robot(self, robot_id: str) -> None:
        self._deadline_steps.pop(robot_id, None)
        self._latest_requests.pop(robot_id, None)
        self._latest_scheduled_requests.pop(robot_id, None)

    def clear(self, robot_id: str) -> None:
        self.reset_robot(robot_id)
        self.latency_tracker.clear(robot_id)

    def deadline(self, robot_id: str) -> float:
        """Return the time until the robot will be starved (seconds)."""
        deadline_step = self._deadline_steps[robot_id]
        latest_request = self._latest_requests.get(robot_id)
        assert latest_request is not None, f"Missing latest request for robot {robot_id}"

        steps_remaining = deadline_step - latest_request.observation_step
        return latest_request.request_timestamp + (steps_remaining / latest_request.control_hz)

    @contextmanager
    def record_timing(self) -> Generator[Callable[[], float], None, None]:
        start = time.perf_counter()
        yield lambda: time.perf_counter() - start

    def flush_decisions(self) -> list[SchedulerDecision]:
        samples = self._decisions
        self._decisions = []
        return samples

    @property
    def schedulable_requests(self) -> list[SlotRequest]:
        """Get all requests that have a greater action start step."""
        result = []
        for req in self._latest_requests.values():
            last = self._latest_scheduled_requests.get(req.robot_id)
            if last is not None and req.action_start_step <= last.action_start_step:
                continue
            result.append(req)
        return result
