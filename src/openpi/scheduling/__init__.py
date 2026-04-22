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
from openpi.simulation.classes import ActionChunk
from openpi.simulation.classes import RobotState

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
        # Mirror of each robot's scheduled-chunk history. The same ActionChunk / RobotState
        # types the offline simulator uses (see openpi.simulation.classes), so deadlines,
        # in-flight chunk counts, and schedulable checks can be derived from a single source
        # of truth and compared directly against sim ground truth in tests.
        self._mirror: dict[str, RobotState] = {}
        self._decisions: list[SchedulerDecision] = []
        self.latency_tracker = EMALatencyTracker()  # TODO: allow different latency trackers
        self.next_batch_id = itertools.count(1)
        self._in_flight = 0

    def update(self, request: SlotRequest) -> None:
        self._latest_requests[request.robot_id] = request
        self._mirror.setdefault(request.robot_id, RobotState(step_based_coverage=True))
        self.latency_tracker.update_obs(request.robot_id, request.arrival_timestamp, request.request_timestamp)

    def update_completion(self, notification: CompletionNotification) -> None:
        self.latency_tracker.update_infer(notification.batch_size, notification.inference_duration)

    def update_ack(self, notification: AckNotification) -> None:
        self.latency_tracker.update_action_delivery(
            notification.robot_id,
            notification.receive_time,
            notification.server_send_time,
        )

    def _record_scheduled_chunk(self, request: SlotRequest, batch_size: int) -> ActionChunk:
        """Append a chunk to the mirror for a request we're dispatching, and remember
        the SlotRequest itself. Subclasses that override ``schedule()`` must call this
        (or ``schedule()`` below which calls it) so mirror-derived state stays current.
        """
        try:
            total_latency_s = self.latency_tracker.total_latency(request.robot_id, batch_size)
        except KeyError:
            total_latency_s = 0.0
        arrival_step = request.observation_step + max(0, round(total_latency_s * request.control_hz))
        chunk = ActionChunk(
            start_action=request.action_start_step,
            horizon=request.execution_horizon,
            arrival_step=arrival_step,
            observation_step=request.observation_step,
        )
        state = self._mirror.setdefault(request.robot_id, RobotState(step_based_coverage=True))
        state.chunks.append(chunk)
        self._latest_scheduled_requests[request.robot_id] = request
        return chunk

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
            # Capture deadlines before mirror mutations shift them, sort earliest first.
            trace = self.collect_trace(batch)
            annotated = []
            for request in batch:
                self._record_scheduled_chunk(request, batch_size)
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
        self._mirror.pop(robot_id, None)
        self._latest_requests.pop(robot_id, None)
        self._latest_scheduled_requests.pop(robot_id, None)

    def clear(self, robot_id: str) -> None:
        self.reset_robot(robot_id)
        self.latency_tracker.clear(robot_id)

    def deadline(self, robot_id: str) -> float:
        """Wall-clock time at which the robot runs out of actions (the 'anticipated'
        deadline — includes chunks dispatched but not yet arrived). Derived from the mirror;
        falls back to the client's ``deadline_step`` hint before the first schedule."""
        latest_request = self._latest_requests.get(robot_id)
        assert latest_request is not None, f"Missing latest request for robot {robot_id}"

        chunks = self._mirror[robot_id].chunks if robot_id in self._mirror else ()
        if chunks:
            deadline_step = max(chunk.start_action + chunk.horizon for chunk in chunks)
        else:
            deadline_step = latest_request.deadline_step

        steps_remaining = deadline_step - latest_request.observation_step
        return latest_request.request_timestamp + (steps_remaining / latest_request.control_hz)

    def in_flight_chunks(self, robot_id: str) -> list[ActionChunk]:
        """Chunks dispatched for this robot whose predicted arrival step is still
        beyond the latest observation step — i.e. still on the wire to the robot."""
        latest_request = self._latest_requests.get(robot_id)
        if latest_request is None or robot_id not in self._mirror:
            return []
        return [c for c in self._mirror[robot_id].chunks if c.arrival_step > latest_request.observation_step]

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
        """Latest-per-robot requests whose observation is newer than any chunk we've already dispatched."""
        result = []
        for req in self._latest_requests.values():
            chunks = self._mirror[req.robot_id].chunks if req.robot_id in self._mirror else ()
            if chunks and req.action_start_step <= chunks[-1].start_action:
                continue
            result.append(req)
        return result
