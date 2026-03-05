from abc import ABC
from abc import abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
import multiprocessing as mp
import time

from openpi.scheduling.latency import LatencyTracker
from openpi.serving.schemas import AckNotification
from openpi.serving.schemas import CompletionNotification
from openpi.serving.schemas import SchedulerTimingSample
from openpi.serving.schemas import SlotRequest


class RequestScheduler(ABC):
    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        batch_profile: dict[int, float] | None = None,
    ):
        self._batch_queue = batch_queue
        self._max_batch_size = max_batch_size
        self._batch_profile_ms: dict[int, float] = batch_profile or {}

        self._latest_requests: dict[str, SlotRequest] = {}
        self._latest_scheduled_requests: dict[str, SlotRequest] = {}
        self._deadlines: dict[str, float] = {}  # includes chunks that have been sent to the GPU but not yet completed
        self._timing_samples: list[SchedulerTimingSample] = []
        self.latency = LatencyTracker()

    def update(self, request: SlotRequest) -> None:
        self._latest_requests[request.robot_id] = request
        if request.deadline is not None and request.deadline > self._deadlines.get(request.robot_id, 0):
            self._deadlines[request.robot_id] = request.deadline
        self.latency.update_obs(request.robot_id, request.arrival_timestamp, request.request_timestamp)

    def update_completion(self, notification: CompletionNotification) -> None:
        self.latency.update_infer(notification.batch_size, notification.inference_duration_ms)

    def update_ack(self, notification: AckNotification) -> None:
        self.latency.update_action_delivery(
            notification.robot_id, notification.receive_time, notification.server_send_time
        )

    def schedule(self) -> None:
        """Return a list of batches of requests to be sent to the GPU."""
        batches = self.get_next_batches()
        for batch in batches:
            for request in batch:
                self._deadlines[request.robot_id] = request.deadline
                self._latest_scheduled_requests[request.robot_id] = request
            self._batch_queue.put_nowait(batch)

    @abstractmethod
    def get_next_batches(self) -> list[list[SlotRequest]]:
        pass

    def reset_robot(self, robot_id: str) -> None:
        self._deadlines.pop(robot_id, None)
        self._latest_requests.pop(robot_id, None)
        self._latest_scheduled_requests.pop(robot_id, None)
        self.latency.reset_robot(robot_id)

    @contextmanager
    def record_timing(self, metric_name: str) -> Generator[None, None, None]:
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter_ns() - start_ns) / 1e6
            self._timing_samples.append(
                SchedulerTimingSample(
                    scheduler_name=self.__class__.__name__,
                    metric_name=metric_name,
                    duration_ms=duration_ms,
                    recorded_at=time.time(),
                )
            )

    def flush_timing_samples(self) -> list[SchedulerTimingSample]:
        samples = self._timing_samples
        self._timing_samples = []
        return samples

    def _get_schedulable_requests(self) -> list[SlotRequest]:
        """Get all requests that are not yet scheduled and past the minimum execution horizon."""
        result = []
        for req in self._latest_requests.values():
            last = self._latest_scheduled_requests.get(req.robot_id)
            if last is req:
                continue
            if last is not None and req.start_step < last.start_step + last.min_execution_horizon:
                continue
            result.append(req)
        return result
