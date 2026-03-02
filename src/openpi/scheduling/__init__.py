from abc import ABC
from abc import abstractmethod
import multiprocessing as mp

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

    def update(self, request: SlotRequest) -> None:
        self._latest_requests[request.robot_id] = request
        if request.deadline is not None and request.deadline > self._deadlines.get(request.robot_id, 0):
            self._deadlines[request.robot_id] = request.deadline

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

    def _get_schedulable_requests(self) -> list[SlotRequest]:
        """Get all requests that are not yet scheduled."""
        # TODO: might add more logic here in the future
        return [
            req
            for req in self._latest_requests.values()
            if req is not self._latest_scheduled_requests.get(req.robot_id, None)
        ]
