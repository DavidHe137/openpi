from abc import ABC
from abc import abstractmethod

from openpi.serving.schemas import CompletionNotification
from openpi.serving.schemas import SlotRequest


class RequestScheduler(ABC):
    def __init__(self, max_batch_size: int = 1):
        self._max_batch_size = max_batch_size
        self._deadlines: dict[str, float] = {}
        # Only tracks last completed start_step per robot
        self._last_start_step: dict[str, int] = {}
        self._pending: dict[str, SlotRequest] = {}

        # Round-robin state
        self._rr_index: int = 0
        self._rr_robot_order: list[str] = []

    def update(self, request: SlotRequest) -> None:
        """Store the latest pending request for this robot."""
        self._pending[request.robot_id] = request

    @abstractmethod
    def schedule(self) -> list[list[SlotRequest]]:
        pass

    def update_deadlines(self, batch: list[SlotRequest]) -> None:
        for request in batch:
            self._deadlines[request.robot_id] = request.deadline

    def notify_complete(self, notification: CompletionNotification) -> None:
        self._last_start_step[notification.robot_id] = notification.start_step

    def reset_robot(self, robot_id: str) -> None:
        self._last_start_step.pop(robot_id, None)
        self._deadlines.pop(robot_id, None)
        self._pending.pop(robot_id, None)
