import heapq

from openpi.serving.schemas import ArrivedRequest


class RequestQueue:
    """Priority queue for requests, ordered by deadline. Drops stale requests."""

    def __init__(self):
        self._last_processed_timestamp: dict[str, float] = {}
        self._queue: list[tuple[float, ArrivedRequest]] = []

    def add(self, request: ArrivedRequest) -> None:
        deadline = request.infer_request.deadline
        heapq.heappush(self._queue, (deadline, request))

    def clear_stale(self) -> None:
        """Remove stale requests (older than last processed for same robot)."""
        while self._queue:
            request: ArrivedRequest = self._queue[0][1]
            robot_id = request.infer_request.robot_id
            request_timestamp = request.infer_request.request_timestamp
            if self._last_processed_timestamp.get(robot_id, 0) > request_timestamp:
                heapq.heappop(self._queue)
            else:
                break

    def pop(self) -> ArrivedRequest:
        request: ArrivedRequest = heapq.heappop(self._queue)[1]
        self._last_processed_timestamp[request.infer_request.robot_id] = request.infer_request.request_timestamp
        return request

    @property
    def empty(self) -> bool:
        return not self._queue
