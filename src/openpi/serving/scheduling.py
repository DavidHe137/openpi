import logging

import numpy as np
from openpi_client.messages import InferResponse

from openpi.serving.schemas import ArrivedRequest

DEFAULT_EXECUTION_HORIZON = 10
DISTANCE_THRESHOLD = 0.5
MINIMUM_EXECUTION_HORIZON = 5

logger = logging.getLogger(__name__)


class RequestScheduler:
    def __init__(self):
        self._deadlines: dict[str, float] = {}
        self._last_response: dict[str, InferResponse] = {}
        # TODO: handle execution horizon for RTC requests

    def _can_infer(self, request: ArrivedRequest) -> bool:
        robot_id = request.infer_request.robot_id
        stale: bool = robot_id in self._deadlines and request.infer_request.deadline < self._deadlines[robot_id]
        inside_execution_minimum: bool = (
            robot_id in self._last_response
            and request.infer_request.start_step < self._last_response[robot_id].start_step + MINIMUM_EXECUTION_HORIZON
        )

        return not stale and not inside_execution_minimum

    def get_next_batch(self, snapshot: dict[str, ArrivedRequest], max_batch_size: int) -> list[ArrivedRequest]:
        # earliest deadline first + max batch size
        candidates: list[ArrivedRequest] = [request for request in snapshot.values() if self._can_infer(request)]
        return sorted(candidates, key=lambda x: x.infer_request.deadline)[:max_batch_size]

    def update_deadlines(self, batch: list[ArrivedRequest]) -> None:
        for request in batch:
            self._deadlines[request.infer_request.robot_id] = request.infer_request.deadline

    def calculate_execution_horizon(self, robot_id: str, actions: np.ndarray) -> int:
        # FIXME: Implement variable execution horizon
        return len(actions)
        # start_index = start_step - last_response.start_step
        # prev_actions = last_response.actions[start_index:]
        # current_actions = actions[: len(prev_actions)]
        # if len(prev_actions) == 0:
        #     logger.info(f"Got no overlap. Previous step is {last_response.start_step}, current step is {start_step}")
        #     return DEFAULT_EXECUTION_HORIZON

        # distance = np.linalg.norm(prev_actions - current_actions, axis=1)
        # logger.info(f"Distance: {distance}")

        # return int(np.argmax(distance > DISTANCE_THRESHOLD))

    def update_last_response(self, robot_id: str, response: InferResponse) -> None:
        self._last_response[robot_id] = response
