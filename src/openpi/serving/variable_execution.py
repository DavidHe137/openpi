import logging

import numpy as np
from openpi_client.messages import InferResponse

DEFAULT_EXECUTION_HORIZON = 10
DISTANCE_THRESHOLD = 0.5

logger = logging.getLogger(__name__)


def calculate_execution_horizon(last_response: InferResponse, start_step: int, actions: np.ndarray) -> int:
    start_index = start_step - last_response.start_step
    prev_actions = last_response.actions[start_index:]
    current_actions = actions[: len(prev_actions)]
    if len(prev_actions) == 0:
        logger.info(f"Got no overlap. Previous step is {last_response.start_step}, current step is {start_step}")
        return DEFAULT_EXECUTION_HORIZON

    distance = np.linalg.norm(prev_actions - current_actions, axis=1)
    logger.info(f"Distance: {distance}")

    return int(np.argmax(distance > DISTANCE_THRESHOLD))
