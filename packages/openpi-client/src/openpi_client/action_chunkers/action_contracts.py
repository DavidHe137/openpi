from abc import ABC, abstractmethod

# FIXME: imports here are so ugly because its shared between client and server
from openpi_client.schemas import ActionChunk


class ActionContract(ABC):
    @abstractmethod
    def estimate_deadline_server_side(
        self, request_timestamp: float, arrival_timestamp: float, execution_horizon: int, control_hz: float
    ) -> float:
        """Estimate the deadline for an infer request on the server side, which may be used for scheduling decisions."""
        pass

    @abstractmethod
    def calculate_execution_horizon(
        self, action_chunk: ActionChunk, step_duration: float, next_step_time: float, next_action_step: int
    ) -> int:
        """Estimate the execution horizon for an action chunk based on its properties and the step duration."""
        pass


class MaximalActionContract(ActionContract):
    """We use as many actions in a chunk as possible"""

    def estimate_deadline_server_side(
        self, request_timestamp: float, arrival_timestamp: float, execution_horizon: int, control_hz: float
    ) -> float:
        return request_timestamp + execution_horizon / control_hz

    def calculate_execution_horizon(
        self, action_chunk: ActionChunk, step_duration: float, next_step_time: float, next_action_step: int
    ) -> int:
        return action_chunk.execution_horizon


class ObservationActionContract(ActionContract):
    """
    A chunk is valid from [observation_step, observation_step + execution_horizon).
    Intended to work with chunks that include actions dropped by delay.
    """

    def estimate_deadline_server_side(
        self, request_timestamp: float, arrival_timestamp: float, execution_horizon: int, control_hz: float
    ) -> float:
        return request_timestamp + execution_horizon / control_hz

    def calculate_execution_horizon(
        self, action_chunk: ActionChunk, step_duration: float, next_step_time: float, next_action_step: int
    ) -> int:
        cutoff = action_chunk.request_timestamp + action_chunk.execution_horizon * step_duration
        time_left = cutoff - next_step_time
        steps_left = int(time_left // step_duration)
        action_step_end = next_action_step + steps_left  # we do not want to include this index
        return max(action_step_end - action_chunk.action_start_step, 0)


class ArrivalActionContract(ActionContract):
    """
    A chunk is valid from [arrival_time, arrival_time + execution_horizon * step_duration).
    Intended to work with chunks that do not include actions dropped by delay, so the client can start executing as soon as the chunk arrives.
    """

    def estimate_deadline_server_side(
        self, request_timestamp: float, arrival_timestamp: float, execution_horizon: int, control_hz: float
    ) -> float:
        return arrival_timestamp + execution_horizon / control_hz

    def calculate_execution_horizon(
        self, action_chunk: ActionChunk, step_duration: float, next_step_time: float, next_action_step: int
    ) -> int:
        return action_chunk.execution_horizon


ACTION_CONTRACT_REGISTRY = {
    "maximal": MaximalActionContract,
    "observation": ObservationActionContract,
    "arrival": ArrivalActionContract,
}
