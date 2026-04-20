import time
from typing import TypeVar

from action_chunk_scheduling.classes import SimulatorParameters
from action_chunk_scheduling.schedulers import LookaheadScheduler

from openpi.scheduling import RequestScheduler
from openpi.serving.schemas import SlotRequest

T = TypeVar("T", int, str)


class LookaheadActionScheduler(RequestScheduler):
    def __init__(self, batch_queue, max_batch_size=1, horizon=1000):
        super().__init__(batch_queue, max_batch_size)
        self.horizon = horizon

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() + self.in_flight > 0 or self.schedulable_requests == []:
            return []

        sim_scheduler = LookaheadScheduler(self.build_simulator_parameters())
        schedule: list[list[str]] = sim_scheduler.search()

        return [[self._latest_requests[robot_index] for robot_index in batch] for batch in schedule]

    def build_simulator_parameters(
        self,
    ) -> SimulatorParameters:
        def convert_to_ms(latencies: dict[T, float]) -> dict[T, int]:
            return {k: int(v * 1000) for k, v in latencies.items()}

        now = time.time()
        start_offsets = {}
        existing_actions = {}
        for robot_id, request in self._latest_requests.items():
            observation_step = request.observation_step
            next_step_time = request.request_timestamp
            i = 0
            while next_step_time < now:
                next_step_time = request.request_timestamp + i * (1 / request.control_hz)
                i += 1

            offset_observation_step = observation_step + i
            deadline_step = self._deadline_steps[robot_id]

            start_offsets[robot_id] = next_step_time - now
            existing_actions[robot_id] = max(0, offset_observation_step - deadline_step)

        return SimulatorParameters(
            end_time=self.horizon,
            robot_ids=list(self._latest_requests.keys()),
            d_infer=convert_to_ms(self.latency_tracker.infer_latency),
            d_send=convert_to_ms(self.latency_tracker.observation_latency),
            d_recv=convert_to_ms(self.latency_tracker.action_latency),
            start_offsets=convert_to_ms(start_offsets),
            existing_actions=existing_actions,
            execution_horizon={
                robot_id: request.execution_horizon for robot_id, request in self._latest_requests.items()
            },
        )
