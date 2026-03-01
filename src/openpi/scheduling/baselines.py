import random

from openpi.scheduling import RequestScheduler
from openpi.serving.schemas import SlotRequest


class GreedyScheduler(RequestScheduler):
    """Earliest-deadline-first: sort all pending requests by deadline."""

    def schedule(self) -> list[list[SlotRequest]]:
        candidates = sorted(self._pending.values(), key=lambda r: r.deadline)
        return [candidates[: self._max_batch_size]] if candidates else []


class RoundRobinScheduler(RequestScheduler):
    """Cycle through robots starting from the current pointer, fill to max_batch_size."""

    def schedule(self) -> list[list[SlotRequest]]:
        candidate_by_robot = {req.robot_id: req for req in self._pending.values()}
        n_robots = len(self._rr_robot_order)
        if not candidate_by_robot or n_robots == 0:
            return []

        batch: list[SlotRequest] = []
        idx = self._rr_index % n_robots
        for _ in range(n_robots):
            robot_id = self._rr_robot_order[idx]
            if robot_id in candidate_by_robot:
                batch.append(candidate_by_robot[robot_id])
            idx = (idx + 1) % n_robots
            if len(batch) == self._max_batch_size:
                break

        self._rr_index = idx
        return [batch] if batch else []


class RandomBatchScheduler(RequestScheduler):
    """Randomly select up to max_batch_size from pending requests."""

    def schedule(self) -> list[list[SlotRequest]]:
        candidates = list(self._pending.values())
        if not candidates:
            return []
        k = min(self._max_batch_size, len(candidates))
        return [random.sample(candidates, k)]
