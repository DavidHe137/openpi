from __future__ import annotations

from collections.abc import Collection
import dataclasses
import multiprocessing as mp
import random
import statistics
import time

from openpi.scheduling import RequestScheduler
from openpi.serving.schemas import SchedulerDecision
from openpi.serving.schemas import SlotRequest


@dataclasses.dataclass
class _RobotSchedulerState:
    age: int = 0
    debt: int = 0
    last_served_decision_idx: int = 0
    deficit: float = 0.0


class GreedyScheduler(RequestScheduler):
    """Earliest-deadline-first: sort all pending requests by deadline."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0:
            return []

        candidates = self._get_schedulable_requests()
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda r: self._deadlines.get(r.robot_id, r.deadline))
        earliest_deadline = self._deadlines.get(candidates[0].robot_id, candidates[0].deadline)
        batch_size = self.get_largest_batch_size(earliest_deadline)
        return [candidates[:batch_size]]

    def get_largest_batch_size(self, deadline: float) -> int:
        """Return the largest batch size whose profiled latency fits within the time remaining until deadline."""
        time_remaining_ms = (deadline - time.time()) * 1e3
        for batch_size in range(self._max_batch_size, 0, -1):
            if self._batch_profile_ms.get(batch_size, 0) <= time_remaining_ms:
                return batch_size
        return self.most_efficient_batch_size

    @property
    def most_efficient_batch_size(self) -> int:
        """Batch size with the best throughput (requests / ms). Falls back to 1."""
        if not self._batch_profile_ms:
            return 1
        return max(self._batch_profile_ms, key=lambda bs: bs / self._batch_profile_ms[bs])


class RoundRobinScheduler(RequestScheduler):
    """Cycle through robots starting from the current pointer, fill to max_batch_size."""

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        batch_profile: dict[int, float] | None = None,
    ):
        super().__init__(batch_queue, max_batch_size, batch_profile)
        self._rr_index: int = 0
        self._rr_robot_order: list[str] = []

    def update(self, request: SlotRequest) -> None:
        super().update(request)
        if request.robot_id not in self._rr_robot_order:
            self._rr_robot_order.append(request.robot_id)

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0:
            return []

        candidate_by_robot = {req.robot_id: req for req in self._get_schedulable_requests()}
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

    def reset_robot(self, robot_id: str) -> None:
        super().reset_robot(robot_id)
        # FIXME: temporary hack to remove robot on reset
        if robot_id in self._rr_robot_order:
            removed_index = self._rr_robot_order.index(robot_id)
            self._rr_robot_order.remove(robot_id)
            if removed_index < self._rr_index:
                self._rr_index = max(0, self._rr_index - 1)


class WDRRScheduler(RequestScheduler):
    """Weighted Deficit Round Robin with fairness boosts."""

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        batch_profile: dict[int, float] | None = None,
        *,
        scheduler_ema_alpha: float = 0.1,
        scheduler_lambda_age: float = 0.25,
        scheduler_lambda_debt: float = 0.5,
        scheduler_service_window_decisions: int = 20,
        wdrr_q0: float = 1.0,
    ) -> None:
        super().__init__(
            batch_queue=batch_queue,
            max_batch_size=max_batch_size,
            batch_profile=batch_profile,
            latency_alpha=scheduler_ema_alpha,
        )
        self._lambda_age = float(scheduler_lambda_age)
        self._lambda_debt = float(scheduler_lambda_debt)
        self._service_window_decisions = int(scheduler_service_window_decisions)
        self._q0 = float(wdrr_q0)
        self._state_by_robot: dict[str, _RobotSchedulerState] = {}
        self._decision_idx: int = 0
        self._rr_index: int = 0
        self._rr_robot_order: list[str] = []

    def update(self, request: SlotRequest) -> None:
        super().update(request)
        self._get_state(request.robot_id)
        if request.robot_id not in self._rr_robot_order:
            self._rr_robot_order.append(request.robot_id)

    def reset_robot(self, robot_id: str) -> None:
        super().reset_robot(robot_id)
        self._state_by_robot.pop(robot_id, None)
        if robot_id in self._rr_robot_order:
            removed_index = self._rr_robot_order.index(robot_id)
            self._rr_robot_order.remove(robot_id)
            if removed_index < self._rr_index:
                self._rr_index = max(0, self._rr_index - 1)

    def schedule(self) -> None:
        candidates = self._get_schedulable_requests()
        eligible_robot_ids = {request.robot_id for request in candidates}

        with self.record_timing() as duration:
            batches = self.get_next_batches()

        for batch in batches:
            batch_size = len(batch)
            if batch_size == 0:
                continue

            candidate_entries = sorted(
                ({"robot_id": r.robot_id, "deadline": self._deadlines.get(r.robot_id, r.deadline)} for r in candidates),
                key=lambda x: x["deadline"],
            )
            batch_entries = sorted(
                ({"robot_id": r.robot_id, "deadline": self._deadlines.get(r.robot_id, r.deadline)} for r in batch),
                key=lambda x: x["deadline"],
            )

            annotated: list[SlotRequest] = []
            for request in batch:
                self._deadlines[request.robot_id] = request.deadline + request.execution_horizon / request.control_hz
                self._latest_scheduled_requests[request.robot_id] = request
                d_ms = self.latency.total_delivery_ms(request.robot_id, batch_size)
                step_ms = 1000.0 / request.control_hz
                d_steps = round(d_ms / step_ms) if d_ms is not None else 0
                annotated.append(dataclasses.replace(request, estimated_d_param=d_steps))

            scheduled_robot_ids = {request.robot_id for request in batch}
            self._apply_post_dispatch_updates(eligible_robot_ids, scheduled_robot_ids)

            self._decisions.append(
                SchedulerDecision(
                    scheduler_name=self.__class__.__name__,
                    metric_name="batch_scheduled",
                    duration_ms=duration() * 1e3,
                    recorded_at=time.time(),
                    candidates=candidate_entries,
                    scheduled=batch_entries,
                )
            )
            self._batch_queue.put_nowait(annotated)

    def _get_state(self, robot_id: str) -> _RobotSchedulerState:
        state = self._state_by_robot.get(robot_id)
        if state is None:
            state = _RobotSchedulerState(last_served_decision_idx=self._decision_idx)
            self._state_by_robot[robot_id] = state
        return state

    def _component_fallback_mean(
        self,
        per_robot_values: dict[str, float],
        active_robot_ids: Collection[str],
    ) -> float:
        values = [float(per_robot_values[robot_id]) for robot_id in active_robot_ids if robot_id in per_robot_values]
        return float(sum(values) / len(values)) if values else 0.0

    def _infer_fallback_mean(self) -> float:
        infer_map = self.latency.infer_map
        values = list(infer_map.values())
        return float(sum(values) / len(values)) if values else 0.0

    def _estimate_costs_ms(
        self,
        candidates: list[SlotRequest],
        *,
        batch_size: int,
    ) -> dict[str, float]:
        if not candidates:
            return {}

        active_robot_ids = [request.robot_id for request in candidates]
        obs_map = self.latency.obs_network_map
        action_map = self.latency.action_delivery_map
        infer_ms = self.latency.infer_ms(batch_size)
        if infer_ms is None:
            infer_ms = self._infer_fallback_mean()

        obs_mean = self._component_fallback_mean(obs_map, active_robot_ids)
        action_mean = self._component_fallback_mean(action_map, active_robot_ids)
        inferred_costs: dict[str, float] = {}
        for request in candidates:
            obs_ms = float(obs_map.get(request.robot_id, obs_mean))
            action_ms = float(action_map.get(request.robot_id, action_mean))
            horizon_ms = float(request.execution_horizon) * (1000.0 / float(request.control_hz))
            total = max(0.0, obs_ms) + max(0.0, action_ms) + max(0.0, float(infer_ms)) + max(0.0, horizon_ms)
            inferred_costs[request.robot_id] = total
        return inferred_costs

    def _score_bonus(self, robot_id: str) -> float:
        state = self._get_state(robot_id)
        return self._lambda_age * float(state.age) + self._lambda_debt * float(state.debt)

    def _overdue_robot_ids(self, eligible_robot_ids: Collection[str]) -> set[str]:
        overdue: set[str] = set()
        for robot_id in eligible_robot_ids:
            state = self._get_state(robot_id)
            if (self._decision_idx - state.last_served_decision_idx) >= self._service_window_decisions:
                overdue.add(robot_id)
        return overdue

    def _apply_post_dispatch_updates(self, eligible_robot_ids: set[str], scheduled_robot_ids: set[str]) -> None:
        for robot_id in eligible_robot_ids:
            state = self._get_state(robot_id)
            if robot_id in scheduled_robot_ids:
                state.age = 0
                state.debt = 0
                state.last_served_decision_idx = self._decision_idx
            else:
                state.age += 1
                state.debt += 1
        self._decision_idx += 1

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0:
            return []

        candidate_by_robot = {request.robot_id: request for request in self._get_schedulable_requests()}
        if not candidate_by_robot:
            return []

        if not self._rr_robot_order:
            self._rr_robot_order = sorted(candidate_by_robot)

        batch_size = min(self._max_batch_size, len(candidate_by_robot))
        candidates = list(candidate_by_robot.values())
        cost_by_robot_ms = self._estimate_costs_ms(candidates, batch_size=batch_size)

        eligible_robot_ids = list(candidate_by_robot.keys())
        c_ref = statistics.median(cost_by_robot_ms[robot_id] for robot_id in eligible_robot_ids)
        c_ref = max(c_ref, 1e-6)
        quantum_by_robot = {
            robot_id: self._q0 * (1.0 / max(cost_by_robot_ms[robot_id] / c_ref, 1e-6))
            for robot_id in eligible_robot_ids
        }
        priority_by_robot = {
            robot_id: self._get_state(robot_id).deficit + self._score_bonus(robot_id) for robot_id in eligible_robot_ids
        }

        selected_robot_ids: list[str] = []
        overdue_robot_ids = self._overdue_robot_ids(set(eligible_robot_ids))
        for robot_id in sorted(overdue_robot_ids, key=lambda rid: (-priority_by_robot[rid], rid)):
            if robot_id not in candidate_by_robot:
                continue
            selected_robot_ids.append(robot_id)
            if len(selected_robot_ids) >= batch_size:
                break

        n_robots = len(self._rr_robot_order)
        if len(selected_robot_ids) < batch_size and n_robots > 0:
            idx = self._rr_index % n_robots
            for _ in range(n_robots):
                robot_id = self._rr_robot_order[idx]
                if robot_id in candidate_by_robot and robot_id not in selected_robot_ids:
                    state = self._get_state(robot_id)
                    state.deficit += quantum_by_robot[robot_id]
                    priority = state.deficit + self._score_bonus(robot_id)
                    priority_by_robot[robot_id] = priority
                    if priority >= 1.0:
                        selected_robot_ids.append(robot_id)
                        if len(selected_robot_ids) >= batch_size:
                            idx = (idx + 1) % n_robots
                            break
                idx = (idx + 1) % n_robots
            self._rr_index = idx

        if not selected_robot_ids:
            fallback_robot_id = sorted(eligible_robot_ids, key=lambda rid: (-priority_by_robot[rid], rid))[0]
            selected_robot_ids = [fallback_robot_id]

        for robot_id in selected_robot_ids:
            self._get_state(robot_id).deficit -= 1.0

        return [[candidate_by_robot[robot_id] for robot_id in selected_robot_ids]]


class RandomBatchScheduler(RequestScheduler):
    """Randomly select up to max_batch_size from pending requests."""

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0:
            return []

        candidates = list(self._get_schedulable_requests())
        if not candidates:
            return []

        k = min(self._max_batch_size, len(candidates))
        return [random.sample(candidates, random.randint(1, k))]
