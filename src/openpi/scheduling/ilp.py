from __future__ import annotations

import math
import multiprocessing as mp
import time

import pulp

from openpi.scheduling import RequestScheduler
from openpi.serving.schemas import SlotRequest


def _push_plan_left(
    plan: dict[int, list[int]],
    d_inf: dict[int, int],
    num_ticks: int,
    server_blocked: set[int],
) -> dict[int, list[int]]:
    """Repack batches to the earliest available slot, respecting blocked ticks."""
    batches = sorted(plan.items())
    new_plan: dict[int, list[int]] = {}
    local_blocked = set(server_blocked)

    for _, r_idxs in batches:
        b = len(r_idxs)
        length = d_inf[b]
        # Find earliest contiguous free gap of `length` ticks
        t = 0
        t_sched = None
        while t + length <= num_ticks:
            if all(t + i not in local_blocked for i in range(length)):
                t_sched = t
                break
            t += 1
        if t_sched is None:
            break
        new_plan[t_sched] = r_idxs
        local_blocked.update(range(t_sched, t_sched + length))

    return new_plan


def _solve_ilp(
    num_ticks: int,
    num_robots: int,
    tiers: range,
    d_inf: dict[int, int],
    d_recv: dict[int, int],
    chunk_ticks: int,
    current_cov_ticks: dict[int, int],
    server_blocked: frozenset[int],
    ilp_timeout_s: float,
    *,
    fill_mode: bool,
) -> dict[int, list[int]]:
    """Solve one ILP over the horizon [0, num_ticks) and return tick -> list[r_idx]."""
    already_covered = {(t, r): t < current_cov_ticks.get(r, 0) for t in range(num_ticks) for r in range(num_robots)}

    prob = pulp.LpProblem("ILPScheduler", pulp.LpMinimize)

    x = {
        (t, r, b): pulp.LpVariable(f"x_{t}_{r}_{b}", cat=pulp.LpBinary)
        for t in range(num_ticks)
        for r in range(num_robots)
        for b in tiers
    }
    y = {(t, b): pulp.LpVariable(f"y_{t}_{b}", cat=pulp.LpBinary) for t in range(num_ticks) for b in tiers}

    if fill_mode:
        prob += -pulp.lpSum(x.values())
    else:
        s = {
            (t, r): pulp.LpVariable(f"s_{t}_{r}", lowBound=0, upBound=1)
            for t in range(num_ticks)
            for r in range(num_robots)
            if not already_covered[t, r]
        }
        prob += pulp.lpSum(s.values()) if s else pulp.lpSum([])

        for t in range(num_ticks):
            for r in range(num_robots):
                if already_covered[t, r]:
                    continue
                covering = []
                for b in tiers:
                    offset = d_inf[b] + d_recv[r]
                    lb = max(0, t - chunk_ticks - offset + 1)
                    ub = t - offset
                    covering.extend(x[tau, r, b] for tau in range(lb, min(ub + 1, num_ticks)))
                prob += (
                    pulp.lpSum(covering) + s[t, r] >= 1,
                    f"coverage_{t}_{r}",
                )

    # Server exclusivity
    for t in range(num_ticks):
        active = []
        for b in tiers:
            active.extend(y[tau, b] for tau in range(max(0, t - d_inf[b] + 1), t + 1))
        rhs = 0 if t in server_blocked else 1
        prob += (pulp.lpSum(active) <= rhs, f"server_{t}")

    # Tier uniqueness
    for t in range(num_ticks):
        prob += (pulp.lpSum(y[t, b] for b in tiers) <= 1, f"tier_{t}")

    # Batch consistency
    for t in range(num_ticks):
        for b in tiers:
            prob += (
                pulp.lpSum(x[t, r, b] for r in range(num_robots)) <= b * y[t, b],
                f"consistency_{t}_{b}",
            )

    # Linking: each robot assigned to at most one tier per tick
    for t in range(num_ticks):
        for r in range(num_robots):
            prob += (
                pulp.lpSum(x[t, r, b] for b in tiers) <= 1,
                f"linking_{t}_{r}",
            )

    try:
        solver = pulp.GUROBI(msg=0, timeLimit=ilp_timeout_s)
        prob.solve(solver)
    except Exception:
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=ilp_timeout_s, threads=4))

    plan: dict[int, list[int]] = {}
    for t in range(num_ticks):
        scheduled = [r for r in range(num_robots) if any((pulp.value(x[t, r, b]) or 0) > 0.5 for b in tiers)]
        if scheduled:
            plan[t] = scheduled

    return plan


class ILPScheduler(RequestScheduler):
    """Receding-horizon ILP scheduler that minimises robot starvation.

    On the first call with pending requests (or when the robot set changes), solves
    an ILP over `horizon_ms` and returns all planned batches at once.  Subsequent
    calls return [] until the horizon expires, then trigger a fresh replan.

    With max_iterations > 1, additional ILP passes fill remaining server capacity
    with re-inferences (freshness improvement without extra starvation).
    """

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        batch_profile: dict[int, float] | None = None,
        *,
        horizon_ms: int = 2000,
        timestep_ms: int = 50,
        action_horizon_steps: int = 10,
        control_hz: int = 20,
        max_iterations: int = 2,
        ilp_timeout_s: float = 1.0,
    ) -> None:
        super().__init__(batch_queue, max_batch_size, batch_profile)

        if timestep_ms <= 0:
            raise ValueError("timestep_ms must be positive")
        if horizon_ms <= 0:
            raise ValueError("horizon_ms must be positive")
        if action_horizon_steps <= 0:
            raise ValueError("action_horizon_steps must be positive")
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")

        self._timestep_ms = float(timestep_ms)
        self._horizon_ticks = max(1, math.ceil(horizon_ms / timestep_ms))
        self._chunk_ticks = max(1, math.ceil(action_horizon_steps / control_hz * 1000.0 / timestep_ms))
        self._chunk_duration_s = action_horizon_steps / control_hz
        self._latency_ticks = {
            b: max(1, math.ceil(self._latency_ms(b) / timestep_ms)) for b in range(1, max_batch_size + 1)
        }
        self._latency_s = {b: self._latency_ms(b) / 1000.0 for b in range(1, max_batch_size + 1)}
        self._max_iterations = max_iterations
        self._ilp_timeout_s = ilp_timeout_s

        self._predicted_valid_until: dict[str, float] = {}
        self._server_available_at: float = 0.0
        self._plan_end_time: float = 0.0
        self._known_robot_ids: frozenset[str] = frozenset()

    def schedule(self) -> None:
        batches = self.get_next_batches()
        now = time.time()
        for batch in batches:
            batch_size = len(batch)
            start_time = max(now, self._server_available_at)
            finish_time = start_time + self._latency_s[batch_size]
            self._server_available_at = finish_time
            for request in batch:
                self._deadlines[request.robot_id] = request.deadline
                self._latest_scheduled_requests[request.robot_id] = request
                self._predicted_valid_until[request.robot_id] = finish_time + self._chunk_duration_s
            self._batch_queue.put_nowait(batch)
            now = finish_time

    def get_next_batches(self) -> list[list[SlotRequest]]:
        if not self._latest_requests:
            return []

        now = time.time()
        current_robot_ids = frozenset(self._latest_requests)

        if now < self._plan_end_time and current_robot_ids == self._known_robot_ids:
            return []

        with self.record_timing("ilp_solve"):
            batches = self._solve_plan(now, current_robot_ids)

        self._plan_end_time = now + self._horizon_ticks * self._timestep_ms / 1000.0
        self._known_robot_ids = current_robot_ids
        return batches

    def update_completion(self, notification) -> None:
        super().update_completion(notification)
        self._server_available_at = min(self._server_available_at, time.time())

    def reset_robot(self, robot_id: str) -> None:
        super().reset_robot(robot_id)
        self._predicted_valid_until.pop(robot_id, None)

    def _latency_ms(self, batch_size: int) -> float:
        latency_ms = self._batch_profile_ms.get(batch_size)
        if latency_ms is None:
            raise ValueError(f"Missing batch profile entry for batch_size={batch_size}")
        return latency_ms

    def _solve_plan(self, now: float, robot_ids_set: frozenset[str]) -> list[list[SlotRequest]]:
        robot_ids = sorted(robot_ids_set)
        num_robots = len(robot_ids)
        if num_robots == 0:
            return []

        num_ticks = self._horizon_ticks
        tiers = range(1, self._max_batch_size + 1)
        d_inf = {b: self._latency_ticks[b] for b in tiers}
        min_d_inf = min(d_inf.values())

        # Per-robot recv latency in ticks (0 if unknown)
        d_recv = {}
        for r_idx, rid in enumerate(robot_ids):
            delivery_ms = self.latency.action_delivery_ms(rid) or 0.0
            d_recv[r_idx] = max(0, math.ceil(delivery_ms / self._timestep_ms))

        # Initial coverage from predicted_valid_until
        current_cov_ticks: dict[int, int] = {}
        for r_idx, rid in enumerate(robot_ids):
            pvu = self._predicted_valid_until.get(rid, 0.0)
            remaining_ms = max(0.0, (pvu - now) * 1000.0)
            current_cov_ticks[r_idx] = max(0, math.ceil(remaining_ms / self._timestep_ms))

        combined_plan: dict[int, list[int]] = {}
        server_blocked: set[int] = set()

        for iteration in range(self._max_iterations):
            free_count = sum(1 for t in range(num_ticks) if t not in server_blocked)
            if free_count < min_d_inf:
                break

            new_plan = _solve_ilp(
                num_ticks=num_ticks,
                num_robots=num_robots,
                tiers=tiers,
                d_inf=d_inf,
                d_recv=d_recv,
                chunk_ticks=self._chunk_ticks,
                current_cov_ticks=current_cov_ticks,
                server_blocked=frozenset(server_blocked),
                ilp_timeout_s=self._ilp_timeout_s,
                fill_mode=(iteration > 0),
            )
            if not new_plan:
                break

            packed = _push_plan_left(new_plan, d_inf, num_ticks, server_blocked)
            if not packed:
                break

            combined_plan.update(packed)

            # Update coverage for next iteration
            for tau, r_idxs in packed.items():
                b = len(r_idxs)
                for r_idx in r_idxs:
                    arrival_tick = tau + d_inf[b] + d_recv[r_idx]
                    coverage_end = arrival_tick + self._chunk_ticks
                    current_cov_ticks[r_idx] = max(current_cov_ticks.get(r_idx, 0), coverage_end)

            # Mark server ticks as blocked
            for tau, r_idxs in packed.items():
                b = len(r_idxs)
                server_blocked.update(range(tau, min(tau + d_inf[b], num_ticks)))

        # Convert tick-indexed plan to SlotRequest batches (sorted by tick)
        request_by_robot = {rid: self._latest_requests[rid] for rid in robot_ids if rid in self._latest_requests}
        batches = []
        for tick in sorted(combined_plan):
            batch = [
                request_by_robot[robot_ids[r_idx]]
                for r_idx in combined_plan[tick]
                if robot_ids[r_idx] in request_by_robot
            ]
            if batch:
                batches.append(batch)

        return batches
