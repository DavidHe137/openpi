from math import ceil

import pulp

from .classes import CONTROL_PERIOD
from .classes import SimulatorParameters
from .classes import time_to_step

SOLVER_TIMEOUT = 100


def _delay_in_steps(ms: int) -> int:
    """Convert a millisecond delay to control steps (ceiling)."""
    return ceil(ms / CONTROL_PERIOD)


def solve_ilp_batched(params: SimulatorParameters) -> dict[int, list[int]]:
    """
    Solve the optimal action chunk scheduling problem with batching.

    NOTE: This formulation assumes a uniform control-step grid (no per-robot
    jitter). Use solve_ilp_batched_exact for non-uniform step times.

    The ILP operates on a control-step time axis. Millisecond-based delays
    (d_infer, d_send, d_recv) are converted to control steps via ceiling
    division, which is conservative (slightly overestimates delays).

    Minimizes total starvation (number of (robot, step) pairs with no coverage).
    Returns a dict mapping control_step -> list of robot IDs to infer at that step.
    """
    T = time_to_step(params.end_time)
    R = params.num_robots
    B_max = len(params.d_infer) - 1
    tiers = sorted(params.d_infer.keys())

    # Convert ms delays to control steps
    # TODO: need to make sure order is correct
    d_send_steps = {robot_id: _delay_in_steps(d) for robot_id, d in params.d_send.items()}
    d_recv_steps = {robot_id: _delay_in_steps(d) for robot_id, d in params.d_recv.items()}
    d_infer_steps = {tier: _delay_in_steps(d) for tier, d in params.d_infer.items()}
    H = params.execution_horizon  # already in control steps

    prob = pulp.LpProblem("ActionChunkSchedulingBatched", pulp.LpMinimize)

    # x[t, r, b] = 1 if robot r is scheduled at step t in batch tier b
    x = {}
    for t in range(T):
        for r in params.robot_ids:
            for b in tiers:
                x[t, r, b] = pulp.LpVariable(f"x_{t}_{r}_{b}", cat=pulp.LpBinary)

    # y[t, b] = 1 if a batch of tier b starts at step t
    y = {}
    for t in range(T):
        for b in tiers:
            y[t, b] = pulp.LpVariable(f"y_{t}_{b}", cat=pulp.LpBinary)

    # s[t, r]: starvation slack (1 if robot r has no coverage at step t)
    s = {}
    for t in range(T):
        for r in params.robot_ids:
            s[t, r] = pulp.LpVariable(f"s_{t}_{r}", lowBound=0, upBound=1)

    # Objective: minimize total starvation
    prob += pulp.lpSum(s[t, r] for t in range(T) for r in params.robot_ids)

    # Coverage constraint: robot r covered at step t if some chunk covers it.
    # Chunk (tau, b) covers step t if:
    #   - Arrived: tau + d_infer_steps[b] + d_recv_steps[r] <= t
    #   - Still has actions: tau - d_send_steps[r] + H[r] - 1 >= t
    #     i.e. tau >= t - H[r] + 1 + d_send_steps[r]
    for t in range(T):
        for r in params.robot_ids:
            covering = []
            for b in tiers:
                lb = max(0, t - H[r] + 1 + d_send_steps[r])
                ub = min(T - 1, t - d_infer_steps[b] - d_recv_steps[r])
                for tau in range(lb, ub + 1):
                    covering.append(x[tau, r, b])
            prob += (
                pulp.lpSum(covering) + s[t, r] >= 1,
                f"coverage_{t}_{r}",
            )

    # Server exclusivity: at most one batch active at any step.
    # Batch (tau, b) occupies the server during [tau, tau + d_infer_steps[b] - 1].
    for t in range(T):
        active = []
        for b in tiers:
            lb = max(0, t - d_infer_steps[b] + 1)
            for tau in range(lb, t + 1):
                active.append(y[tau, b])
        prob += (
            pulp.lpSum(active) <= 1,
            f"server_{t}",
        )

    # Tier uniqueness: at most one tier chosen per step.
    for t in range(T):
        prob += (
            pulp.lpSum(y[t, b] for b in tiers) <= 1,
            f"tier_{t}",
        )

    # Batch consistency: robots in tier b at t cannot exceed b, and only if y[t,b]=1.
    for t in range(T):
        for b in tiers:
            prob += (
                pulp.lpSum(x[t, r, b] for r in params.robot_ids) <= b * y[t, b],
                f"batch_consistency_{t}_{b}",
            )

    # Linking: each robot assigned to at most one tier per step.
    for t in range(T):
        for r in params.robot_ids:
            prob += (
                pulp.lpSum(x[t, r, b] for b in tiers) <= 1,
                f"linking_{t}_{r}",
            )

    # Try Gurobi, fall back to CBC
    try:
        solver = pulp.GUROBI(msg=1, timeLimit=SOLVER_TIMEOUT)
        prob.solve(solver)
    except Exception as e:
        print(f"   Gurobi not available ({e}), falling back to CBC")
        prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=10))

    if prob.status != pulp.constants.LpStatusOptimal:
        print(f"Warning: ILP solver status: {pulp.LpStatus[prob.status]}")

    total_starvation = int(pulp.value(prob.objective))
    print(f"   ILP optimal starvation: {total_starvation} robot-steps")

    # Extract solution
    plan = {}
    for t in range(T):
        robots = [r for r in params.robot_ids if any(pulp.value(x[t, r, b]) > 0.5 for b in tiers)]
        if robots:
            plan[t] = robots

    return plan


def compute_lp_lower_bound(params: SimulatorParameters) -> float:
    """
    Compute a lower bound on optimal starvation via LP relaxation.

    This is the same formulation as solve_ilp_batched_exact but with
    continuous [0,1] variables instead of binary. Much faster to solve
    and gives a valid lower bound on the ILP optimum.

    Uses params.time_resolution to discretize the time axis (default 1ms).
    """
    dt = params.time_resolution
    T = params.num_time_slots
    R = params.num_robots
    B_max = len(params.d_infer) - 1
    tiers = range(1, B_max + 1)

    # Delays in slot units (ceiling division to stay conservative)
    d_infer_slots = [ceil(d / dt) for d in params.d_infer.values()]
    d_send_slots = [ceil(d / dt) for d in params.d_send.values()]
    d_recv_slots = [ceil(d / dt) for d in params.d_recv.values()]

    H = params.execution_horizon
    T_steps_per_robot = {r: params.num_steps_robot(r) for r in params.robot_ids}

    print(f"   LP: {T} time slots (resolution={dt}ms), {R} robots, max_batch={B_max}")

    prob = pulp.LpProblem("LPLowerBound", pulp.LpMinimize)

    # Continuous relaxation: x, y in [0, 1] instead of binary
    x = {}
    for t in range(T):
        for r in params.robot_ids:
            for b in tiers:
                x[t, r, b] = pulp.LpVariable(f"x_{t}_{r}_{b}", lowBound=0, upBound=1)

    y = {}
    for t in range(T):
        for b in tiers:
            y[t, b] = pulp.LpVariable(f"y_{t}_{b}", lowBound=0, upBound=1)

    s = {}
    for r in params.robot_ids:
        for step in range(T_steps_per_robot[r]):
            s[step, r] = pulp.LpVariable(f"s_{step}_{r}", lowBound=0, upBound=1)

    prob += pulp.lpSum(s[step, r] for r in params.robot_ids for step in range(T_steps_per_robot[r]))

    # Coverage: convert slot index back to ms for step lookups
    coverage = {r: [[] for _ in range(T_steps_per_robot[r])] for r in params.robot_ids}
    for tau in range(T):
        tau_ms = params.slot_to_time(tau)
        for r in params.robot_ids:
            obs_time = tau_ms - params.d_send[r]
            if obs_time < 0:
                continue
            obs_step = params.time_to_step_robot(obs_time, r)
            for b in tiers:
                arrival_time = tau_ms + params.d_infer[b] + params.d_recv[r]
                arrival_step = params.time_to_step_robot(arrival_time, r)
                last_step = obs_step + H[r] - 1
                lo = max(0, arrival_step)
                hi = min(T_steps_per_robot[r] - 1, last_step)
                for step in range(lo, hi + 1):
                    coverage[r][step].append(x[tau, r, b])

    for r in params.robot_ids:
        for step in range(T_steps_per_robot[r]):
            prob += (
                pulp.lpSum(coverage[r][step]) + s[step, r] >= 1,
                f"coverage_{step}_{r}",
            )

    # Server exclusivity (in slot units)
    for t in range(T):
        active = []
        for b in tiers:
            lb = max(0, t - d_infer_slots[b] + 1)
            for tau in range(lb, t + 1):
                active.append(y[tau, b])
        prob += (pulp.lpSum(active) <= 1, f"server_{t}")

    # Tier uniqueness
    for t in range(T):
        prob += (pulp.lpSum(y[t, b] for b in tiers) <= 1, f"tier_{t}")

    # Batch consistency
    for t in range(T):
        for b in tiers:
            prob += (
                pulp.lpSum(x[t, r, b] for r in params.robot_ids) <= b * y[t, b],
                f"batch_consistency_{t}_{b}",
            )

    # Linking
    for t in range(T):
        for r in params.robot_ids:
            prob += (
                pulp.lpSum(x[t, r, b] for b in tiers) <= 1,
                f"linking_{t}_{r}",
            )

    # Solve LP relaxation
    try:
        solver = pulp.GUROBI(msg=1)
        prob.solve(solver)
    except Exception as e:
        print(f"   Gurobi not available ({e}), falling back to CBC")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.constants.LpStatusOptimal:
        print(f"Warning: LP solver status: {pulp.LpStatus[prob.status]}")

    lb = pulp.value(prob.objective)
    print(f"   LP relaxation lower bound: {lb:.2f} robot-steps")
    return lb


def _greedy_warm_start(params: SimulatorParameters) -> list[tuple[int, list[str]]]:
    """Run GreedyScheduler and return [(start_time_ms, robot_indices), ...]."""
    from schedulers import GreedyScheduler
    from sim import Simulator

    sim = Simulator(params)
    scheduler = GreedyScheduler(params)
    sim.run(scheduler)
    print(f"   Greedy warm start: starvation={sim.calculate_starvation()}")
    return [(b.start_time, b.robot_ids) for b in sim.batch_history]


def solve_ilp_batched_exact(params: SimulatorParameters, warm_start: bool = False) -> dict[int, list[int]]:
    """
    Solve the optimal action chunk scheduling problem with configurable time resolution.

    Unlike solve_ilp_batched (which rounds delays to control steps), this
    formulation uses time-slot decision variables and exact floor-division
    to compute which control steps each chunk covers.

    Decision axis: time slots (ms / time_resolution)
    Objective axis: control steps (starvation measured per control step)

    Note: the coverage model assumes ideal action tracking (no starvation
    cascade), so the ILP's objective is a lower bound on true starvation.

    Returns a dict mapping ms start_time -> list of robot IDs.
    """
    dt = params.time_resolution
    T = params.num_time_slots
    R = params.num_robots
    B_max = len(params.d_infer) - 1
    tiers = sorted(params.d_infer.keys())

    # Delays in slot units (ceiling division to stay conservative)
    d_infer_slots = {tier: ceil(delay / dt) for tier, delay in params.d_infer.items()}

    H = params.execution_horizon  # already in control steps

    # Per-robot step counts (may differ with jittered step times)
    T_steps_per_robot = {r: params.num_steps_robot(r) for r in params.robot_ids}

    print(f"   ILP exact: {T} time slots (resolution={dt}ms), {R} robots, max_batch={B_max}")

    prob = pulp.LpProblem("ActionChunkSchedulingExact", pulp.LpMinimize)

    # x[t, r, b] = 1 if robot r is in a tier-b batch starting at slot t
    x = {}
    for t in range(T):
        for r in params.robot_ids:
            for b in tiers:
                x[t, r, b] = pulp.LpVariable(f"x_{t}_{r}_{b}", cat=pulp.LpBinary)

    # y[t, b] = 1 if a tier-b batch starts at slot t
    y = {}
    for t in range(T):
        for b in tiers:
            y[t, b] = pulp.LpVariable(f"y_{t}_{b}", cat=pulp.LpBinary)

    # s[step, r]: starvation (1 if robot r has no coverage at control step)
    s = {}
    for r in params.robot_ids:
        for step in range(T_steps_per_robot[r]):
            s[step, r] = pulp.LpVariable(f"s_{step}_{r}", lowBound=0, upBound=1)

    # Objective: minimize total starvation over control steps
    prob += pulp.lpSum(s[step, r] for r in params.robot_ids for step in range(T_steps_per_robot[r]))

    # Build coverage mapping: convert slot index back to ms for step lookups
    coverage = {r: [[] for _ in range(T_steps_per_robot[r])] for r in params.robot_ids}

    for tau in range(T):
        tau_ms = params.slot_to_time(tau)
        for r in params.robot_ids:
            obs_time = tau_ms - params.d_send[r]
            if obs_time < 0:
                continue
            obs_step = params.time_to_step_robot(obs_time, r)

            for b in tiers:
                arrival_time = tau_ms + params.d_infer[b] + params.d_recv[r]
                arrival_step = params.time_to_step_robot(arrival_time, r)
                last_step = obs_step + H[r] - 1

                lo = max(0, arrival_step)
                hi = min(T_steps_per_robot[r] - 1, last_step)
                for step in range(lo, hi + 1):
                    coverage[r][step].append(x[tau, r, b])

    # Coverage constraints
    for r in params.robot_ids:
        for step in range(T_steps_per_robot[r]):
            prob += (
                pulp.lpSum(coverage[r][step]) + s[step, r] >= 1,
                f"coverage_{step}_{r}",
            )

    # Server exclusivity: at most one batch active at any slot.
    # Batch (tau, b) occupies [tau, tau + d_infer_slots[b] - 1].
    for t in range(T):
        active = []
        for b in tiers:
            lb = max(0, t - d_infer_slots[b] + 1)
            for tau in range(lb, t + 1):
                active.append(y[tau, b])
        prob += (
            pulp.lpSum(active) <= 1,
            f"server_{t}",
        )

    # Tier uniqueness: at most one tier per slot.
    for t in range(T):
        prob += (
            pulp.lpSum(y[t, b] for b in tiers) <= 1,
            f"tier_{t}",
        )

    # Batch consistency: robots in tier b at t cannot exceed b.
    for t in range(T):
        for b in tiers:
            prob += (
                pulp.lpSum(x[t, r, b] for r in params.robot_ids) <= b * y[t, b],
                f"batch_consistency_{t}_{b}",
            )

    # Linking: each robot in at most one tier per slot.
    for t in range(T):
        for r in params.robot_ids:
            prob += (
                pulp.lpSum(x[t, r, b] for b in tiers) <= 1,
                f"linking_{t}_{r}",
            )

    # Warm start from greedy schedule
    if warm_start:
        greedy_batches = _greedy_warm_start(params)
        for t_start, robots in greedy_batches:
            slot = params.time_to_slot(t_start)
            b = len(robots)
            if slot < T and b in tiers:
                y[slot, b].setInitialValue(1)
                for r in robots:
                    x[slot, r, b].setInitialValue(1)

    # Solve
    try:
        solver = pulp.GUROBI(msg=1, timeLimit=SOLVER_TIMEOUT, warmStart=warm_start)
        prob.solve(solver)
    except Exception as e:
        print(f"   Gurobi not available ({e}), falling back to CBC")
        prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=10))

    if prob.status != pulp.constants.LpStatusOptimal:
        print(f"Warning: ILP solver status: {pulp.LpStatus[prob.status]}")

    total_starvation = int(pulp.value(prob.objective))
    print(f"   ILP exact optimal starvation: {total_starvation} robot-steps")

    # Extract schedule keyed by ms start time
    plan = {}
    for t in range(T):
        robots = [r for r in params.robot_ids if any(pulp.value(x[t, r, b]) > 0.5 for b in tiers)]
        if robots:
            plan[params.slot_to_time(t)] = robots

    return plan
