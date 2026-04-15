"""Solve for an optimal schedule offline using latency data from a previous run.

Usage:
    uv run scripts/solve_offline_schedule.py \
        --baseline-dir ./data/libero/baseline_run \
        --num-steps 100 \
        --solver batched \
        --output ./data/libero/optimal_plan.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import tyro

# Add simulator to path
_SIM_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "third_party" / "action-chunk-scheduling")
sys.path.insert(0, _SIM_DIR)

from classes import SimulatorParameters
from solve import compute_lp_lower_bound
from solve import solve_ilp_batched
from solve import solve_ilp_batched_exact


def load_latencies_from_server_metrics(
    baseline_dir: pathlib.Path,
) -> tuple[dict[str, float], dict[int, float], dict[str, float], dict[str, int], dict[str, list[float]], list[str]]:
    """Extract per-robot and per-batch latencies from server_metrics_history.json.

    All returned latencies are in seconds.

    Returns:
        (obs_latency, infer_latency, action_latency, execution_horizons, step_timestamps, robot_ids)
    """
    metrics_path = baseline_dir / "server_metrics_history.json"
    assert metrics_path.exists(), f"Missing {metrics_path}"
    metrics = json.loads(metrics_path.read_text())

    robots = metrics.get("robots", {})

    obs_latencies: dict[str, list[float]] = {}
    action_latencies: dict[str, list[float]] = {}
    execution_horizons: dict[str, int] = {}
    step_timestamps: dict[str, list[float]] = {}

    for robot_id, robot_data in robots.items():
        for episode in robot_data.get("episodes", []):
            for req in episode.get("requests", []):
                obs_latencies.setdefault(robot_id, []).append(req["server_arrival_time"] - req["request_timestamp"])
                execution_horizons[robot_id] = req.get("execution_horizon", 10)

            for resp in episode.get("responses", []):
                receive_time = resp.get("receive_time", 0)
                server_send_time = resp.get("server_send_time", 0)
                if receive_time > 0 and server_send_time > 0:
                    action_latencies.setdefault(robot_id, []).append(receive_time - server_send_time)

            if episode.get("step_timestamps"):
                step_timestamps[robot_id] = episode["step_timestamps"]

    robot_ids = sorted(robots.keys())

    obs_latency_avg = {rid: sum(v) / len(v) for rid, v in obs_latencies.items() if v}
    action_latency_avg = {rid: sum(v) / len(v) for rid, v in action_latencies.items() if v}

    # Per-batch-size inference latency from batch records.
    # BatchSummary serializes as positional list:
    # [batch_id, robot_ids, request_ids, inference_start_time, inference_end_time]
    raw_batches = metrics.get("batches", [])
    infer_by_size: dict[int, list[float]] = {}
    for b in raw_batches:
        if isinstance(b, dict):
            batch_size = len(b["robot_ids"])
            duration = b["inference_end_time"] - b["inference_start_time"]
        else:
            batch_size = len(b[1])
            duration = b[4] - b[3]
        infer_by_size.setdefault(batch_size, []).append(duration)

    infer_latency_avg = {k: sum(v) / len(v) for k, v in infer_by_size.items()}

    return obs_latency_avg, infer_latency_avg, action_latency_avg, execution_horizons, step_timestamps, robot_ids


def seconds_to_ms(s: float) -> int:
    """Convert seconds to milliseconds (ceiling, minimum 1)."""
    return max(1, round(s * 1000))


def build_simulator_parameters(
    baseline_dir: pathlib.Path,
    num_steps: int,
    control_hz: float,
) -> tuple[SimulatorParameters, dict[str, int], dict[int, str]]:
    """Build SimulatorParameters from a baseline run's metrics.

    SimulatorParameters uses milliseconds for all time values (d_send, d_recv,
    d_infer, end_time) and control steps for execution_horizon.
    """
    obs_lat, infer_lat, action_lat, exec_horizons, step_ts, robot_ids = load_latencies_from_server_metrics(baseline_dir)

    robot_id_to_index = {rid: i for i, rid in enumerate(robot_ids)}
    index_to_robot_id = {i: rid for rid, i in robot_id_to_index.items()}

    control_period_ms = round(1000 / control_hz)

    d_send = [seconds_to_ms(obs_lat.get(rid, 0.01)) for rid in robot_ids]
    d_recv = [seconds_to_ms(action_lat.get(rid, 0.01)) for rid in robot_ids]
    execution_horizon = [exec_horizons.get(rid, 10) for rid in robot_ids]

    # d_infer[0] = 0 (unused placeholder), then d_infer[b] for batch sizes 1..max
    max_batch_size = max(infer_lat.keys()) if infer_lat else 1
    d_infer = [0]
    for b in range(1, max_batch_size + 1):
        if b in infer_lat:
            d_infer.append(seconds_to_ms(infer_lat[b]))
        else:
            nearest = min(infer_lat.keys(), key=lambda k: abs(k - b))
            d_infer.append(seconds_to_ms(infer_lat[nearest]))

    end_time = num_steps * control_period_ms

    # Build per-robot control step times from real timestamps if available.
    # Convert absolute timestamps to relative ms from the earliest robot's first step.
    control_step_times: list[list[int]] = []
    if step_ts and all(rid in step_ts for rid in robot_ids):
        t0 = min(step_ts[rid][0] for rid in robot_ids)
        for rid in robot_ids:
            times_ms = [round((t - t0) * 1000) for t in step_ts[rid][:num_steps]]
            # Extend past end_time with uniform spacing so the sim has enough steps
            while not times_ms or times_ms[-1] < end_time + control_period_ms * 5:
                last = times_ms[-1] if times_ms else 0
                times_ms.append(last + control_period_ms)
            control_step_times.append(times_ms)

    params = SimulatorParameters(
        end_time=end_time,
        num_robots=len(robot_ids),
        d_infer=d_infer,
        d_send=d_send,
        d_recv=d_recv,
        execution_horizon=execution_horizon,
        control_step_times=control_step_times,
    )

    return params, robot_id_to_index, index_to_robot_id


def plan_to_replay_json(
    plan: dict[int, list[int]],
    index_to_robot_id: dict[int, str],
) -> dict:
    """Convert solver plan to JSON format for ReplayScheduler.

    The plan keys are time units (control steps or ms depending on solver).
    ReplayScheduler only uses robot_ids and ordering, not absolute timestamps.
    """
    batches = []
    batch_id = 0
    for step in sorted(plan.keys()):
        robot_indices = plan[step]
        robot_ids = [index_to_robot_id[i] for i in robot_indices]
        batches.append(
            {
                "batch_id": batch_id,
                "robot_ids": robot_ids,
                "request_ids": [-1] * len(robot_indices),
                "inference_start_time": 0.0,
                "inference_end_time": 0.0,
            }
        )
        batch_id += 1

    return {"batches": batches}


SOLVERS = {
    "batched": solve_ilp_batched,
    "exact": solve_ilp_batched_exact,
}


def simulate_and_plot(
    params: SimulatorParameters,
    plan: dict[int, list[int]],
    solver: str,
    output_dir: pathlib.Path,
) -> None:
    """Run the plan through the simulator and generate visualizations."""
    from schedulers import FixedScheduler
    from sim import Simulator
    from visualize import plot_actions_left
    from visualize import plot_gpu_timeline

    # Convert plan to ordered list of batches for FixedScheduler
    schedule = [plan[t] for t in sorted(plan.keys())]
    scheduler = FixedScheduler(params, schedule)
    sim = Simulator(params)
    sim.run(scheduler)

    starvation = sim.calculate_starvation()
    print(f"\nSimulated starvation: {starvation} robot-steps")

    output_dir.mkdir(parents=True, exist_ok=True)

    actions_left = sim.actions_left()
    plot_actions_left(
        actions_left,
        title=f"{solver} ILP: Actions Left (starvation={starvation})",
        filename=str(output_dir / "actions_left.png"),
    )
    plot_gpu_timeline(
        sim.batch_history,
        params.num_robots,
        params.end_time,
        title=f"{solver} ILP: GPU Timeline",
        filename=str(output_dir / "gpu_timeline.png"),
    )
    print(f"Plots saved to {output_dir}/")


def main(
    baseline_dir: pathlib.Path,
    output: pathlib.Path = pathlib.Path("optimal_plan.json"),
    num_steps: int = 100,
    control_hz: float = 20.0,
    solver: str = "batched",
    lp_bound: bool = False,
    plot: bool = False,
    time_resolution: int = 1,
) -> None:
    """Solve for an optimal offline schedule and write a replay file.

    Args:
        baseline_dir: Output directory from a previous run (must contain server_metrics_history.json).
        output: Path to write the replay JSON file.
        num_steps: Number of control steps to plan for.
        control_hz: Control frequency in Hz.
        solver: Which ILP solver to use (batched, exact).
        lp_bound: Also compute the LP relaxation lower bound.
        plot: Simulate the plan and generate visualization plots.
        time_resolution: Time discretization in ms for exact/LP solvers (default 1). Higher = faster but less precise.
    """
    assert solver in SOLVERS, f"Unknown solver {solver!r}, expected one of: {list(SOLVERS)}"

    print(f"Loading latencies from {baseline_dir}...")
    params, robot_id_to_index, index_to_robot_id = build_simulator_parameters(baseline_dir, num_steps, control_hz)
    params.time_resolution = time_resolution

    print("\nSimulator parameters:")
    print(f"  num_robots:         {params.num_robots}")
    print(f"  end_time:           {params.end_time} ms ({num_steps} steps)")
    print(f"  d_infer (ms):       {params.d_infer}")
    print(f"  d_send (ms):        {params.d_send}")
    print(f"  d_recv (ms):        {params.d_recv}")
    print(f"  execution_horizon:  {params.execution_horizon}")
    print(f"  time_resolution:    {params.time_resolution} ms")
    print(f"  robot map:          {robot_id_to_index}")

    if lp_bound:
        print("\nComputing LP lower bound...")
        compute_lp_lower_bound(params)

    print(f"\nSolving with {solver}...")
    solve_fn = SOLVERS[solver]
    plan = solve_fn(params)

    print(f"\nPlan has {len(plan)} scheduled batches")

    replay_data = plan_to_replay_json(plan, index_to_robot_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay_data, indent=2))
    print(f"Wrote replay file to {output}")

    if plot:
        simulate_and_plot(params, plan, solver, baseline_dir)


if __name__ == "__main__":
    tyro.cli(main)
