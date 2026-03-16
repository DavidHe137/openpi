import dataclasses
import queue
import time
from typing import Any

from openpi_client.messages import InferType

from openpi.scheduling import receding_horizon_ilp as rhilp
from openpi.serving.schemas import SlotRequest


@dataclasses.dataclass(frozen=True)
class SolveResult:
    start_tick: int
    horizon_end_tick: int
    boundary_tick: int
    batches_by_tick: dict[int, tuple[str, ...]]
    d_infer_tick: dict[int, int]
    d_send_tick: dict[str, int]
    d_recv_tick: dict[str, int]
    horizon_tick: dict[str, int]
    solve_ms: float
    success: bool
    error: str | None = None


def _make_request(
    *,
    robot_id: str,
    request_id: int,
    action_start_step: int = 0,
    min_execution_horizon: int = 0,
    control_hz: float = 20.0,
    deadline: float = 0.0,
    arrival_timestamp: float = 1.0,
    request_timestamp: float = 1.0,
) -> SlotRequest:
    return SlotRequest(
        slot_index=request_id,
        robot_id=robot_id,
        request_id=request_id,
        arrival_timestamp=arrival_timestamp,
        observation_step=action_start_step,
        action_start_step=action_start_step,
        request_timestamp=request_timestamp,
        deadline=deadline,
        min_execution_horizon=min_execution_horizon,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
        control_hz=control_hz,
    )


def _make_success_result(
    solve_input: Any,
    batches_by_tick: dict[int, tuple[str, ...]],
) -> SolveResult:
    return SolveResult(
        start_tick=solve_input.start_tick,
        horizon_end_tick=solve_input.start_tick + solve_input.horizon_steps,
        boundary_tick=solve_input.start_tick + solve_input.execute_steps,
        batches_by_tick=batches_by_tick,
        d_infer_tick=solve_input.d_infer_tick,
        d_send_tick=solve_input.d_send_tick,
        d_recv_tick=solve_input.d_recv_tick,
        horizon_tick=solve_input.horizon_tick,
        solve_ms=1.0,
        success=True,
    )


def test_discretization_uses_per_robot_control_hz(monkeypatch):
    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_validate_gurobi_available", staticmethod(lambda: None))

    captured_inputs: list[Any] = []

    def fake_solve(solve_input: Any) -> SolveResult:
        captured_inputs.append(solve_input)
        return _make_success_result(solve_input, {})

    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_solve_ilp", staticmethod(fake_solve))

    scheduler = rhilp.RecedingHorizonILPScheduler(
        queue.Queue(),
        max_batch_size=2,
        batch_profile={1: 15.0, 2: 23.0},
        tick_ms=10,
        action_horizon_steps=10,
    )
    try:
        scheduler.update(
            _make_request(
                robot_id="r1",
                request_id=1,
                control_hz=20.0,
                arrival_timestamp=1.035,
                request_timestamp=1.0,
            )
        )
        scheduler.update(_make_request(robot_id="r2", request_id=2, control_hz=10.0))

        scheduler.latency.update_action_delivery("r1", 1.021, 1.0)  # 21ms

        scheduler.schedule()  # kickoff solve
        time.sleep(0.02)
        scheduler.schedule()  # harvest completion

        assert captured_inputs
        solve_input = captured_inputs[0]
        assert solve_input.d_infer_tick[1] == 2
        assert solve_input.d_infer_tick[2] == 3
        assert solve_input.d_send_tick["r1"] == 4
        assert solve_input.d_recv_tick["r1"] == 3
        assert solve_input.horizon_tick["r1"] == 50
        assert solve_input.horizon_tick["r2"] == 100
    finally:
        scheduler.close()


def test_bootstrap_waits_for_first_plan(monkeypatch):
    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_validate_gurobi_available", staticmethod(lambda: None))

    def fake_solve(solve_input: Any) -> SolveResult:
        time.sleep(0.05)
        return _make_success_result(solve_input, {solve_input.start_tick: ("r1",)})

    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_solve_ilp", staticmethod(fake_solve))

    batch_queue: queue.Queue = queue.Queue()
    scheduler = rhilp.RecedingHorizonILPScheduler(
        batch_queue,
        max_batch_size=1,
        batch_profile={1: 10.0},
        tick_ms=10,
        horizon_steps=20,
    )
    try:
        scheduler.update(_make_request(robot_id="r1", request_id=1))

        scheduler.schedule()
        assert batch_queue.qsize() == 0

        time.sleep(0.07)
        scheduler.schedule()
        assert batch_queue.qsize() == 1

        samples = scheduler.flush_timing_samples()
        assert any(sample.metric_name == "bootstrap_wait_ms" for sample in samples)
    finally:
        scheduler.close()


def test_schedule_is_non_blocking_while_solving(monkeypatch):
    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_validate_gurobi_available", staticmethod(lambda: None))

    def slow_solve(solve_input: Any) -> SolveResult:
        time.sleep(0.12)
        return _make_success_result(solve_input, {solve_input.start_tick: ("r1",)})

    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_solve_ilp", staticmethod(slow_solve))

    scheduler = rhilp.RecedingHorizonILPScheduler(
        queue.Queue(),
        max_batch_size=1,
        batch_profile={1: 10.0},
    )
    try:
        scheduler.update(_make_request(robot_id="r1", request_id=1))

        t0 = time.perf_counter()
        scheduler.schedule()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 40.0
    finally:
        scheduler.close()


def test_swaps_to_pending_plan_at_boundary(monkeypatch):
    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_validate_gurobi_available", staticmethod(lambda: None))

    def fake_solve(solve_input: Any) -> SolveResult:
        if solve_input.start_tick == 0:
            return _make_success_result(solve_input, {0: ("r1",), 40: ("r1",), 80: ("r1",)})
        return _make_success_result(solve_input, {40: ("r2",), 80: ("r2",)})

    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_solve_ilp", staticmethod(fake_solve))

    batch_queue: queue.Queue = queue.Queue()
    scheduler = rhilp.RecedingHorizonILPScheduler(
        batch_queue,
        max_batch_size=1,
        batch_profile={1: 10.0},
    )
    try:
        tick = {"value": 0}
        monkeypatch.setattr(scheduler, "_now_tick", lambda: tick["value"])

        scheduler.update(_make_request(robot_id="r1", request_id=1))
        scheduler.update(_make_request(robot_id="r2", request_id=2))

        scheduler.schedule()  # kickoff first solve
        time.sleep(0.02)
        scheduler.schedule()  # activate first plan + dispatch tick 0
        _ = batch_queue.get_nowait()

        time.sleep(0.02)
        scheduler.schedule()  # harvest pending solve

        tick["value"] = 40
        scheduler.schedule()
        batch = batch_queue.get_nowait()
        assert [request.robot_id for request in batch] == ["r2"]
    finally:
        scheduler.close()


def test_uses_active_tail_when_pending_plan_is_late(monkeypatch):
    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_validate_gurobi_available", staticmethod(lambda: None))

    def fake_solve(solve_input: Any) -> SolveResult:
        if solve_input.start_tick == 0:
            return _make_success_result(solve_input, {0: ("r1",), 80: ("r1",)})
        time.sleep(0.2)
        return _make_success_result(solve_input, {40: ("r2",)})

    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_solve_ilp", staticmethod(fake_solve))

    batch_queue: queue.Queue = queue.Queue()
    scheduler = rhilp.RecedingHorizonILPScheduler(
        batch_queue,
        max_batch_size=1,
        batch_profile={1: 10.0},
    )
    try:
        tick = {"value": 0}
        monkeypatch.setattr(scheduler, "_now_tick", lambda: tick["value"])

        scheduler.update(_make_request(robot_id="r1", request_id=1))
        scheduler.update(_make_request(robot_id="r2", request_id=2))

        scheduler.schedule()  # kickoff first solve
        time.sleep(0.02)
        scheduler.schedule()  # activate + dispatch tick 0
        _ = batch_queue.get_nowait()

        scheduler.update(_make_request(robot_id="r1", request_id=3, action_start_step=1))
        tick["value"] = 80
        scheduler.schedule()
        batch = batch_queue.get_nowait()
        assert [request.robot_id for request in batch] == ["r1"]
    finally:
        scheduler.close()


def test_deadline_changes_do_not_change_solve_input(monkeypatch):
    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_validate_gurobi_available", staticmethod(lambda: None))

    captured_inputs_a: list[Any] = []
    captured_inputs_b: list[Any] = []

    def fake_solve_a(solve_input: Any) -> SolveResult:
        captured_inputs_a.append(solve_input)
        return _make_success_result(solve_input, {})

    def fake_solve_b(solve_input: Any) -> SolveResult:
        captured_inputs_b.append(solve_input)
        return _make_success_result(solve_input, {})

    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_solve_ilp", staticmethod(fake_solve_a))
    scheduler_a = rhilp.RecedingHorizonILPScheduler(queue.Queue(), max_batch_size=1, batch_profile={1: 10.0})
    scheduler_a.update(_make_request(robot_id="r1", request_id=1, deadline=1.0))
    scheduler_a.schedule()

    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_solve_ilp", staticmethod(fake_solve_b))
    scheduler_b = rhilp.RecedingHorizonILPScheduler(queue.Queue(), max_batch_size=1, batch_profile={1: 10.0})
    scheduler_b.update(_make_request(robot_id="r1", request_id=1, deadline=1_000_000.0))
    scheduler_b.schedule()

    try:
        assert captured_inputs_a
        assert captured_inputs_b
        a = captured_inputs_a[0]
        b = captured_inputs_b[0]

        assert a.robot_ids == b.robot_ids
        assert a.d_infer_tick == b.d_infer_tick
        assert a.d_send_tick == b.d_send_tick
        assert a.d_recv_tick == b.d_recv_tick
        assert a.horizon_tick == b.horizon_tick
        assert a.earliest_sched_tick == b.earliest_sched_tick
    finally:
        scheduler_a.close()
        scheduler_b.close()


def test_retry_after_solve_failure(monkeypatch):
    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_validate_gurobi_available", staticmethod(lambda: None))

    call_count = {"value": 0}

    def fake_solve(solve_input: Any) -> SolveResult:
        call_count["value"] += 1
        if call_count["value"] == 1:
            return SolveResult(
                start_tick=solve_input.start_tick,
                horizon_end_tick=solve_input.start_tick + solve_input.horizon_steps,
                boundary_tick=solve_input.start_tick + solve_input.execute_steps,
                batches_by_tick={},
                d_infer_tick=solve_input.d_infer_tick,
                d_send_tick=solve_input.d_send_tick,
                d_recv_tick=solve_input.d_recv_tick,
                horizon_tick=solve_input.horizon_tick,
                solve_ms=1.0,
                success=False,
                error="no feasible solution",
            )
        return _make_success_result(solve_input, {solve_input.start_tick: ("r1",)})

    monkeypatch.setattr(rhilp.RecedingHorizonILPScheduler, "_solve_ilp", staticmethod(fake_solve))

    batch_queue: queue.Queue = queue.Queue()
    scheduler = rhilp.RecedingHorizonILPScheduler(
        batch_queue,
        max_batch_size=1,
        batch_profile={1: 10.0},
    )
    try:
        tick = {"value": 0}
        monkeypatch.setattr(scheduler, "_now_tick", lambda: tick["value"])

        scheduler.update(_make_request(robot_id="r1", request_id=1))

        scheduler.schedule()  # first solve kickoff
        time.sleep(0.02)
        scheduler.schedule()  # process failure + retry kickoff
        time.sleep(0.02)
        scheduler.schedule()  # process second success + dispatch

        assert call_count["value"] >= 2
        assert batch_queue.qsize() == 1
    finally:
        scheduler.close()
