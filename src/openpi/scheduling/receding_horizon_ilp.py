from __future__ import annotations

from collections.abc import Collection
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import logging
import math
import multiprocessing as mp
import time
from typing import Any

from openpi.scheduling import RequestScheduler
from openpi.serving.schemas import SchedulerTimingSample
from openpi.serving.schemas import SlotRequest
from openpi.serving.schemas import WarmupSeed

try:
    import gurobipy as gp
except Exception:  # pragma: no cover - exercised in runtime setup paths
    gp = None


DEFAULT_CONTROL_HZ = 20.0

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _CommittedChunk:
    infer_tick: int
    d_infer_tick: int
    d_send_tick: int
    d_recv_tick: int
    horizon_tick: int


@dataclasses.dataclass(frozen=True)
class _SolveInput:
    start_tick: int
    horizon_steps: int
    execute_steps: int
    solve_timeout_s: float
    max_batch_size: int
    robot_ids: tuple[str, ...]
    d_infer_tick: dict[int, int]
    d_send_tick: dict[str, int]
    d_recv_tick: dict[str, int]
    horizon_tick: dict[str, int]
    earliest_sched_tick: dict[str, int]
    committed_chunks: dict[str, tuple[_CommittedChunk, ...]]


@dataclasses.dataclass(frozen=True)
class _SolveResult:
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
    status_code: int | None = None
    status_name: str | None = None
    sol_count: int | None = None
    timed_out: bool = False
    objective: float | None = None
    mip_gap: float | None = None


@dataclasses.dataclass
class _PlanState:
    start_tick: int
    boundary_tick: int
    horizon_end_tick: int
    batches_by_tick: dict[int, tuple[str, ...]]
    d_infer_tick: dict[int, int]
    d_send_tick: dict[str, int]
    d_recv_tick: dict[str, int]
    horizon_tick: dict[str, int]
    cursor_tick: int


class RecedingHorizonILPScheduler(RequestScheduler):
    """Asynchronous receding-horizon ILP scheduler with 10ms default discretization."""

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        batch_profile: dict[int, float] | None = None,
        *,
        tick_ms: int = 10,
        horizon_steps: int = 160,
        execution_fraction: float = 0.25,
        solve_timeout_ms: int = 500,
        action_horizon_steps: int = 10,
        default_control_hz: float = DEFAULT_CONTROL_HZ,
    ) -> None:
        super().__init__(batch_queue, max_batch_size, batch_profile)

        assert tick_ms >= 1, "tick_ms must be >= 1"
        assert horizon_steps >= 1, "horizon_steps must be >= 1"
        assert solve_timeout_ms >= 1, "solve_timeout_ms must be >= 1"
        assert 0 < execution_fraction <= 1, "execution_fraction must satisfy 0 < execution_fraction <= 1"
        assert action_horizon_steps >= 1, "action_horizon_steps must be >= 1"
        assert default_control_hz > 0, "default_control_hz must be > 0"

        self._validate_gurobi_available()

        self._tick_ms = tick_ms
        self._tick_s = tick_ms / 1000.0
        self._horizon_steps = horizon_steps
        self._execution_fraction = execution_fraction
        self._execute_steps = max(1, math.floor(horizon_steps * execution_fraction))
        self._solve_timeout_s = solve_timeout_ms / 1000.0
        self._action_horizon_steps = action_horizon_steps
        self._default_control_hz = default_control_hz

        self._epoch_monotonic = time.monotonic()
        self._bootstrap_start_monotonic = self._epoch_monotonic
        self._bootstrap_recorded = False
        self._bootstrap_robot_ids: set[str] = set()
        self._bootstrap_waiting_missing_ids: tuple[str, ...] | None = None

        self._active_plan: _PlanState | None = None
        self._pending_plan: _PlanState | None = None

        self._solve_future: Future[_SolveResult] | None = None
        self._solve_kickoff_monotonic: float | None = None
        self._solve_seq = 0
        self._inflight_solve_id: int | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="receding-ilp")

        self._server_available_tick = 0
        self._committed_chunks: dict[str, list[_CommittedChunk]] = {}

        logger.info(
            "RecedingHorizonILPScheduler configured: tick_ms=%d horizon_steps=%d "
            "execution_fraction=%.3f execute_steps=%d solve_timeout_ms=%d action_horizon_steps=%d",
            self._tick_ms,
            self._horizon_steps,
            self._execution_fraction,
            self._execute_steps,
            solve_timeout_ms,
            self._action_horizon_steps,
        )

    def update(self, request: SlotRequest) -> None:
        """Update latest request and latency; deadlines are intentionally ignored."""
        self._latest_requests[request.robot_id] = request
        self.latency.update_obs(request.robot_id, request.arrival_timestamp, request.request_timestamp)

    def update_warmup(self, warmup_seed: WarmupSeed) -> None:
        self._bootstrap_robot_ids.add(warmup_seed.robot_id)

    def schedule(self) -> None:
        if self._batch_queue.qsize() > 0:
            return

        now_tick = self._now_tick()
        with self.record_timing("schedule_decision"):
            candidate = self._compute_dispatch_candidate(now_tick)

        if candidate is None:
            return

        batch_size = len(candidate)
        annotated: list[SlotRequest] = []
        for request in candidate:
            self._latest_scheduled_requests[request.robot_id] = request
            d_ms = self.latency.total_delivery_ms(request.robot_id, batch_size)
            step_ms = 1000.0 / self._safe_control_hz(request.control_hz)
            d_steps = round(d_ms / step_ms) if d_ms is not None else 0
            annotated.append(dataclasses.replace(request, estimated_d_param=d_steps))

        self._batch_queue.put_nowait(annotated)
        self._register_dispatched_batch(now_tick, tuple(annotated))

    def get_next_batches(self) -> list[list[SlotRequest]]:
        """Unused because this scheduler overrides schedule() for custom lifecycle handling."""
        return []

    def reset_robot(self, robot_id: str) -> None:
        super().reset_robot(robot_id)
        self._committed_chunks.pop(robot_id, None)
        self._bootstrap_waiting_missing_ids = None

    def close(self) -> None:
        """Release solver worker resources."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _compute_dispatch_candidate(self, now_tick: int) -> tuple[SlotRequest, ...] | None:
        self._poll_solve_completion(now_tick)

        if self._active_plan is None:
            missing_bootstrap = self._missing_bootstrap_robot_ids()
            if missing_bootstrap:
                if self._bootstrap_waiting_missing_ids != missing_bootstrap:
                    logger.info(
                        "ILP bootstrap waiting for first request from %d/%d robots (missing=%s)",
                        len(missing_bootstrap),
                        len(self._bootstrap_robot_ids),
                        ",".join(missing_bootstrap),
                    )
                    self._bootstrap_waiting_missing_ids = missing_bootstrap
                return None
            self._bootstrap_waiting_missing_ids = None
            self._kickoff_solve(start_tick=now_tick, now_tick=now_tick)
            return None

        self._swap_pending_if_ready(now_tick)
        self._kickoff_next_receding_solve(now_tick)

        if now_tick < self._server_available_tick:
            return None

        schedulable = {request.robot_id: request for request in self._get_schedulable_requests()}
        if not schedulable:
            return None

        dispatch = self._pop_due_dispatch(self._active_plan, now_tick, set(schedulable))
        if dispatch is None:
            return None

        _, robot_ids = dispatch
        selected = tuple(schedulable[robot_id] for robot_id in robot_ids if robot_id in schedulable)
        return selected or None

    def _missing_bootstrap_robot_ids(self) -> tuple[str, ...]:
        if not self._bootstrap_robot_ids:
            return ()
        return tuple(
            sorted(robot_id for robot_id in self._bootstrap_robot_ids if robot_id not in self._latest_requests)
        )

    def _kickoff_next_receding_solve(self, now_tick: int) -> None:
        if self._active_plan is None:
            return
        if self._pending_plan is not None or self._solve_future is not None:
            return

        if self._active_plan.cursor_tick >= self._active_plan.horizon_end_tick:
            target = now_tick
        else:
            target = max(now_tick, self._active_plan.boundary_tick)
        self._kickoff_solve(start_tick=target, now_tick=now_tick)

    def _kickoff_solve(self, *, start_tick: int, now_tick: int) -> None:
        if self._solve_future is not None:
            return

        solve_input = self._build_solve_input(start_tick=start_tick, now_tick=now_tick)
        if solve_input is None:
            return

        self._record_metric("ilp_replan_kickoff", 0.0)
        self._solve_seq += 1
        solve_id = self._solve_seq
        d_infer_vals = tuple(solve_input.d_infer_tick.values())
        d_send_vals = tuple(solve_input.d_send_tick.values())
        d_recv_vals = tuple(solve_input.d_recv_tick.values())
        horizon_vals = tuple(solve_input.horizon_tick.values())
        impossible_robots = 0
        for robot_id in solve_input.robot_ids:
            d_send = solve_input.d_send_tick[robot_id]
            d_recv = solve_input.d_recv_tick[robot_id]
            horizon = solve_input.horizon_tick[robot_id]
            has_feasible_tier = any(
                d_inf + d_recv <= (horizon - 1 - d_send) for d_inf in solve_input.d_infer_tick.values()
            )
            if not has_feasible_tier:
                impossible_robots += 1

        logger.info(
            "ILP solve start: solve_id=%d start_tick=%d now_tick=%d horizon_steps=%d execute_steps=%d robots=%d",
            solve_id,
            start_tick,
            now_tick,
            solve_input.horizon_steps,
            solve_input.execute_steps,
            len(solve_input.robot_ids),
        )
        logger.info(
            "ILP discretization: solve_id=%d d_infer_tick[min,max]=[%d,%d] "
            "d_send_tick[min,max]=[%d,%d] d_recv_tick[min,max]=[%d,%d] "
            "chunk_horizon_tick[min,max]=[%d,%d] impossible_robots=%d/%d",
            solve_id,
            min(d_infer_vals),
            max(d_infer_vals),
            min(d_send_vals),
            max(d_send_vals),
            min(d_recv_vals),
            max(d_recv_vals),
            min(horizon_vals),
            max(horizon_vals),
            impossible_robots,
            len(solve_input.robot_ids),
        )
        self._solve_kickoff_monotonic = time.monotonic()
        self._inflight_solve_id = solve_id
        self._solve_future = self._executor.submit(self._solve_ilp, solve_input)

    def _build_solve_input(self, *, start_tick: int, now_tick: int) -> _SolveInput | None:
        if not self._latest_requests:
            return None

        robot_ids = tuple(sorted(self._latest_requests))

        d_infer_tick = {batch_size: self._infer_ticks(batch_size) for batch_size in range(1, self._max_batch_size + 1)}
        d_send_tick = {robot_id: self._send_ticks(robot_id) for robot_id in robot_ids}
        d_recv_tick = {robot_id: self._recv_ticks(robot_id) for robot_id in robot_ids}
        horizon_tick = {
            robot_id: self._chunk_horizon_ticks(self._latest_requests[robot_id].control_hz) for robot_id in robot_ids
        }

        earliest_sched_tick = {robot_id: self._earliest_sched_tick(robot_id, start_tick) for robot_id in robot_ids}

        committed_copy = {robot_id: list(chunks) for robot_id, chunks in self._committed_chunks.items()}

        if self._active_plan is not None and start_tick == self._active_plan.boundary_tick:
            for tick in range(self._active_plan.cursor_tick, self._active_plan.boundary_tick):
                batch = self._active_plan.batches_by_tick.get(tick)
                if not batch:
                    continue
                batch_size = len(batch)
                d_infer = self._active_plan.d_infer_tick.get(batch_size)
                if d_infer is None:
                    continue
                for robot_id in batch:
                    committed_copy.setdefault(robot_id, []).append(
                        _CommittedChunk(
                            infer_tick=tick,
                            d_infer_tick=d_infer,
                            d_send_tick=self._active_plan.d_send_tick.get(robot_id, 0),
                            d_recv_tick=self._active_plan.d_recv_tick.get(robot_id, 0),
                            horizon_tick=self._active_plan.horizon_tick.get(robot_id, 1),
                        )
                    )

        return _SolveInput(
            start_tick=start_tick,
            horizon_steps=self._horizon_steps,
            execute_steps=self._execute_steps,
            solve_timeout_s=self._solve_timeout_s,
            max_batch_size=self._max_batch_size,
            robot_ids=robot_ids,
            d_infer_tick=d_infer_tick,
            d_send_tick=d_send_tick,
            d_recv_tick=d_recv_tick,
            horizon_tick=horizon_tick,
            earliest_sched_tick=earliest_sched_tick,
            committed_chunks={robot_id: tuple(chunks) for robot_id, chunks in committed_copy.items()},
        )

    def _poll_solve_completion(self, now_tick: int) -> None:
        if self._solve_future is None or not self._solve_future.done():
            return

        future = self._solve_future
        self._solve_future = None
        solve_id = self._inflight_solve_id
        kickoff_to_ready_ms = self._kickoff_to_ready_ms()
        self._inflight_solve_id = None

        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("ILP solve crashed: solve_id=%s", solve_id)
            self._record_metric("ilp_solve_ms", 0.0)
            self._record_metric("plan_kickoff_to_ready_ms", kickoff_to_ready_ms)
            self._record_metric("ilp_solve_error", 0.0)
            self._solve_kickoff_monotonic = None
            result = _SolveResult(
                start_tick=now_tick,
                horizon_end_tick=now_tick,
                boundary_tick=now_tick,
                batches_by_tick={},
                d_infer_tick={},
                d_send_tick={},
                d_recv_tick={},
                horizon_tick={},
                solve_ms=0.0,
                success=False,
                error=str(exc),
                status_name="exception",
            )

        self._record_metric("ilp_solve_ms", result.solve_ms)
        self._record_metric("plan_kickoff_to_ready_ms", kickoff_to_ready_ms)
        self._solve_kickoff_monotonic = None
        status_name = result.status_name
        status_code = result.status_code
        sol_count = result.sol_count
        timed_out = result.timed_out
        objective = result.objective
        mip_gap = result.mip_gap

        if not result.success:
            logger.warning(
                "ILP solve failed: solve_id=%s start_tick=%d boundary_tick=%d solve_ms=%.2f "
                "kickoff_to_ready_ms=%.2f status=%s(%s) sol_count=%s error=%s",
                solve_id,
                result.start_tick,
                result.boundary_tick,
                result.solve_ms,
                kickoff_to_ready_ms,
                status_name,
                status_code,
                sol_count,
                result.error,
            )
            return

        log_fn = logger.warning if timed_out else logger.info
        log_fn(
            "ILP solve done: solve_id=%s start_tick=%d boundary_tick=%d horizon_end_tick=%d "
            "solve_ms=%.2f kickoff_to_ready_ms=%.2f planned_batches=%d "
            "status=%s(%s) timed_out=%s sol_count=%s objective=%s mip_gap=%s",
            solve_id,
            result.start_tick,
            result.boundary_tick,
            result.horizon_end_tick,
            result.solve_ms,
            kickoff_to_ready_ms,
            len(result.batches_by_tick),
            status_name,
            status_code,
            timed_out,
            sol_count,
            objective,
            mip_gap,
        )

        new_plan = _PlanState(
            start_tick=result.start_tick,
            boundary_tick=result.boundary_tick,
            horizon_end_tick=result.horizon_end_tick,
            batches_by_tick=result.batches_by_tick,
            d_infer_tick=result.d_infer_tick,
            d_send_tick=result.d_send_tick,
            d_recv_tick=result.d_recv_tick,
            horizon_tick=result.horizon_tick,
            cursor_tick=result.start_tick,
        )

        if self._active_plan is None:
            self._active_plan = new_plan
            self._pending_plan = None
            self._record_metric("ilp_plan_activated", 0.0)
            if not self._bootstrap_recorded:
                bootstrap_wait_ms = (time.monotonic() - self._bootstrap_start_monotonic) * 1000.0
                self._record_metric("bootstrap_wait_ms", bootstrap_wait_ms)
                self._bootstrap_recorded = True
            return

        if result.start_tick < self._active_plan.boundary_tick:
            return
        self._pending_plan = new_plan

    def _swap_pending_if_ready(self, now_tick: int) -> None:
        if self._active_plan is None or self._pending_plan is None:
            return
        if now_tick < self._active_plan.boundary_tick:
            return
        # Do not cut over until the committed prefix has actually been consumed.
        # This is critical when execution_fraction == 1.0, where boundary == horizon end.
        if self._active_plan.cursor_tick < self._active_plan.boundary_tick:
            return
        self._active_plan = self._pending_plan
        self._pending_plan = None
        self._record_metric("ilp_plan_activated", 0.0)

    def _register_dispatched_batch(self, now_tick: int, requests: tuple[SlotRequest, ...]) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return

        d_infer_tick = self._infer_ticks(batch_size)
        self._server_available_tick = max(self._server_available_tick, now_tick + d_infer_tick)

        for request in requests:
            robot_id = request.robot_id
            self._committed_chunks.setdefault(robot_id, []).append(
                _CommittedChunk(
                    infer_tick=now_tick,
                    d_infer_tick=d_infer_tick,
                    d_send_tick=self._send_ticks(robot_id),
                    d_recv_tick=self._recv_ticks(robot_id),
                    horizon_tick=self._chunk_horizon_ticks(request.control_hz),
                )
            )

    def _infer_ms(self, batch_size: int) -> float:
        value = self.latency.infer_ms(batch_size)
        if value is None:
            value = self._batch_profile_ms.get(batch_size)
        if value is None:
            raise RuntimeError(
                f"Missing inference latency estimate for batch_size={batch_size}. "
                "Warmup profile and runtime EMA are both unavailable."
            )
        return value

    def _infer_ticks(self, batch_size: int) -> int:
        return self._to_positive_ticks(self._infer_ms(batch_size))

    def _send_ticks(self, robot_id: str) -> int:
        value = self.latency.obs_network_ms(robot_id)
        if value is None:
            return 0
        return self._to_nonnegative_ticks(max(0.0, value))

    def _recv_ticks(self, robot_id: str) -> int:
        value = self.latency.action_delivery_ms(robot_id)
        if value is None:
            return 0
        return self._to_nonnegative_ticks(max(0.0, value))

    def _chunk_horizon_ticks(self, control_hz: float) -> int:
        hz = self._safe_control_hz(control_hz)
        duration_ms = (self._action_horizon_steps * 1000.0) / hz
        return self._to_positive_ticks(duration_ms)

    def _earliest_sched_tick(self, robot_id: str, start_tick: int) -> int:
        request = self._latest_requests[robot_id]
        last = self._latest_scheduled_requests.get(robot_id)
        if last is None:
            return start_tick

        required_action_step = last.action_start_step + last.min_execution_horizon
        remaining_steps = required_action_step - request.action_start_step
        if remaining_steps <= 0:
            return start_tick

        wait_ms = remaining_steps * (1000.0 / self._safe_control_hz(request.control_hz))
        return start_tick + self._to_nonnegative_ticks(wait_ms)

    def _safe_control_hz(self, control_hz: float) -> float:
        return control_hz if control_hz > 0 else self._default_control_hz

    def _now_tick(self) -> int:
        return math.floor((time.monotonic() - self._epoch_monotonic) / self._tick_s)

    def _to_positive_ticks(self, duration_ms: float) -> int:
        return max(1, math.ceil(duration_ms / self._tick_ms))

    def _to_nonnegative_ticks(self, duration_ms: float) -> int:
        return max(0, math.ceil(duration_ms / self._tick_ms))

    def _kickoff_to_ready_ms(self) -> float:
        if self._solve_kickoff_monotonic is None:
            return 0.0
        return (time.monotonic() - self._solve_kickoff_monotonic) * 1000.0

    def _record_metric(self, metric_name: str, duration_ms: float) -> None:
        self._timing_samples.append(
            SchedulerTimingSample(
                scheduler_name=self.__class__.__name__,
                metric_name=metric_name,
                duration_ms=duration_ms,
                recorded_at=time.time(),
            )
        )

    @staticmethod
    def _chunk_valid_at(chunk: _CommittedChunk, tick: int) -> bool:
        arrival = chunk.infer_tick + chunk.d_infer_tick + chunk.d_recv_tick
        valid_until = chunk.infer_tick - chunk.d_send_tick + chunk.horizon_tick
        return arrival <= tick < valid_until

    @staticmethod
    def _pop_due_dispatch(
        plan: _PlanState,
        now_tick: int,
        schedulable_robot_ids: Collection[str],
    ) -> tuple[int, tuple[str, ...]] | None:
        while plan.cursor_tick < plan.horizon_end_tick:
            tick = plan.cursor_tick
            if tick > now_tick:
                return None

            plan.cursor_tick += 1
            batch = plan.batches_by_tick.get(tick)
            if not batch:
                continue

            selected = tuple(robot_id for robot_id in batch if robot_id in schedulable_robot_ids)
            if selected:
                return tick, selected
        return None

    @staticmethod
    def _validate_gurobi_available() -> None:
        if gp is None:
            raise RuntimeError("receding_horizon_ilp requires gurobipy, but it is not importable in this environment.")
        try:
            model = gp.Model("receding_horizon_ilp_probe")
            model.Params.OutputFlag = 0
            model.dispose()
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("receding_horizon_ilp requires a working Gurobi installation/license.") from exc

    @staticmethod
    def _gurobi_status_name(status: int) -> str:
        status_names = {
            gp.GRB.LOADED: "LOADED",
            gp.GRB.OPTIMAL: "OPTIMAL",
            gp.GRB.INFEASIBLE: "INFEASIBLE",
            gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
            gp.GRB.UNBOUNDED: "UNBOUNDED",
            gp.GRB.CUTOFF: "CUTOFF",
            gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            gp.GRB.NODE_LIMIT: "NODE_LIMIT",
            gp.GRB.TIME_LIMIT: "TIME_LIMIT",
            gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            gp.GRB.INTERRUPTED: "INTERRUPTED",
            gp.GRB.NUMERIC: "NUMERIC",
            gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
            gp.GRB.INPROGRESS: "INPROGRESS",
            gp.GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
            gp.GRB.WORK_LIMIT: "WORK_LIMIT",
            gp.GRB.MEM_LIMIT: "MEM_LIMIT",
        }
        return status_names.get(status, f"UNKNOWN_{status}")

    @classmethod
    def _solve_ilp(cls, solve_input: _SolveInput) -> _SolveResult:
        t0 = time.perf_counter()
        start_tick = solve_input.start_tick
        horizon_end_tick = start_tick + solve_input.horizon_steps
        boundary_tick = min(horizon_end_tick, start_tick + solve_input.execute_steps)
        common = {
            "start_tick": start_tick,
            "horizon_end_tick": horizon_end_tick,
            "boundary_tick": boundary_tick,
            "d_infer_tick": solve_input.d_infer_tick,
            "d_send_tick": solve_input.d_send_tick,
            "d_recv_tick": solve_input.d_recv_tick,
            "horizon_tick": solve_input.horizon_tick,
        }

        if not solve_input.robot_ids:
            return _SolveResult(
                **common,
                batches_by_tick={},
                solve_ms=(time.perf_counter() - t0) * 1000.0,
                success=True,
                status_name="EMPTY_ROBOT_SET",
                status_code=0,
                sol_count=0,
            )

        model: Any | None = None
        try:
            model = gp.Model("RecedingHorizonILP")
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = solve_input.solve_timeout_s

            tiers = range(1, solve_input.max_batch_size + 1)

            x: dict[tuple[int, str, int], Any] = {}
            y: dict[tuple[int, int], Any] = {}
            s: dict[tuple[int, str], Any] = {}

            for tick in range(start_tick, horizon_end_tick):
                for robot_id in solve_input.robot_ids:
                    covered = any(
                        cls._chunk_valid_at(chunk, tick) for chunk in solve_input.committed_chunks.get(robot_id, ())
                    )
                    if not covered:
                        s[tick, robot_id] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0)

                    for tier in tiers:
                        x[tick, robot_id, tier] = model.addVar(vtype=gp.GRB.BINARY)

                for tier in tiers:
                    y[tick, tier] = model.addVar(vtype=gp.GRB.BINARY)

            model.update()
            model.setObjective(gp.quicksum(s.values()), gp.GRB.MINIMIZE)

            for tick in range(start_tick, horizon_end_tick):
                for robot_id in solve_input.robot_ids:
                    earliest = solve_input.earliest_sched_tick.get(robot_id, start_tick)
                    if tick < earliest:
                        for tier in tiers:
                            model.addConstr(x[tick, robot_id, tier] == 0)

                    covered = any(
                        cls._chunk_valid_at(chunk, tick) for chunk in solve_input.committed_chunks.get(robot_id, ())
                    )
                    if covered:
                        continue

                    terms: list[Any] = []
                    d_send = solve_input.d_send_tick[robot_id]
                    d_recv = solve_input.d_recv_tick[robot_id]
                    horizon = solve_input.horizon_tick[robot_id]
                    for tier in tiers:
                        d_infer = solve_input.d_infer_tick[tier]
                        lower = max(start_tick, tick - (horizon - 1 - d_send))
                        upper = min(horizon_end_tick - 1, tick - d_infer - d_recv)
                        if lower > upper:
                            continue
                        terms.extend(x[tau, robot_id, tier] for tau in range(lower, upper + 1))

                    model.addConstr(gp.quicksum(terms) + s[tick, robot_id] >= 1)

            for tick in range(start_tick, horizon_end_tick):
                active_terms: list[Any] = []
                for tier in tiers:
                    d_infer = solve_input.d_infer_tick[tier]
                    lower = max(start_tick, tick - d_infer + 1)
                    active_terms.extend(y[tau, tier] for tau in range(lower, tick + 1))
                model.addConstr(gp.quicksum(active_terms) <= 1)

            for tick in range(start_tick, horizon_end_tick):
                model.addConstr(gp.quicksum(y[tick, tier] for tier in tiers) <= 1)

            for tick in range(start_tick, horizon_end_tick):
                for tier in tiers:
                    model.addConstr(
                        gp.quicksum(x[tick, robot_id, tier] for robot_id in solve_input.robot_ids)
                        <= tier * y[tick, tier]
                    )

            for tick in range(start_tick, horizon_end_tick):
                for robot_id in solve_input.robot_ids:
                    model.addConstr(gp.quicksum(x[tick, robot_id, tier] for tier in tiers) <= 1)

            model.optimize()

            solve_ms = (time.perf_counter() - t0) * 1000.0
            status_code = int(model.Status)
            status_name = cls._gurobi_status_name(status_code)
            sol_count = int(model.SolCount)
            timed_out = status_code == gp.GRB.TIME_LIMIT

            objective: float | None = None
            if sol_count > 0:
                objective = float(model.ObjVal)

            mip_gap: float | None = None
            try:
                if sol_count > 0:
                    mip_gap = float(model.MIPGap)
            except Exception:
                mip_gap = None

            if sol_count < 1:
                return _SolveResult(
                    **common,
                    batches_by_tick={},
                    solve_ms=solve_ms,
                    success=False,
                    error=f"gurobi status={status_name}({status_code}), no feasible solution",
                    status_code=status_code,
                    status_name=status_name,
                    sol_count=sol_count,
                    timed_out=timed_out,
                    objective=objective,
                    mip_gap=mip_gap,
                )

            batches_by_tick: dict[int, tuple[str, ...]] = {}
            for tick in range(start_tick, horizon_end_tick):
                scheduled = [
                    robot_id
                    for robot_id in solve_input.robot_ids
                    if any(x[tick, robot_id, tier].X > 0.5 for tier in tiers)
                ]
                if scheduled:
                    batches_by_tick[tick] = tuple(scheduled)

            return _SolveResult(
                **common,
                batches_by_tick=batches_by_tick,
                solve_ms=solve_ms,
                success=True,
                status_code=status_code,
                status_name=status_name,
                sol_count=sol_count,
                timed_out=timed_out,
                objective=objective,
                mip_gap=mip_gap,
            )

        except Exception as exc:
            return _SolveResult(
                **common,
                batches_by_tick={},
                solve_ms=(time.perf_counter() - t0) * 1000.0,
                success=False,
                error=str(exc),
                status_name="exception",
            )
        finally:
            if model is not None:
                model.dispose()
