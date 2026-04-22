"""Parity tests for the single-process sim harness.

These tests drive the real ``ActionChunkBroker`` and real ``RequestScheduler``
subclasses through ``SimRuntime`` and assert that the resulting trace matches
what can be derived from first principles (known ``d_net_s``, per-batch-size
inference latency, and observation cadence).

The sim is the ground truth — tests compare broker/scheduler state against
hand-computed reference values rather than against a separate shadow model.
"""

from __future__ import annotations

import math
import queue

import pytest
from openpi_client.schemas import Action

from openpi.scheduling.baselines import FixedSizeGreedyScheduler
from openpi.simulation.runtime.runtime import RobotTraceEntry
from openpi.simulation.runtime.runtime import SimRuntime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_runtime(
    *,
    max_batch_size: int = 1,
    latency_s_by_batch_size: dict[int, float] | None = None,
    control_hz: int = 20,
    action_horizon: int = 10,
    action_dim: int = 7,
    execution_horizon: int = 5,
    d_net_s: float = 0.0,
) -> SimRuntime:
    if latency_s_by_batch_size is None:
        latency_s_by_batch_size = {bs: 0.05 for bs in range(1, max_batch_size + 1)}
    sched = FixedSizeGreedyScheduler(queue.Queue(), max_batch_size=max_batch_size)
    return SimRuntime(
        scheduler=sched,
        latency_s_by_batch_size=latency_s_by_batch_size,
        control_hz=control_hz,
        action_horizon=action_horizon,
        action_dim=action_dim,
        execution_horizon=execution_horizon,
        d_net_s=d_net_s,
    )


def action_fingerprint(action: Action) -> tuple:
    """Collapse an Action into a comparable tuple for determinism checks."""
    return (
        action.step,
        action.action_chunk_index,
        action.index_in_chunk,
        float(action.action[0]),
        float(action.action[-1]),
    )


def trace_fingerprint(trace: list[RobotTraceEntry]) -> list[tuple]:
    return [(e.step, e.sim_time_s, action_fingerprint(e.action)) for e in trace]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_identical_runs_produce_identical_traces(self):
        def run():
            rt = build_runtime(
                latency_s_by_batch_size={1: 0.05},
                execution_horizon=5,
            )
            rt.add_robot("r0")
            rt.schedule_robot("r0", num_steps=30)
            rt.run_until(2.0)
            return trace_fingerprint(rt.trace("r0"))

        assert run() == run()

    def test_two_robots_identical_runs(self):
        def run():
            rt = build_runtime(
                max_batch_size=1,
                latency_s_by_batch_size={1: 0.05},
            )
            rt.add_robot("r0")
            rt.add_robot("r1")
            rt.schedule_robot("r0", num_steps=20)
            rt.schedule_robot("r1", num_steps=20, start_offset_s=0.025)
            rt.run_until(2.0)
            return (trace_fingerprint(rt.trace("r0")), trace_fingerprint(rt.trace("r1")))

        assert run() == run()


# ---------------------------------------------------------------------------
# Single-robot, zero-latency golden trace
# ---------------------------------------------------------------------------


class TestZeroLatencyGolden:
    """With d_net=0 and d_infer=0, an inference launched at step N completes
    before step N+1. The broker always has a fresh chunk ready, so step N+1
    consumes a chunk produced from observation N — i.e. ``a0 == N``.
    """

    def test_each_step_uses_previous_observation_chunk(self):
        rt = build_runtime(
            latency_s_by_batch_size={1: 0.0},
            execution_horizon=5,
            d_net_s=0.0,
        )
        rt.add_robot("r0")
        rt.schedule_robot("r0", num_steps=12)
        rt.run_until(1.0)

        trace = rt.trace("r0")
        assert trace[0].action.action_chunk_index is None
        assert trace[0].action.index_in_chunk is None
        for entry in trace[1:]:
            assert entry.action.action_chunk_index is not None, f"expected chunk at step {entry.step}"
            assert entry.action.index_in_chunk == 0
            assert float(entry.action.action[0]) == pytest.approx(entry.step - 1)


# ---------------------------------------------------------------------------
# Null-prefix: how many null actions before the first real action
# ---------------------------------------------------------------------------


class TestNullPrefix:
    """First chunk can be consumed only after a full round-trip:

        send obs_0 at t = 0
        response arrives at t = d_net + d_infer + d_net

    A step at time ``t_i = i/hz`` is null iff ``t_i <= roundtrip`` — equality
    goes null because the event loop fires the step's ``broker.infer`` (scheduled
    up-front) before any callback produced later at the same sim time. So
    ``null_prefix = floor(roundtrip * hz) + 1``.
    """

    @pytest.mark.parametrize(
        ("d_infer", "d_net", "hz"),
        [
            (0.05, 0.0, 20),  # rt=0.050 → 2 null
            (0.10, 0.0, 20),  # rt=0.100 → 3 null (exact multiple, tie → null)
            (0.05, 0.01, 20),  # rt=0.070 → 2 null
            (0.025, 0.0, 20),  # rt=0.025 (<1 step) → 1 null
        ],
    )
    def test_null_prefix_length_matches_roundtrip(self, d_infer: float, d_net: float, hz: int):
        rt = build_runtime(
            latency_s_by_batch_size={1: d_infer},
            control_hz=hz,
            execution_horizon=5,
            d_net_s=d_net,
        )
        rt.add_robot("r0")
        rt.schedule_robot("r0", num_steps=20)
        rt.run_until(2.0)

        trace = rt.trace("r0")
        null_prefix = 0
        for entry in trace:
            if entry.action.action_chunk_index is None:
                null_prefix += 1
            else:
                break

        roundtrip = 2 * d_net + d_infer
        expected = math.floor(roundtrip * hz) + 1
        assert null_prefix == expected, (
            f"null_prefix={null_prefix} expected={expected} "
            f"(d_net={d_net}, d_infer={d_infer}, hz={hz})"
        )


# ---------------------------------------------------------------------------
# Chunk provenance: action[0] encodes the observation_step of the chunk
# ---------------------------------------------------------------------------


class TestChunkProvenance:
    """SimGPU fills ``actions[:, :] = observation_step`` per request, so every
    real action emitted by the broker should have ``action[0] ==
    chunk.observation_step`` — a direct check that the broker's chunk
    bookkeeping matches the request that produced it.
    """

    def test_action_matches_chunk_observation_step(self):
        rt = build_runtime(
            latency_s_by_batch_size={1: 0.05},
            execution_horizon=5,
        )
        broker = rt.add_robot("r0")
        rt.schedule_robot("r0", num_steps=30)
        rt.run_until(2.0)

        for entry in rt.trace("r0"):
            if entry.action.action_chunk_index is None:
                continue
            chunk = broker.action_chunks[entry.action.action_chunk_index]
            assert float(entry.action.action[0]) == pytest.approx(chunk.observation_step)


# ---------------------------------------------------------------------------
# Two-robot EDF with max_batch_size=1
# ---------------------------------------------------------------------------


class TestTwoRobotEDF:
    def test_both_robots_make_progress(self):
        rt = build_runtime(
            max_batch_size=1,
            latency_s_by_batch_size={1: 0.05},
            execution_horizon=5,
        )
        rt.add_robot("r0")
        rt.add_robot("r1")
        rt.schedule_robot("r0", num_steps=12)
        rt.schedule_robot("r1", num_steps=12, start_offset_s=0.025)
        rt.run_until(2.0)

        trace_r0 = rt.trace("r0")
        trace_r1 = rt.trace("r1")
        assert any(e.action.action_chunk_index is not None for e in trace_r0)
        assert any(e.action.action_chunk_index is not None for e in trace_r1)

    def test_requests_are_served_from_both_robots(self):
        """FixedSizeGreedy with batch=1 should alternate when both robots have
        pending requests — neither robot should be permanently starved.
        """
        rt = build_runtime(
            max_batch_size=1,
            latency_s_by_batch_size={1: 0.05},
            execution_horizon=5,
        )
        broker_r0 = rt.add_robot("r0")
        broker_r1 = rt.add_robot("r1")
        rt.schedule_robot("r0", num_steps=20)
        rt.schedule_robot("r1", num_steps=20, start_offset_s=0.025)
        rt.run_until(2.0)

        # Both brokers should have received multiple chunks.
        assert len(broker_r0.action_chunks) >= 3
        assert len(broker_r1.action_chunks) >= 3


# ---------------------------------------------------------------------------
# Scheduler state tracks broker activity
# ---------------------------------------------------------------------------


class TestSchedulerState:
    def test_latest_requests_reflects_last_observation(self):
        rt = build_runtime(latency_s_by_batch_size={1: 0.05})
        rt.add_robot("r0")
        rt.schedule_robot("r0", num_steps=10)
        rt.run_until(1.0)

        # The last observation the scheduler saw should be the last one the
        # broker sent — step 9 at t=0.45 → ``_next_action_step`` varies, but
        # ``observation_step`` is recorded directly on SlotRequest.
        latest = rt.scheduler._latest_requests["r0"]
        assert latest.observation_step == 9

    def test_latency_tracker_captures_profiled_infer_latency(self):
        rt = build_runtime(latency_s_by_batch_size={1: 0.05})
        rt.add_robot("r0")
        rt.schedule_robot("r0", num_steps=10)
        rt.run_until(1.0)

        # Pre-seeded in __post_init__, then continuously updated by
        # update_completion. All completions use exactly 0.05s so the EMA
        # should still be 0.05.
        assert rt.scheduler.latency_tracker.infer_latency[1] == pytest.approx(0.05)

    def test_no_requests_in_flight_after_run_completes(self):
        rt = build_runtime(latency_s_by_batch_size={1: 0.05})
        rt.add_robot("r0")
        rt.schedule_robot("r0", num_steps=10)
        rt.run_until(2.0)

        assert rt.scheduler.in_flight == 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_broker_reset_clears_scheduler_state(self):
        rt = build_runtime(latency_s_by_batch_size={1: 0.05})
        broker = rt.add_robot("r0")
        rt.schedule_robot("r0", num_steps=10)
        rt.run_until(0.5)

        assert "r0" in rt.scheduler._latest_requests

        broker.reset()
        rt.run_until_empty()

        # reset_robot should have cleared per-robot scheduler entries.
        assert "r0" not in rt.scheduler._latest_requests
        assert "r0" not in rt.scheduler._latest_scheduled_requests
        assert "r0" not in rt.scheduler._mirror

    def test_broker_state_after_reset(self):
        rt = build_runtime(latency_s_by_batch_size={1: 0.05})
        broker = rt.add_robot("r0")
        rt.schedule_robot("r0", num_steps=10)
        rt.run_until(0.5)

        broker.reset()

        assert broker.action_chunks == []
        assert len(broker._action_queue) == 0
        assert broker._next_observation_step == 0
        assert broker._next_action_step == 0
