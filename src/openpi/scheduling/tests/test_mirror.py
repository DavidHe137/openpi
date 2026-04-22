"""Tests for ``RequestScheduler``'s mirror of simulated robot state.

The mirror stores one ``RobotState`` per robot (the same dataclass the offline
simulator uses), with one ``ActionChunk`` appended per dispatched inference.
Deadlines, schedulable checks, and in-flight counts are all derived from it.
"""

from __future__ import annotations

import queue

from openpi_client.messages import InferType
import pytest

from openpi.scheduling.baselines import FixedSizeGreedyScheduler
from openpi.serving.schemas import SlotRequest
from openpi.shared.clock import SimClock


def make_request(
    *,
    robot_id: str = "r0",
    observation_step: int = 0,
    action_start_step: int | None = None,
    deadline_step: int = 10,
    execution_horizon: int = 10,
    control_hz: float = 20.0,
    request_timestamp: float = 0.0,
) -> SlotRequest:
    """Construct a minimal SlotRequest for scheduler tests."""
    return SlotRequest(
        slot_index=0,
        robot_id=robot_id,
        request_id=0,
        arrival_timestamp=request_timestamp,
        observation_step=observation_step,
        action_start_step=observation_step if action_start_step is None else action_start_step,
        request_timestamp=request_timestamp,
        deadline_step=deadline_step,
        execution_horizon=execution_horizon,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
        control_hz=control_hz,
    )


def make_scheduler(*, max_batch_size: int = 1) -> FixedSizeGreedyScheduler:
    sched = FixedSizeGreedyScheduler(queue.Queue(), max_batch_size=max_batch_size, clock=SimClock())
    # Seed latency tracker so schedule() can call total_latency without KeyError.
    for bs in range(1, max_batch_size + 1):
        sched.latency_tracker.update_infer(bs, 0.0)
    return sched


# ---------------------------------------------------------------------------
# Mirror initialisation / update
# ---------------------------------------------------------------------------


class TestMirrorInit:
    def test_fresh_scheduler_has_empty_mirror(self):
        sched = make_scheduler()
        assert sched._mirror == {}

    def test_update_seeds_robot_state_without_chunks(self):
        sched = make_scheduler()
        sched.update(make_request())
        assert "r0" in sched._mirror
        assert sched._mirror["r0"].chunks == []
        assert sched._mirror["r0"].step_based_coverage is True

    def test_repeat_update_does_not_duplicate_state(self):
        sched = make_scheduler()
        sched.update(make_request(observation_step=0))
        first = sched._mirror["r0"]
        sched.update(make_request(observation_step=1))
        assert sched._mirror["r0"] is first


# ---------------------------------------------------------------------------
# schedule() populates the mirror
# ---------------------------------------------------------------------------


class TestScheduleAppendsChunk:
    def test_schedule_appends_chunk_with_request_fields(self):
        sched = make_scheduler()
        # Seed obs + action latency so total_latency doesn't KeyError.
        sched.latency_tracker.observation_latency["r0"] = 0.0
        sched.latency_tracker.action_latency["r0"] = 0.0

        sched.update(make_request(observation_step=5, action_start_step=5, execution_horizon=10))
        sched.schedule()

        chunks = sched._mirror["r0"].chunks
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.start_action == 5
        assert chunk.horizon == 10
        assert chunk.observation_step == 5

    def test_arrival_step_advances_by_total_latency_in_control_steps(self):
        sched = make_scheduler()
        # 50 ms obs + 50 ms infer + 50 ms action = 150 ms = 3 steps at 20 Hz.
        sched.latency_tracker.observation_latency["r0"] = 0.05
        sched.latency_tracker.infer_latency[1] = 0.05
        sched.latency_tracker.action_latency["r0"] = 0.05

        sched.update(make_request(observation_step=10, action_start_step=10, control_hz=20.0))
        sched.schedule()

        (chunk,) = sched._mirror["r0"].chunks
        assert chunk.arrival_step == 10 + 3

    def test_zero_latency_makes_chunk_arrive_at_observation_step(self):
        sched = make_scheduler()
        sched.latency_tracker.observation_latency["r0"] = 0.0
        sched.latency_tracker.action_latency["r0"] = 0.0
        sched.update(make_request(observation_step=7, action_start_step=7))
        sched.schedule()
        (chunk,) = sched._mirror["r0"].chunks
        assert chunk.arrival_step == 7


# ---------------------------------------------------------------------------
# Deadline derivation from the mirror
# ---------------------------------------------------------------------------


class TestDeadline:
    def test_pre_schedule_deadline_uses_client_hint(self):
        sched = make_scheduler()
        # 20 Hz, observation_step=0, client says starvation at step 10 → 10/20 = 0.5 s
        # from request_timestamp=0.0.
        sched.update(make_request(observation_step=0, deadline_step=10, control_hz=20.0))
        assert sched.deadline("r0") == pytest.approx(0.5)

    def test_post_schedule_deadline_equals_horizon_expiry(self):
        sched = make_scheduler()
        sched.latency_tracker.observation_latency["r0"] = 0.0
        sched.latency_tracker.action_latency["r0"] = 0.0
        # observation_step=4, action_start_step=4, horizon=6 → starvation at step 10.
        # 10 steps - observation_step 4 = 6 steps remaining / 20 Hz = 0.3 s from t=0.
        sched.update(make_request(observation_step=4, action_start_step=4, execution_horizon=6, control_hz=20.0))
        sched.schedule()
        assert sched.deadline("r0") == pytest.approx(0.3)

    def test_multiple_chunks_deadline_is_furthest_expiry(self):
        sched = make_scheduler()
        sched.latency_tracker.observation_latency["r0"] = 0.0
        sched.latency_tracker.action_latency["r0"] = 0.0

        # First dispatch: chunk at [2, 2+5).
        sched.update(make_request(observation_step=2, action_start_step=2, execution_horizon=5, control_hz=20.0))
        sched.schedule()
        # Pretend the batch was picked up + inference completed so the greedy
        # scheduler is willing to dispatch again.
        sched._batch_queue.get_nowait()
        sched.notify_batch_complete()
        # Second dispatch: chunk at [7, 7+5), same robot. Bumps deadline to step 12.
        sched.update(make_request(observation_step=7, action_start_step=7, execution_horizon=5, control_hz=20.0))
        sched.schedule()

        # 12 steps - observation_step 7 = 5 steps / 20 Hz = 0.25 s (from new request_timestamp=0).
        assert sched.deadline("r0") == pytest.approx(0.25)
        assert len(sched._mirror["r0"].chunks) == 2


# ---------------------------------------------------------------------------
# schedulable_requests filter
# ---------------------------------------------------------------------------


class TestSchedulable:
    def test_new_robot_is_schedulable(self):
        sched = make_scheduler()
        sched.update(make_request(observation_step=0))
        assert [r.robot_id for r in sched.schedulable_requests] == ["r0"]

    def test_already_scheduled_obs_is_not_schedulable(self):
        sched = make_scheduler()
        sched.latency_tracker.observation_latency["r0"] = 0.0
        sched.latency_tracker.action_latency["r0"] = 0.0
        sched.update(make_request(observation_step=5, action_start_step=5))
        sched.schedule()

        # Same observation arrives again (stale) → action_start_step==last chunk's start_action.
        sched.update(make_request(observation_step=5, action_start_step=5))
        assert sched.schedulable_requests == []

    def test_newer_obs_becomes_schedulable_again(self):
        sched = make_scheduler()
        sched.latency_tracker.observation_latency["r0"] = 0.0
        sched.latency_tracker.action_latency["r0"] = 0.0
        sched.update(make_request(observation_step=5, action_start_step=5))
        sched.schedule()

        sched.update(make_request(observation_step=6, action_start_step=6))
        assert [r.robot_id for r in sched.schedulable_requests] == ["r0"]


# ---------------------------------------------------------------------------
# In-flight chunks
# ---------------------------------------------------------------------------


class TestInFlightChunks:
    def test_zero_latency_chunk_is_not_in_flight(self):
        sched = make_scheduler()
        sched.latency_tracker.observation_latency["r0"] = 0.0
        sched.latency_tracker.action_latency["r0"] = 0.0
        sched.update(make_request(observation_step=0, action_start_step=0))
        sched.schedule()
        assert sched.in_flight_chunks("r0") == []

    def test_positive_latency_chunk_is_in_flight_until_obs_catches_up(self):
        sched = make_scheduler()
        # 150 ms total = 3 steps at 20 Hz.
        sched.latency_tracker.observation_latency["r0"] = 0.05
        sched.latency_tracker.infer_latency[1] = 0.05
        sched.latency_tracker.action_latency["r0"] = 0.05

        sched.update(make_request(observation_step=0, action_start_step=0, control_hz=20.0))
        sched.schedule()
        assert len(sched.in_flight_chunks("r0")) == 1

        # Robot reports a new obs at step 3 → chunk has arrived, no longer in flight.
        sched.update(make_request(observation_step=3, action_start_step=3, control_hz=20.0))
        assert sched.in_flight_chunks("r0") == []


# ---------------------------------------------------------------------------
# reset / clear
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_robot_drops_mirror_entry(self):
        sched = make_scheduler()
        sched.latency_tracker.observation_latency["r0"] = 0.0
        sched.latency_tracker.action_latency["r0"] = 0.0
        sched.update(make_request(observation_step=2, action_start_step=2))
        sched.schedule()
        assert "r0" in sched._mirror

        sched.reset_robot("r0")
        assert "r0" not in sched._mirror
        assert "r0" not in sched._latest_requests
        assert "r0" not in sched._latest_scheduled_requests
