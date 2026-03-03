from __future__ import annotations

from openpi_client.messages import InferType

from openpi.scheduling import lookahead
from openpi.scheduling.lookahead import LookaheadScheduler
from openpi.serving.schemas import SlotRequest


class FakeQueue:
    def __init__(self) -> None:
        self.items: list[list[SlotRequest]] = []

    def qsize(self) -> int:
        return len(self.items)

    def full(self) -> bool:
        return False

    def put_nowait(self, batch: list[SlotRequest]) -> None:
        self.items.append(batch)

    def clear(self) -> None:
        self.items.clear()


def _make_request(robot_id: str, request_id: int, deadline: float) -> SlotRequest:
    return SlotRequest(
        slot_index=0,
        robot_id=robot_id,
        request_id=request_id,
        arrival_timestamp=deadline,
        start_step=0,
        request_timestamp=deadline,
        deadline=deadline,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
    )


def test_lookahead_prefers_batch_that_reduces_joint_starvation(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr(lookahead.time, "time", lambda: now)

    scheduler = LookaheadScheduler(
        FakeQueue(),
        max_batch_size=2,
        batch_profile={1: 120.0, 2: 150.0},
        horizon_ms=300,
        timestep_ms=10,
        action_horizon_steps=10,
        control_hz=20,
    )
    scheduler.update(_make_request("r1", 1, 100.10))
    scheduler.update(_make_request("r2", 2, 100.10))

    batches = scheduler.get_next_batches()

    assert [[request.robot_id for request in batch] for batch in batches] == [["r1", "r2"]]


def test_lookahead_tracks_predicted_server_busy_after_queue_drains(monkeypatch) -> None:
    current_time = [100.0]
    monkeypatch.setattr(lookahead.time, "time", lambda: current_time[0])

    queue = FakeQueue()
    scheduler = LookaheadScheduler(
        queue,
        max_batch_size=1,
        batch_profile={1: 100.0},
        horizon_ms=200,
        timestep_ms=10,
        action_horizon_steps=10,
        control_hz=20,
    )
    scheduler.update(_make_request("r1", 1, 100.05))
    scheduler.schedule()
    assert len(queue.items) == 1

    queue.clear()  # Simulate the GPU worker dequeuing the batch while still computing it.
    current_time[0] = 100.05
    scheduler.update(_make_request("r2", 2, 100.05))
    assert scheduler.get_next_batches() == []

    current_time[0] = 100.11
    batches = scheduler.get_next_batches()

    assert [[request.robot_id for request in batch] for batch in batches] == [["r2"]]


def test_lookahead_uses_predicted_inflight_chunk_for_robot_urgency(monkeypatch) -> None:
    current_time = [100.0]
    monkeypatch.setattr(lookahead.time, "time", lambda: current_time[0])

    queue = FakeQueue()
    scheduler = LookaheadScheduler(
        queue,
        max_batch_size=1,
        batch_profile={1: 100.0},
        horizon_ms=200,
        timestep_ms=10,
        action_horizon_steps=10,
        control_hz=20,
    )
    scheduler.update(_make_request("r1", 1, 100.05))
    scheduler.schedule()
    queue.clear()

    current_time[0] = 100.02
    scheduler.update(_make_request("r1", 2, 100.03))

    current_time[0] = 100.11
    scheduler.update(_make_request("r2", 3, 100.13))
    batches = scheduler.get_next_batches()

    assert [[request.robot_id for request in batch] for batch in batches] == [["r2"]]


def test_lookahead_records_planning_time_summary(monkeypatch) -> None:
    now = 100.0
    counter_values = iter([1_000_000, 6_000_000])
    monkeypatch.setattr(lookahead.time, "time", lambda: now)
    monkeypatch.setattr(lookahead.time, "perf_counter_ns", lambda: next(counter_values))

    scheduler = LookaheadScheduler(
        FakeQueue(),
        max_batch_size=1,
        batch_profile={1: 100.0},
        horizon_ms=200,
        timestep_ms=10,
        action_horizon_steps=10,
        control_hz=20,
    )
    scheduler.update(_make_request("r1", 1, 100.05))

    scheduler.get_next_batches()
    summary = scheduler.timing_summary()

    assert summary is not None
    assert summary["planning_calls"] == 1.0
    assert summary["mean_planning_time_ms"] == 5.0
    assert summary["p99_planning_time_ms"] == 5.0
