from __future__ import annotations

import importlib.util
import pathlib
import time

from openpi_client.messages import InferType

from openpi.scheduling.baselines import GreedyPlusScheduler
from openpi.scheduling.baselines import WDRRScheduler
from openpi.serving.scheduler import _SCHEDULER_REGISTRY
from openpi.serving.schemas import SlotRequest


class _InMemoryQueue:
    def __init__(self) -> None:
        self.items: list[list[SlotRequest]] = []

    def qsize(self) -> int:
        return len(self.items)

    def put_nowait(self, item: list[SlotRequest]) -> None:
        self.items.append(item)

    def pop(self) -> list[SlotRequest]:
        assert self.items, "Queue is empty"
        return self.items.pop(0)


def _make_request(
    robot_id: str,
    *,
    request_id: int,
    action_start_step: int,
    deadline_offset_s: float,
    execution_horizon: int = 10,
    control_hz: float = 20.0,
) -> SlotRequest:
    now = time.time()
    return SlotRequest(
        slot_index=0,
        robot_id=robot_id,
        request_id=request_id,
        arrival_timestamp=now,
        observation_step=action_start_step,
        action_start_step=action_start_step,
        request_timestamp=now - 0.01,
        deadline=now + deadline_offset_s,
        execution_horizon=execution_horizon,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
        control_hz=control_hz,
    )


def _seed_latency(scheduler, robot_id: str, *, obs_ms: float, action_ms: float, infer_ms: float = 8.0) -> None:
    scheduler.latency.update_obs(robot_id, 1.0, 1.0 - obs_ms / 1000.0)
    scheduler.latency.update_action_delivery(robot_id, 1.0 + action_ms / 1000.0, 1.0)
    scheduler.latency.update_infer(1, infer_ms)


def _dispatch_once(scheduler, queue: _InMemoryQueue, requests: list[SlotRequest]) -> list[str]:
    for request in requests:
        scheduler.update(request)
    scheduler.schedule()
    batch = queue.pop()
    return [request.robot_id for request in batch]


def test_greedy_plus_prioritizes_tighter_deadline_when_costs_similar() -> None:
    queue = _InMemoryQueue()
    scheduler = GreedyPlusScheduler(
        queue,
        max_batch_size=1,
        batch_profile={1: 1.0},
        scheduler_lambda_age=0.0,
        scheduler_lambda_debt=0.0,
        greedy_plus_lambda_var=0.0,
    )
    _seed_latency(scheduler, "robot_a", obs_ms=10.0, action_ms=10.0)
    _seed_latency(scheduler, "robot_b", obs_ms=10.0, action_ms=10.0)

    picked = _dispatch_once(
        scheduler,
        queue,
        [
            _make_request("robot_a", request_id=1, action_start_step=0, deadline_offset_s=0.1),
            _make_request("robot_b", request_id=2, action_start_step=0, deadline_offset_s=1.0),
        ],
    )
    assert picked == ["robot_a"]


def test_greedy_plus_prioritizes_lower_cost_when_urgency_similar() -> None:
    queue = _InMemoryQueue()
    scheduler = GreedyPlusScheduler(
        queue,
        max_batch_size=1,
        batch_profile={1: 1.0},
        scheduler_lambda_age=0.0,
        scheduler_lambda_debt=0.0,
        greedy_plus_lambda_var=0.0,
    )
    _seed_latency(scheduler, "robot_fast", obs_ms=2.0, action_ms=2.0)
    _seed_latency(scheduler, "robot_slow", obs_ms=80.0, action_ms=80.0)

    picked = _dispatch_once(
        scheduler,
        queue,
        [
            _make_request("robot_fast", request_id=1, action_start_step=0, deadline_offset_s=1.0),
            _make_request("robot_slow", request_id=2, action_start_step=0, deadline_offset_s=1.0),
        ],
    )
    assert picked == ["robot_fast"]


def test_greedy_plus_enforces_max_consecutive_guardrail() -> None:
    queue = _InMemoryQueue()
    scheduler = GreedyPlusScheduler(
        queue,
        max_batch_size=1,
        batch_profile={1: 1.0},
        scheduler_lambda_age=0.0,
        scheduler_lambda_debt=0.0,
        greedy_plus_lambda_var=0.0,
        greedy_plus_max_consecutive=1,
        scheduler_service_window_decisions=100,
    )
    _seed_latency(scheduler, "robot_a", obs_ms=10.0, action_ms=10.0)
    _seed_latency(scheduler, "robot_b", obs_ms=10.0, action_ms=10.0)

    first = _dispatch_once(
        scheduler,
        queue,
        [
            _make_request("robot_a", request_id=1, action_start_step=0, deadline_offset_s=0.05),
            _make_request("robot_b", request_id=2, action_start_step=0, deadline_offset_s=5.0),
        ],
    )
    second = _dispatch_once(
        scheduler,
        queue,
        [
            _make_request("robot_a", request_id=3, action_start_step=1, deadline_offset_s=0.05),
            _make_request("robot_b", request_id=4, action_start_step=1, deadline_offset_s=5.0),
        ],
    )
    assert first == ["robot_a"]
    assert second == ["robot_b"]


def test_greedy_plus_enforces_sliding_service_floor() -> None:
    queue = _InMemoryQueue()
    scheduler = GreedyPlusScheduler(
        queue,
        max_batch_size=1,
        batch_profile={1: 1.0},
        scheduler_lambda_age=0.0,
        scheduler_lambda_debt=0.0,
        greedy_plus_lambda_var=0.0,
        greedy_plus_max_consecutive=100,
        scheduler_service_window_decisions=2,
    )
    _seed_latency(scheduler, "robot_a", obs_ms=8.0, action_ms=8.0)
    _seed_latency(scheduler, "robot_b", obs_ms=8.0, action_ms=8.0)

    picks = [
        _dispatch_once(
            scheduler,
            queue,
            [
                _make_request("robot_a", request_id=2 * t + 1, action_start_step=t, deadline_offset_s=0.05),
                _make_request("robot_b", request_id=2 * t + 2, action_start_step=t, deadline_offset_s=20.0),
            ],
        )[0]
        for t in range(3)
    ]

    assert picks[0] == "robot_a"
    assert picks[1] == "robot_a"
    assert picks[2] == "robot_b"


def test_wdrr_prefers_low_cost_robot_over_many_decisions() -> None:
    queue = _InMemoryQueue()
    scheduler = WDRRScheduler(
        queue,
        max_batch_size=1,
        batch_profile={1: 1.0},
        scheduler_lambda_age=0.0,
        scheduler_lambda_debt=0.0,
        scheduler_service_window_decisions=1000,
        wdrr_q0=1.0,
    )
    _seed_latency(scheduler, "robot_fast", obs_ms=2.0, action_ms=2.0)
    _seed_latency(scheduler, "robot_slow", obs_ms=120.0, action_ms=120.0)

    fast_count = 0
    slow_count = 0
    for t in range(80):
        picked = _dispatch_once(
            scheduler,
            queue,
            [
                _make_request("robot_fast", request_id=2 * t + 1, action_start_step=t, deadline_offset_s=2.0),
                _make_request("robot_slow", request_id=2 * t + 2, action_start_step=t, deadline_offset_s=2.0),
            ],
        )[0]
        if picked == "robot_fast":
            fast_count += 1
        else:
            slow_count += 1

    assert fast_count > slow_count
    assert slow_count > 0


def test_wdrr_high_cost_robot_still_gets_service() -> None:
    queue = _InMemoryQueue()
    scheduler = WDRRScheduler(
        queue,
        max_batch_size=1,
        batch_profile={1: 1.0},
        scheduler_lambda_age=0.25,
        scheduler_lambda_debt=0.5,
        scheduler_service_window_decisions=5,
        wdrr_q0=1.0,
    )
    _seed_latency(scheduler, "robot_fast", obs_ms=2.0, action_ms=2.0)
    _seed_latency(scheduler, "robot_slow", obs_ms=150.0, action_ms=150.0)

    slow_count = 0
    for t in range(40):
        picked = _dispatch_once(
            scheduler,
            queue,
            [
                _make_request("robot_fast", request_id=2 * t + 1, action_start_step=t, deadline_offset_s=2.0),
                _make_request("robot_slow", request_id=2 * t + 2, action_start_step=t, deadline_offset_s=2.0),
            ],
        )[0]
        if picked == "robot_slow":
            slow_count += 1

    assert slow_count > 0


def test_wdrr_is_deterministic_for_fixed_inputs() -> None:
    def _run_sequence() -> list[str]:
        queue = _InMemoryQueue()
        scheduler = WDRRScheduler(
            queue,
            max_batch_size=1,
            batch_profile={1: 1.0},
            scheduler_lambda_age=0.1,
            scheduler_lambda_debt=0.2,
            scheduler_service_window_decisions=7,
            wdrr_q0=1.0,
        )
        _seed_latency(scheduler, "robot_a", obs_ms=5.0, action_ms=5.0)
        _seed_latency(scheduler, "robot_b", obs_ms=50.0, action_ms=50.0)
        _seed_latency(scheduler, "robot_c", obs_ms=25.0, action_ms=25.0)

        return [
            _dispatch_once(
                scheduler,
                queue,
                [
                    _make_request("robot_a", request_id=3 * t + 1, action_start_step=t, deadline_offset_s=2.0),
                    _make_request("robot_b", request_id=3 * t + 2, action_start_step=t, deadline_offset_s=2.0),
                    _make_request("robot_c", request_id=3 * t + 3, action_start_step=t, deadline_offset_s=2.0),
                ],
            )[0]
            for t in range(30)
        ]

    assert _run_sequence() == _run_sequence()


def test_wdrr_fallback_selects_robot_when_no_priority_crosses_threshold() -> None:
    queue = _InMemoryQueue()
    scheduler = WDRRScheduler(
        queue,
        max_batch_size=1,
        batch_profile={1: 1.0},
        scheduler_lambda_age=0.0,
        scheduler_lambda_debt=0.0,
        scheduler_service_window_decisions=100,
        wdrr_q0=0.1,
    )
    _seed_latency(scheduler, "robot_a", obs_ms=20.0, action_ms=20.0)
    _seed_latency(scheduler, "robot_b", obs_ms=20.0, action_ms=20.0)

    picked = _dispatch_once(
        scheduler,
        queue,
        [
            _make_request("robot_a", request_id=1, action_start_step=0, deadline_offset_s=2.0),
            _make_request("robot_b", request_id=2, action_start_step=0, deadline_offset_s=2.0),
        ],
    )
    assert picked == ["robot_a"]


def test_scheduler_registry_contains_new_algorithms() -> None:
    assert "greedy_plus" in _SCHEDULER_REGISTRY
    assert "wdrr" in _SCHEDULER_REGISTRY


def test_serve_policy_builds_scheduler_kwargs_for_new_algorithms() -> None:
    script_path = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "serve_policy.py"
    spec = importlib.util.spec_from_file_location("serve_policy", script_path)
    assert spec is not None
    assert spec.loader is not None
    serve_policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve_policy)

    literal = serve_policy.Args.__annotations__["scheduling_algorithm"]
    assert "greedy_plus" in literal.__args__
    assert "wdrr" in literal.__args__

    greedy_plus_args = serve_policy.Args(scheduling_algorithm="greedy_plus")
    greedy_plus_kwargs = serve_policy.build_scheduler_kwargs(greedy_plus_args, action_horizon_steps=10)
    assert greedy_plus_kwargs is not None
    assert "scheduler_ema_alpha" in greedy_plus_kwargs
    assert "greedy_plus_lambda_var" in greedy_plus_kwargs
    assert "greedy_plus_max_consecutive" in greedy_plus_kwargs

    wdrr_args = serve_policy.Args(scheduling_algorithm="wdrr")
    wdrr_kwargs = serve_policy.build_scheduler_kwargs(wdrr_args, action_horizon_steps=10)
    assert wdrr_kwargs is not None
    assert "scheduler_ema_alpha" in wdrr_kwargs
    assert "wdrr_q0" in wdrr_kwargs
