"""Parity tests between ``Simulator`` and ``ActionChunkBroker``.

Drives both under the same fixed batch schedule. At each control step, any
sim-predicted chunk whose ``arrival_step`` equals that step is injected into
the broker before ``infer()`` is called — mirroring the real background-thread
delivery. Then we compare per-step starvation / executed-action to the sim's
prediction.

The harness exists to surface the semantic gap between the sim's two coverage
modes and the broker's queue-based execution: step-based coverage treats a
chunk as a fixed time window of length ``horizon``, while the broker
(and action-based coverage) treats it as ``horizon`` executable actions, which
can drift past the time window when the robot starves mid-chunk.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np

from openpi.simulation.classes import CONTROL_PERIOD
from openpi.simulation.classes import ActionChunk as SimActionChunk
from openpi.simulation.classes import RobotState
from openpi.simulation.classes import SimulationState
from openpi.simulation.classes import SimulatorParameters
from openpi.simulation.sim import Simulator
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.schemas import ActionChunk as BrokerActionChunk
from openpi_client.schemas import Observation


def _make_broker(control_hz: int, execution_horizon: int) -> ActionChunkBroker:
    ws_mock = MagicMock()
    # Receive blocks forever so the background thread never processes a response.
    block = threading.Event()
    ws_mock.receive.side_effect = lambda: block.wait()
    return ActionChunkBroker(ws_mock, control_hz=control_hz, execution_horizon=execution_horizon)


def _make_obs(step: int) -> Observation:
    return Observation(
        state=np.zeros(1),
        step=step,
        image=np.zeros((1, 1, 3)),
        wrist_image=np.zeros((1, 1, 3)),
    )


def _inject_sim_chunk(broker: ActionChunkBroker, sim_chunk: SimActionChunk, request_id: int) -> None:
    """Mirror ``_on_response`` with data sourced from a sim chunk."""
    actions = np.zeros((sim_chunk.horizon, 1))
    with broker._lock:
        broker_chunk = BrokerActionChunk(
            observation_step=sim_chunk.observation_step,
            action_start_step=sim_chunk.start_action,
            execution_start_step=broker._next_observation_step,
            actions=actions,
            execution_horizon=sim_chunk.horizon,
            request_timestamp=0.0,
            response_timestamp=0.0,
            request_id=request_id,
        )
        broker._action_chunks.append(broker_chunk)
        broker._update_action_queue(broker_chunk)


def _advance_sim_idle(sim: Simulator, target_time: int) -> None:
    """Idle forward without serving. No-op if already past ``target_time``."""
    if target_time <= sim.state.time:
        return
    sim.state.time = target_time
    for robot_id, robot in sim.state.robots.items():
        step = sim.params.time_to_step_robot(target_time, robot_id)
        if step >= 0:
            robot.advance_to(step)


def _derive_sim_starvation(robot: RobotState) -> list[bool]:
    """Per-step starvation in action-based coverage terms.

    Step ``s`` is starved if no chunk is available to advance the next
    action index — i.e. no chunk with ``arrival_step <= s`` that covers
    the next action to execute.
    """
    starved: list[bool] = []
    action_number = 0
    for s in range(len(robot.action_index)):
        has_chunk = any(c.arrival_step <= s and c.covers(action_number) for c in robot.chunks)
        starved.append(not has_chunk)
        if has_chunk:
            action_number += 1
    return starved


def _derive_sim_starvation_step_based(robot: RobotState) -> list[bool]:
    """Per-step starvation in step-based coverage terms — fixed time-window."""
    return [not any(c.covers_step(s) for c in robot.chunks) for s in range(len(robot.action_index))]


def _drive(params: SimulatorParameters, schedule_events: list[tuple[int, list[str]]]):
    """Run sim and one broker per robot under the same schedule.

    Returns (sim, broker_actions) where broker_actions[robot_id] is a list of
    ``Action`` — one per control step in ``[0, end_step)``.
    """
    sim = Simulator(params)
    for t, batch in schedule_events:
        _advance_sim_idle(sim, t)
        sim.step(batch)

    end_step = params.end_time // CONTROL_PERIOD
    for robot in sim.state.robots.values():
        if end_step - 1 >= 0:
            robot.advance_to(end_step - 1)

    control_hz = 1000 // CONTROL_PERIOD
    brokers = {
        rid: _make_broker(control_hz=control_hz, execution_horizon=params.execution_horizon[rid])
        for rid in params.robot_ids
    }
    arrivals: dict[str, dict[int, list[tuple[int, SimActionChunk]]]] = {rid: {} for rid in params.robot_ids}
    for rid in params.robot_ids:
        for req_id, chunk in enumerate(sim.state.robots[rid].chunks):
            arrivals[rid].setdefault(chunk.arrival_step, []).append((req_id, chunk))

    broker_actions: dict[str, list] = {rid: [] for rid in params.robot_ids}
    for s in range(end_step):
        for rid in params.robot_ids:
            for req_id, chunk in arrivals[rid].get(s, []):
                _inject_sim_chunk(brokers[rid], chunk, req_id)
            broker_actions[rid].append(brokers[rid].infer(_make_obs(s)))
    return sim, broker_actions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_continuous_serve_action_based_matches_broker():
    """Serve r1 every d_infer interval; action-based coverage should track the broker."""
    params = SimulatorParameters(
        end_time=1000,  # 20 control steps
        d_infer={1: 50},  # one control period
        robot_ids=["r1"],
        d_send={"r1": 0},
        d_recv={"r1": 0},
        execution_horizon={"r1": 10},
        step_based_coverage=False,
    )
    schedule = [(t, ["r1"]) for t in range(0, params.end_time, 50)]
    sim, broker_actions = _drive(params, schedule)

    broker_starved = [a.action_chunk_index is None for a in broker_actions["r1"]]
    sim_starved = _derive_sim_starvation(sim.state.robots["r1"])
    assert broker_starved == sim_starved, f"\nbroker: {broker_starved}\nsim:    {sim_starved}"

    action_number = 0
    for s, action in enumerate(broker_actions["r1"]):
        if action.action_chunk_index is not None:
            assert action.step == action_number, (
                f"step {s}: broker popped .step={action.step}, expected action index {action_number}"
            )
            action_number += 1


def test_single_serve_step_based_diverges_from_broker():
    """One serve + long tail: step-based coverage underestimates broker reach by one
    action's worth of steps because the broker's queue extends past the fixed window."""
    params = SimulatorParameters(
        end_time=1000,
        d_infer={1: 50},
        robot_ids=["r1"],
        d_send={"r1": 0},
        d_recv={"r1": 0},
        execution_horizon={"r1": 10},
        step_based_coverage=True,  # current default
    )
    sim, broker_actions = _drive(params, [(0, ["r1"])])
    broker_starved = [a.action_chunk_index is None for a in broker_actions["r1"]]
    sim_starved_step = _derive_sim_starvation_step_based(sim.state.robots["r1"])
    sim_starved_action = _derive_sim_starvation(sim.state.robots["r1"])

    # Broker: step 0 starved (chunk arrives at step 1), then 10 actions pop at steps 1-10,
    # then starved 11-19. Total = 10 starved steps.
    assert sum(broker_starved) == 10
    assert broker_starved[10] is False
    assert all(broker_starved[s] for s in (0, 11, 12, 13, 14, 15, 16, 17, 18, 19))

    # Step-based view marks step 10 as starved (time window [1, 9]). Action-based matches broker.
    assert sim_starved_step != broker_starved
    assert sim_starved_step[10] is True
    assert sim_starved_action == broker_starved


def test_simulator_accepts_initial_state_seed():
    """Simulator should honor a pre-populated SimulationState: existing chunks drive
    advance_to, server_available_at delays the next batch, and undo does not unwind
    past the seeded state."""
    params = SimulatorParameters(
        end_time=500,
        d_infer={1: 50},
        robot_ids=["r1"],
        d_send={"r1": 0},
        d_recv={"r1": 0},
        execution_horizon={"r1": 10},
        step_based_coverage=False,
    )
    seeded_chunk = SimActionChunk(start_action=0, horizon=5, arrival_step=2, observation_step=0)
    seed = SimulationState(
        time=0,
        server_available_at=100,
        robots={"r1": RobotState(step_based_coverage=False, chunks=[seeded_chunk])},
    )
    sim = Simulator(params, initial_state=seed)

    assert sim.state is seed
    assert sim.state.server_available_at == 100
    assert sim.state.robots["r1"].chunks == [seeded_chunk]

    sim.state.robots["r1"].advance_to(4)
    # chunk covers actions 0..4, arrival=2 → steps 0,1 starved, 2,3,4 cover actions 1,2,3.
    assert sim.state.robots["r1"].action_index == [0, 0, 1, 2, 3]

    # A step() at time 0 should respect the seeded server_available_at when bumping state.time.
    sim.step(["r1"])
    assert sim.state.time == 100
    # Now two chunks: the seeded one plus the newly-scheduled one.
    assert len(sim.state.robots["r1"].chunks) == 2


def test_starvation_gap_then_recover():
    """Serve once, skip serves while chunk drains + starves, then serve again.
    Action-based coverage and broker must agree on where starvation occurs."""
    params = SimulatorParameters(
        end_time=1500,  # 30 steps
        d_infer={1: 50},
        robot_ids=["r1"],
        d_send={"r1": 0},
        d_recv={"r1": 0},
        execution_horizon={"r1": 10},
        step_based_coverage=False,
    )
    # Serve at t=0. Wait until chunk is drained + some starvation, then serve again at t=700.
    schedule = [(0, ["r1"]), (700, ["r1"])]
    sim, broker_actions = _drive(params, schedule)

    broker_starved = [a.action_chunk_index is None for a in broker_actions["r1"]]
    sim_starved = _derive_sim_starvation(sim.state.robots["r1"])
    assert broker_starved == sim_starved, f"\nbroker: {broker_starved}\nsim:    {sim_starved}"
