"""Tests for ActionChunkBroker.

Most tests bypass the background thread by calling _update_action_queue directly.
The threading test at the bottom exercises the full receive path and will currently
fail because InferResponse lacks an `observation_step` field (needed by
ActionChunk.from_infer_response).
"""

import queue
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.messages import InferResponse
from openpi_client.schemas import ActionChunk, Observation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_obs(step: int) -> Observation:
    return Observation(
        state=np.zeros(7),
        step=step,
        image=np.zeros((224, 224, 3)),
        wrist_image=np.zeros((224, 224, 3)),
    )


def make_action_chunk(
    action_start_step: int,
    execution_horizon: int,
    observation_step: int = 0,
    execution_start_step: int = 0,
    action_dim: int = 7,
    request_id: int = 0,
) -> ActionChunk:
    """Build an ActionChunk with actions filled as row-index floats for easy assertions."""
    actions = np.arange(execution_horizon * action_dim, dtype=float).reshape(execution_horizon, action_dim)
    return ActionChunk(
        observation_step=observation_step,
        action_start_step=action_start_step,
        execution_start_step=execution_start_step,
        actions=actions,
        execution_horizon=execution_horizon,
        request_timestamp=0.0,
        response_timestamp=1.0,
        request_id=request_id,
    )


def make_infer_response(
    action_start_step: int,
    execution_horizon: int,
    observation_step: int = 0,
    action_dim: int = 7,
    request_id: int = 0,
) -> InferResponse:
    actions = np.arange(execution_horizon * action_dim, dtype=float).reshape(execution_horizon, action_dim)
    return InferResponse(
        robot_id="test",
        request_id=request_id,
        observation_step=observation_step,
        action_start_step=action_start_step,
        request_timestamp=0.0,
        actions=actions,
        execution_horizon=execution_horizon,
    )


def make_broker(
    receive_queue: queue.Queue | None = None,
    control_hz: int = 10,
    execution_horizon: int = 0,
) -> tuple[ActionChunkBroker, MagicMock]:
    """Create a broker with a mocked websocket client.

    If receive_queue is provided, ws_mock.receive() blocks on it (use to inject
    responses from the test). Otherwise receive() blocks forever, keeping the
    background thread out of the way.
    """
    ws_mock = MagicMock()
    if receive_queue is not None:
        ws_mock.receive.side_effect = receive_queue.get
    else:
        block = threading.Event()
        ws_mock.receive.side_effect = lambda: block.wait()
    broker = ActionChunkBroker(ws_mock, control_hz=control_hz, execution_horizon=execution_horizon)
    return broker, ws_mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def broker_and_mock() -> tuple[ActionChunkBroker, MagicMock]:
    return make_broker()


# ---------------------------------------------------------------------------
# _update_action_queue
# ---------------------------------------------------------------------------


class TestUpdateActionQueue:
    def test_populates_queue(self, broker_and_mock):
        broker, ws_mock = broker_and_mock
        chunk = make_action_chunk(action_start_step=1, execution_horizon=5)
        broker._action_chunks.append(chunk)
        broker._update_action_queue(chunk)

        assert len(broker._action_queue) == 5
        for i, action in enumerate(broker._action_queue):
            assert action.step == 1 + i
            assert action.index_in_chunk == i
            np.testing.assert_array_equal(action.action, chunk.get_action(i))

    def test_action_chunk_index_reflects_chunks_list(self, broker_and_mock):
        broker, _ = broker_and_mock
        chunk = make_action_chunk(action_start_step=1, execution_horizon=3)
        broker._action_chunks.append(chunk)
        broker._update_action_queue(chunk)

        for action in broker._action_queue:
            assert action.action_chunk_index == 0

    def test_prunes_stale_tail_before_appending(self, broker_and_mock):
        """New chunk starting at step 3 should replace steps 3+ from the old chunk."""
        broker, _ = broker_and_mock
        chunk1 = make_action_chunk(action_start_step=1, execution_horizon=5, request_id=0)
        broker._action_chunks.append(chunk1)
        broker._update_action_queue(chunk1)
        # queue: steps 1,2,3,4,5 from chunk1

        chunk2 = make_action_chunk(action_start_step=3, execution_horizon=3, request_id=1)
        broker._action_chunks.append(chunk2)
        broker._update_action_queue(chunk2)
        # expected: steps 1,2 from chunk1 + steps 3,4,5 from chunk2

        actions = list(broker._action_queue)
        assert len(actions) == 5
        assert actions[0].step == 1 and actions[0].action_chunk_index == 0
        assert actions[1].step == 2 and actions[1].action_chunk_index == 0
        assert actions[2].step == 3 and actions[2].action_chunk_index == 1
        assert actions[3].step == 4 and actions[3].action_chunk_index == 1
        assert actions[4].step == 5 and actions[4].action_chunk_index == 1


# ---------------------------------------------------------------------------
# infer()
# ---------------------------------------------------------------------------


class TestInfer:
    def test_null_action_when_queue_empty(self, broker_and_mock):
        broker, _ = broker_and_mock
        # _observation_step starts at 0 after reset(), so first valid step is 1
        action = broker.infer(make_obs(1))

        assert action.action_chunk_index is None
        assert action.index_in_chunk is None
        np.testing.assert_array_equal(action.action, np.zeros(7))

    def test_null_action_gripper_from_last_chunk(self, broker_and_mock):
        """Null action gripper mirrors the last action of the most recent chunk."""
        broker, _ = broker_and_mock
        chunk = make_action_chunk(action_start_step=1, execution_horizon=3)
        broker._action_chunks.append(chunk)
        # Don't populate the queue — force null action path

        action = broker.infer(make_obs(1))

        assert action.action[-1] == chunk.get_action(-1)[-1]

    def test_returns_action_from_queue(self, broker_and_mock):
        broker, ws_mock = broker_and_mock
        chunk = make_action_chunk(action_start_step=1, execution_horizon=3)
        broker._action_chunks.append(chunk)
        broker._update_action_queue(chunk)
        ws_mock.reset_mock()

        action = broker.infer(make_obs(1))

        assert action.step == 1
        assert action.action_chunk_index == 0
        assert action.index_in_chunk == 0
        np.testing.assert_array_equal(action.action, chunk.get_action(0))

    def test_actions_consumed_in_order(self, broker_and_mock):
        broker, ws_mock = broker_and_mock
        chunk = make_action_chunk(action_start_step=1, execution_horizon=3)
        broker._action_chunks.append(chunk)
        broker._update_action_queue(chunk)
        ws_mock.reset_mock()

        for expected_index in range(3):
            action = broker.infer(make_obs(expected_index + 1))
            assert action.index_in_chunk == expected_index

    def test_out_of_order_observation_raises(self, broker_and_mock):
        broker, _ = broker_and_mock
        with pytest.raises(AssertionError, match="Observations must be streamed in order"):
            broker.infer(make_obs(5))

    def test_actions_left_history(self, broker_and_mock):
        broker, ws_mock = broker_and_mock
        chunk = make_action_chunk(action_start_step=1, execution_horizon=3)
        broker._action_chunks.append(chunk)
        broker._update_action_queue(chunk)
        ws_mock.reset_mock()

        broker.infer(make_obs(1))  # pops 1, 2 left
        broker.infer(make_obs(2))  # pops 1, 1 left
        broker.infer(make_obs(3))  # pops 1, 0 left
        broker.infer(make_obs(4))  # null action, 0 left

        assert broker.actions_left_history == [3, 2, 1, 0]

    def test_action_step_increments_only_on_real_action(self, broker_and_mock):
        broker, ws_mock = broker_and_mock
        chunk = make_action_chunk(action_start_step=1, execution_horizon=2)
        broker._action_chunks.append(chunk)
        broker._update_action_queue(chunk)
        ws_mock.reset_mock()

        broker.infer(make_obs(1))  # real action
        broker.infer(make_obs(2))  # real action
        broker.infer(make_obs(3))  # null action - _action_step should not increment

        assert broker._action_step == 2

    def test_send_called_on_each_infer(self, broker_and_mock):
        broker, ws_mock = broker_and_mock
        broker.infer(make_obs(1))
        ws_mock.send.assert_called_once()


# ---------------------------------------------------------------------------
# min_execution_horizon
# ---------------------------------------------------------------------------


class TestMinExecutionHorizon:
    def test_send_always_called_with_full_queue(self):
        """Broker sends on every infer even when queue is full (no threshold suppression)."""
        broker, ws_mock = make_broker(execution_horizon=5)
        chunk = make_action_chunk(action_start_step=1, execution_horizon=10)
        broker._action_chunks.append(chunk)
        broker._update_action_queue(chunk)
        ws_mock.reset_mock()

        broker.infer(make_obs(1))  # queue has 10 — should still send
        ws_mock.send.assert_called_once()

    def test_send_passes_execution_horizon_kwarg(self):
        """send() is called with the correct execution_horizon kwarg."""
        broker, ws_mock = make_broker(execution_horizon=7)
        broker.infer(make_obs(1))
        call_kwargs = ws_mock.send.call_args[1]
        assert call_kwargs.get("execution_horizon") == 7

    def test_send_called_on_every_step(self):
        """send() is called on each infer step regardless of queue depth."""
        broker, ws_mock = make_broker(execution_horizon=3)
        chunk = make_action_chunk(action_start_step=1, execution_horizon=10)
        broker._action_chunks.append(chunk)
        broker._update_action_queue(chunk)
        ws_mock.reset_mock()

        for i in range(1, 6):
            broker.infer(make_obs(i))

        assert ws_mock.send.call_count == 5


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_clears_all_state(self, broker_and_mock):
        broker, ws_mock = broker_and_mock
        chunk = make_action_chunk(action_start_step=1, execution_horizon=3)
        broker._action_chunks.append(chunk)
        broker._update_action_queue(chunk)
        broker.infer(make_obs(1))
        ws_mock.reset_mock()

        broker.reset()

        assert len(broker._action_queue) == 0
        assert broker._action_chunks == []
        assert broker._actions_left_history == []
        assert broker._observation_step == 0
        assert broker._action_step == 0

    def test_calls_ws_reset(self, broker_and_mock):
        broker, ws_mock = broker_and_mock
        ws_mock.reset_mock()
        broker.reset()
        ws_mock.reset.assert_called_once()


# ---------------------------------------------------------------------------
# Background thread / _receive_actions
# ---------------------------------------------------------------------------


class TestThreadedReceive:
    def test_received_chunk_populates_queue(self):
        """Background thread receives a response and populates the action queue."""
        response_queue = queue.Queue()
        broker, _ = make_broker(receive_queue=response_queue)

        response = make_infer_response(action_start_step=1, execution_horizon=3)
        response_queue.put(response)

        # Wait for background thread to process
        deadline = time.time() + 1.0
        while time.time() < deadline:
            with broker._lock:
                if len(broker._action_queue) > 0:
                    break
            time.sleep(0.01)

        with broker._lock:
            assert len(broker._action_queue) == 3

    def test_received_chunk_triggers_ack(self):
        response_queue = queue.Queue()
        broker, ws_mock = make_broker(receive_queue=response_queue)
        ws_mock.reset_mock()

        response = make_infer_response(action_start_step=1, execution_horizon=3, request_id=99)
        response_queue.put(response)

        deadline = time.time() + 1.0
        while time.time() < deadline:
            if ws_mock.send_ack.called:
                break
            time.sleep(0.01)

        assert ws_mock.send_ack.called
        call_args = ws_mock.send_ack.call_args[0]
        assert call_args[0] == 99  # request_id


# ---------------------------------------------------------------------------
# Naive async scenario tests
# ---------------------------------------------------------------------------


def inject_chunk(broker, action_start_step, execution_horizon, request_id=0):
    """Inject a chunk directly into the broker, bypassing the background thread.

    Mirrors what _receive_actions does: sets execution_start_step=_observation_step,
    appends to _action_chunks, and calls _update_action_queue — all under the lock.
    """
    with broker._lock:
        chunk = make_action_chunk(
            action_start_step=action_start_step,
            execution_horizon=execution_horizon,
            execution_start_step=broker._observation_step,
            request_id=request_id,
        )
        broker._action_chunks.append(chunk)
        broker._update_action_queue(chunk)
    return chunk


class TestNaiveAsync:
    """End-to-end scenario tests for the async chunking logic.

    Starvation metric: broker.actions_left_history.count(0) — zeros recorded
    before the pop means the queue was empty at that step.
    """

    def _make_broker(self):
        return make_broker(control_hz=10, execution_horizon=3)

    def test_case_a_no_starvation(self):
        """Chunk arrives before queue drains → starvation=0."""
        broker, ws_mock = self._make_broker()

        # Inject chunk1 covering steps 0-9
        inject_chunk(broker, action_start_step=0, execution_horizon=10, request_id=0)

        # Run 3 infers → _action_step=3
        for obs_step in range(1, 4):
            broker.infer(make_obs(obs_step))

        # Inject chunk2 starting at step 3 (overlaps remaining queue tail)
        chunk2 = inject_chunk(broker, action_start_step=3, execution_horizon=10, request_id=1)

        # Filter: 3+i >= 3 → all 10 actions
        with broker._lock:
            assert len(broker._action_queue) == 10
        assert chunk2.execution_start_step == 3

        # Run 10 more infers (obs 4-13)
        for obs_step in range(4, 14):
            broker.infer(make_obs(obs_step))

        assert broker.actions_left_history.count(0) == 0

    def test_case_b_stale_chunk_after_starvation(self):
        """Stale chunk (action_start_step=3) arrives after 1 starvation step → 3 actions."""
        broker, ws_mock = self._make_broker()

        inject_chunk(broker, action_start_step=0, execution_horizon=10, request_id=0)

        # Exhaust chunk1 (obs 1-10)
        for obs_step in range(1, 11):
            broker.infer(make_obs(obs_step))
        # obs 11 → starvation
        broker.infer(make_obs(11))
        assert broker.actions_left_history.count(0) == 1

        # Stale chunk: action_start_step=3, _action_step=10 → filter keeps i=7,8,9 (steps 10,11,12)
        chunk2 = inject_chunk(broker, action_start_step=3, execution_horizon=10, request_id=1)
        with broker._lock:
            assert len(broker._action_queue) == 3
        assert chunk2.execution_start_step == 11

        # Run infer obs 12-14 → records [3,2,1], no new zeros
        for obs_step in range(12, 15):
            broker.infer(make_obs(obs_step))

        assert broker.actions_left_history.count(0) == 1

    def test_case_c_fresh_chunk_after_starvation(self):
        """Fresh chunk (action_start_step=11) arrives after 1 starvation step → 10 actions."""
        broker, ws_mock = self._make_broker()

        inject_chunk(broker, action_start_step=0, execution_horizon=10, request_id=0)

        # Exhaust chunk1 (obs 1-10) + 1 starvation step (obs 11)
        for obs_step in range(1, 11):
            broker.infer(make_obs(obs_step))
        broker.infer(make_obs(11))
        assert broker.actions_left_history.count(0) == 1

        # Fresh chunk starting at step 11: filter 11+i >= 10 → all 10 pass
        inject_chunk(broker, action_start_step=11, execution_horizon=10, request_id=1)
        with broker._lock:
            actions_list = list(broker._action_queue)
        assert len(actions_list) == 10
        assert actions_list[0].step == 11
        assert actions_list[-1].step == 20

        # Run infer obs 12-21 → no new zeros
        for obs_step in range(12, 22):
            broker.infer(make_obs(obs_step))

        assert broker.actions_left_history.count(0) == 1
