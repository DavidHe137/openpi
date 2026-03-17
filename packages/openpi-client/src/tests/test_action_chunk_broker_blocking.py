import queue
import threading
import time
from unittest.mock import MagicMock

import numpy as np

from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.messages import InferResponse
from openpi_client.schemas import Observation


def _make_obs(step: int) -> Observation:
    return Observation(
        state=np.zeros(7),
        step=step,
        image=np.zeros((224, 224, 3)),
        wrist_image=np.zeros((224, 224, 3)),
    )


def _make_response(*, request_id: int, action_start_step: int, execution_horizon: int) -> InferResponse:
    actions = np.arange(execution_horizon * 7, dtype=float).reshape(execution_horizon, 7)
    return InferResponse(
        robot_id="robot_0",
        request_id=request_id,
        observation_step=0,
        action_start_step=action_start_step,
        request_timestamp=0.0,
        actions=actions,
        execution_horizon=execution_horizon,
    )


def test_infer_blocks_until_first_chunk_when_enabled():
    receive_queue: queue.Queue = queue.Queue()
    ws_mock = MagicMock()
    ws_mock.receive.side_effect = receive_queue.get

    broker = ActionChunkBroker(ws_mock, control_hz=20, block_until_first_chunk=True)

    result: dict[str, object] = {}

    def _run_infer() -> None:
        result["action"] = broker.infer(_make_obs(0))

    infer_thread = threading.Thread(target=_run_infer)
    infer_thread.start()

    time.sleep(0.05)
    assert infer_thread.is_alive()
    assert ws_mock.send.call_count == 1

    receive_queue.put(_make_response(request_id=1, action_start_step=0, execution_horizon=4))

    infer_thread.join(timeout=1.0)
    assert not infer_thread.is_alive()
    action = result["action"]
    assert action.action_chunk_index == 0
    assert action.index_in_chunk == 0


def test_infer_returns_null_immediately_when_bootstrap_blocking_disabled():
    ws_mock = MagicMock()
    blocker = threading.Event()
    ws_mock.receive.side_effect = lambda: blocker.wait()

    broker = ActionChunkBroker(ws_mock, control_hz=20, block_until_first_chunk=False)

    t0 = time.perf_counter()
    action = broker.infer(_make_obs(0))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 50.0
    assert action.action_chunk_index is None
    assert action.index_in_chunk is None
