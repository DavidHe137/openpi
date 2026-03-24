import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from openpi_client.action_chunkers.sync import SyncBroker
from openpi_client.schemas import Observation


def _make_observation(step: int = 0) -> Observation:
    return Observation(
        state=np.zeros(7),
        step=step,
        image=np.zeros((224, 224, 3)),
        wrist_image=np.zeros((224, 224, 3)),
    )


def _make_ws_client(action_horizon: int):
    ws = MagicMock()
    ws.server_metadata = SimpleNamespace(action_horizon=action_horizon)
    block = threading.Event()
    ws.receive.side_effect = lambda: block.wait()
    return ws


def test_sync_broker_defaults_to_server_action_horizon_when_zero() -> None:
    ws = _make_ws_client(action_horizon=10)
    broker = SyncBroker(ws_client=ws, control_hz=20, execution_horizon=0)
    assert broker.execution_horizon == 10


def test_sync_broker_uses_requested_execution_horizon() -> None:
    ws = _make_ws_client(action_horizon=10)
    broker = SyncBroker(ws_client=ws, control_hz=20, execution_horizon=4)
    assert broker.execution_horizon == 4

    obs = _make_observation(step=1)
    broker._infer(obs)

    ws.send.assert_called_once()
    assert ws.send.call_args.kwargs["execution_horizon"] == 4


def test_sync_broker_rejects_execution_horizon_above_server_action_horizon() -> None:
    ws = _make_ws_client(action_horizon=10)
    with pytest.raises(AssertionError):
        SyncBroker(ws_client=ws, control_hz=20, execution_horizon=11)
