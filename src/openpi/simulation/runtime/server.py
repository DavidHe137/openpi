"""Single-process stand-in for openpi.serving.server's WS + scheduler + GPU trio.

The goal is to replicate the *state transitions* the real server performs on
the scheduler, without touching websockets, ZMQ, shared memory, or
subprocesses. Every wire message becomes a function call on the scheduler;
every batch the scheduler dispatches is handed to ``SimGPU``; every GPU
completion calls back into the scheduler — the exact same surface as the real
system.

Step-by-step parity with ``openpi.serving.server`` (recv loop) and
``openpi.serving.engine._run_gpu_worker``:

    WS.recv("infer")  ──► SlotRequest build  ──► scheduler.update()
        scheduler.schedule() happens in scheduler.py's main loop.
        → we call scheduler.schedule() here after each state update.

    WS.recv("ack")    ──► AckNotification   ──► scheduler.update_ack()

    WS.recv("reset")  ──► ResetRequest      ──► scheduler.reset_robot()

    GPU done          ──► CompletionNotification + notify_batch_complete
                          InferResponse → router → per-robot queue → send
"""

from __future__ import annotations

import itertools
import queue
from typing import TYPE_CHECKING

from openpi_client.messages import InferType
from openpi_client.schemas import Observation

from openpi.scheduling import RequestScheduler
from openpi.serving.schemas import AckNotification
from openpi.serving.schemas import RequestBatch
from openpi.serving.schemas import SlotRequest

if TYPE_CHECKING:
    from openpi.simulation.runtime.event_loop import EventLoop
    from openpi.simulation.runtime.gpu import SimGPU
    from openpi.simulation.runtime.wire import SimWire


class SimServer:
    """Hosts the scheduler + SimGPU and drives them from wire callbacks."""

    def __init__(
        self,
        event_loop: "EventLoop",
        wire: "SimWire",
        scheduler: RequestScheduler,
        gpu: "SimGPU",
        batch_queue: queue.Queue,
        *,
        control_hz: float,
    ) -> None:
        self._loop = event_loop
        self._wire = wire
        self._scheduler = scheduler
        self._gpu = gpu
        self._batch_queue = batch_queue
        self._control_hz = control_hz
        self._request_id_counter = itertools.count(1)
        self._slot_indices: dict[str, int] = {}

        wire.bind_server(
            on_infer=self._on_infer,
            on_ack=self._on_ack,
            on_reset=self._on_reset,
        )
        gpu.set_on_batch_complete(self._on_batch_complete)

    def register_robot(self, robot_id: str) -> None:
        if robot_id not in self._slot_indices:
            self._slot_indices[robot_id] = len(self._slot_indices)

    def _on_infer(
        self,
        robot_id: str,
        obs: Observation,
        deadline_step: int,
        action_start_step: int,
        execution_horizon: int,
        request_timestamp: float,
    ) -> None:
        self.register_robot(robot_id)
        arrival_timestamp = self._loop.now_s
        slot_req = SlotRequest(
            slot_index=self._slot_indices[robot_id],
            robot_id=robot_id,
            request_id=next(self._request_id_counter),
            arrival_timestamp=arrival_timestamp,
            observation_step=obs.step,
            action_start_step=action_start_step,
            request_timestamp=request_timestamp,
            deadline_step=deadline_step,
            execution_horizon=execution_horizon,
            infer_type=InferType.SYNC,
            params=None,
            noise=None,
            control_hz=self._control_hz,
        )
        self._scheduler.update(slot_req)
        self._schedule_and_drain()

    def _on_ack(
        self,
        robot_id: str,
        request_id: int,
        receive_time: float,
        execution_start_step: int,
        first_executed_index: int,
    ) -> None:
        self._scheduler.update_ack(
            AckNotification(
                robot_id=robot_id,
                request_id=request_id,
                receive_time=receive_time,
                server_send_time=self._loop.now_s,
            )
        )

    def _on_reset(self, robot_id: str) -> None:
        self._scheduler.reset_robot(robot_id)
        self._schedule_and_drain()

    def _on_batch_complete(self) -> None:
        """Called by SimGPU after completion wires up scheduler state."""
        self._schedule_and_drain()

    def _schedule_and_drain(self) -> None:
        self._scheduler.schedule()
        while True:
            try:
                batch: RequestBatch = self._batch_queue.get_nowait()
            except queue.Empty:
                break
            self._gpu.dispatch(batch)
