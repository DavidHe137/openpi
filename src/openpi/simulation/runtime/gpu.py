"""Sim GPU worker: deterministic actions + profiled per-batch-size latency.

Mirrors the production GPU worker (``openpi.serving.engine._run_gpu_worker``)
at the level of messages the scheduler cares about, but doesn't load a model.

For each dispatched ``RequestBatch`` the sim:
  1. schedules inference completion at ``now + latency_s[batch_size]``;
  2. at completion, builds an ``InferResponse`` for each request with actions
     encoded as ``observation_step * 1.0`` so tests can assert that the broker
     received the chunk tied to a specific observation;
  3. notifies the scheduler of completion (``update_completion`` +
     ``notify_batch_complete``) so ``_in_flight`` and infer-latency estimates
     stay in sync with reality;
  4. routes each response through ``SimWire.server_send_response`` which
     schedules delivery to the broker after the network delay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from openpi_client.messages import InferResponse

from openpi.scheduling import RequestScheduler
from openpi.serving.schemas import CompletionNotification
from openpi.serving.schemas import RequestBatch

if TYPE_CHECKING:
    from openpi.simulation.runtime.event_loop import EventLoop
    from openpi.simulation.runtime.wire import SimWire


class SimGPU:
    def __init__(
        self,
        event_loop: "EventLoop",
        wire: "SimWire",
        scheduler: RequestScheduler,
        *,
        latency_s_by_batch_size: dict[int, float],
        action_horizon: int,
        action_dim: int,
    ) -> None:
        self._loop = event_loop
        self._wire = wire
        self._scheduler = scheduler
        self._latency_s = dict(latency_s_by_batch_size)
        self._action_horizon = action_horizon
        self._action_dim = action_dim

        self._last_served_request_id: dict[str, int] = {}
        self._on_batch_complete_cb = None

    def set_on_batch_complete(self, callback) -> None:
        """Install a callback fired after each batch's scheduler-side update.

        Used by SimServer to re-trigger ``scheduler.schedule()`` once the
        server becomes free again.
        """
        self._on_batch_complete_cb = callback

    def dispatch(self, batch: RequestBatch) -> None:
        batch_size = len(batch.requests)
        assert batch_size in self._latency_s, f"No latency profile for batch_size={batch_size}"
        latency = self._latency_s[batch_size]

        fresh = [
            sr
            for sr in batch.requests
            if sr.request_id > self._last_served_request_id.get(sr.robot_id, 0)
        ]
        if not fresh:
            return

        inference_start_time = self._loop.now_s

        def on_complete() -> None:
            inference_end_time = self._loop.now_s
            server_send_time = inference_end_time
            for sr in fresh:
                actions = np.full(
                    (self._action_horizon, self._action_dim),
                    float(sr.observation_step),
                    dtype=np.float32,
                )
                response = InferResponse(
                    robot_id=sr.robot_id,
                    request_id=sr.request_id,
                    observation_step=sr.observation_step,
                    action_start_step=sr.action_start_step,
                    request_timestamp=sr.request_timestamp,
                    execution_horizon=sr.execution_horizon,
                    actions=actions,
                    server_arrival_time=sr.arrival_timestamp,
                    inference_start_time=inference_start_time,
                    inference_end_time=inference_end_time,
                    server_send_time=server_send_time,
                )
                self._last_served_request_id[sr.robot_id] = sr.request_id
                self._scheduler.update_completion(
                    CompletionNotification(
                        robot_id=sr.robot_id,
                        action_start_step=sr.action_start_step,
                        request_id=sr.request_id,
                        batch_size=batch_size,
                        inference_duration=latency,
                    )
                )
                self._wire.server_send_response(response)

            self._scheduler.notify_batch_complete()
            if self._on_batch_complete_cb is not None:
                self._on_batch_complete_cb()

        self._loop.schedule(latency, on_complete)
