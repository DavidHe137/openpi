"""Top-level sim harness.

Given a scheduler and a per-batch-size latency profile, SimRuntime:
  - constructs the event loop, wire, GPU worker, and server glue;
  - registers one real ``ActionChunkBroker`` per robot, each pointing at a
    ``SimWsClient`` (same surface as ``BidirectionalWebsocket``) and started
    with ``start_receive_thread=False`` so the sim injects responses
    directly;
  - schedules ``broker.infer(obs_i)`` calls at ``i / control_hz`` for every
    robot, recording each emitted action on a per-robot trace;
  - steps the event loop until the simulated end time.

Tests can then assert against ``runtime.trace(robot_id)``, inspect the real
scheduler's state (``scheduler.latency_tracker``, ``scheduler._mirror``),
and inspect the real broker's state (``broker.action_chunks``,
``broker._action_queue``) — there is no separate ground-truth mirror because
the sim itself IS the ground truth: d_net and d_infer are known, observation
cadence is known, so the exact action consumed at each step is derivable
from first principles.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import queue
from typing import Optional

import numpy as np
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.schemas import Action
from openpi_client.schemas import Observation
from openpi_client.schemas import ServerMetadata

from openpi.scheduling import RequestScheduler
from openpi.simulation.runtime.event_loop import EventLoop
from openpi.simulation.runtime.gpu import SimGPU
from openpi.simulation.runtime.server import SimServer
from openpi.simulation.runtime.wire import SimWire
from openpi.simulation.runtime.wire import SimWsClient


@dataclass
class RobotTraceEntry:
    step: int
    sim_time_s: float
    action: Action


@dataclass
class SimRuntime:
    scheduler: RequestScheduler
    latency_s_by_batch_size: dict[int, float]
    control_hz: int
    action_horizon: int
    action_dim: int
    execution_horizon: int
    d_net_s: float = 0.0
    state_dim: int = 7
    image_shape: tuple[int, int, int] = (1, 1, 3)

    event_loop: EventLoop = field(init=False)
    wire: SimWire = field(init=False)
    server: SimServer = field(init=False)
    gpu: SimGPU = field(init=False)
    batch_queue: queue.Queue = field(init=False)

    brokers: dict[str, ActionChunkBroker] = field(init=False, default_factory=dict)
    traces: dict[str, list[RobotTraceEntry]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.event_loop = EventLoop()
        self.wire = SimWire(self.event_loop, d_net_s=self.d_net_s)
        self.batch_queue = queue.Queue()
        # Rebind scheduler to our in-process queue + sim clock so the same
        # scheduler instance can be reused across a test if desired.
        self.scheduler._batch_queue = self.batch_queue
        self.scheduler._clock = self.event_loop.clock
        for batch_size, latency in self.latency_s_by_batch_size.items():
            self.scheduler.latency_tracker.update_infer(batch_size, latency)
        self.gpu = SimGPU(
            self.event_loop,
            self.wire,
            self.scheduler,
            latency_s_by_batch_size=self.latency_s_by_batch_size,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
        )
        self.server = SimServer(
            self.event_loop,
            self.wire,
            self.scheduler,
            self.gpu,
            self.batch_queue,
            control_hz=float(self.control_hz),
        )

    # ----- robot registration + driving -----

    def _server_metadata(self) -> ServerMetadata:
        return ServerMetadata(
            config_name="sim",
            checkpoint_dir="",
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            num_steps=0,
            max_batch_size=self.scheduler._max_batch_size,
            env="SIM",
            scheduling_algorithm="sim",
            tunnel_url=None,
            location="sim",
        )

    def add_robot(
        self,
        robot_id: str,
        *,
        start_offset_s: float = 0.0,
        execution_horizon: Optional[int] = None,
    ) -> ActionChunkBroker:
        assert robot_id not in self.brokers, f"Robot {robot_id} already registered"
        ws_client = SimWsClient(
            robot_id=robot_id,
            wire=self.wire,
            clock=self.event_loop.clock,
            server_metadata=self._server_metadata(),
        )
        broker = ActionChunkBroker(
            ws_client=ws_client,
            control_hz=self.control_hz,
            execution_horizon=execution_horizon if execution_horizon is not None else self.execution_horizon,
            start_receive_thread=False,
        )
        # Route GPU responses directly into the broker via its _on_response.
        self.wire.register_client(robot_id, broker._on_response)
        self.brokers[robot_id] = broker
        self.traces[robot_id] = []
        self.server.register_robot(robot_id)
        # Seed per-robot latency estimates (production does this during warmup).
        # Both directions share d_net_s in the sim, so seed obs and action latency at d_net_s.
        self.scheduler.latency_tracker.observation_latency.setdefault(robot_id, self.d_net_s)
        self.scheduler.latency_tracker.action_latency.setdefault(robot_id, self.d_net_s)
        return broker

    def _make_observation(self, step: int) -> Observation:
        return Observation(
            state=np.zeros(self.state_dim, dtype=np.float32),
            step=step,
            image=np.zeros(self.image_shape, dtype=np.uint8),
            wrist_image=np.zeros(self.image_shape, dtype=np.uint8),
        )

    def schedule_robot(
        self,
        robot_id: str,
        num_steps: int,
        *,
        start_offset_s: float = 0.0,
    ) -> None:
        """Fire ``broker.infer(obs_i)`` at ``start_offset + i/control_hz``."""
        assert robot_id in self.brokers, f"Unknown robot_id {robot_id}"
        broker = self.brokers[robot_id]
        trace = self.traces[robot_id]
        step_duration = 1.0 / self.control_hz

        for i in range(num_steps):
            fire_at = start_offset_s + i * step_duration

            def tick(step=i, fire_at=fire_at):
                obs = self._make_observation(step)
                action = broker.infer(obs)
                trace.append(RobotTraceEntry(step=step, sim_time_s=fire_at, action=action))

            self.event_loop.schedule_at(fire_at, tick)

    # ----- accessors -----

    def trace(self, robot_id: str) -> list[RobotTraceEntry]:
        return self.traces[robot_id]

    def run_until(self, end_s: float) -> None:
        self.event_loop.run_until(end_s)

    def run_until_empty(self) -> None:
        self.event_loop.run_until_empty()
