# Sim harness handoff

**Goal:** single-process event-loop simulator that exercises the *real*
`ActionChunkBroker` against the *real* `RequestScheduler` subclasses so we
can assert parity between broker/scheduler internal state and what the sim
"knows" is true. Built to mirror `openpi.serving.server` + `engine.py` +
`scheduler.py` message flow without touching ZMQ / shared memory /
websockets / subprocesses.

## Status

End-to-end plumbing works. Sanity run: 1 robot, 30 steps @ 20 Hz,
`d_infer=50 ms`, `d_net=0`, `FixedSizeGreedyScheduler` → broker emits
chunks correctly, action[0] values equal the observation step that
produced them.

Task tracker state (in TaskList):

- #1–#5 completed (Clock, scheduler wiring, event loop, wire, GPU,
  server, runtime).
- #6 **completed**: initial parity tests landed in
  `src/openpi/simulation/runtime/tests/test_sim_runtime.py` (15 tests, ~0.07s).

## What's in place

### New files

- `src/openpi/shared/clock.py` — `Clock` ABC, `RealClock`, `SimClock`,
  `default_clock()`. Testability seam.
- `src/openpi/simulation/runtime/event_loop.py` — `EventLoop` with
  `SimClock`, heapq priority queue, `schedule`/`schedule_at`/`run_until`/
  `run_until_empty`. Sequence-tie-broken for determinism.
- `src/openpi/simulation/runtime/wire.py` — `SimWire` + `SimWsClient`.
  `SimWsClient` is a drop-in for `BidirectionalWebsocket` (same
  `send`/`send_ack`/`reset`/`server_metadata` surface). `SimWire.bind_server`
  hooks up server-side callbacks; `register_client(robot_id, on_response)`
  hooks up broker-side response delivery. All hops go through
  `event_loop.schedule(d_net_s, cb)`.
- `src/openpi/simulation/runtime/gpu.py` — `SimGPU.dispatch(batch)`
  schedules completion after `latency_s[batch_size]`, builds
  `InferResponse` with `actions[:, :] = observation_step` (so tests can
  assert chunk provenance), calls `scheduler.update_completion` +
  `notify_batch_complete`, fires optional `set_on_batch_complete` callback
  (used by `SimServer` to re-run `scheduler.schedule()` once the server is
  free again).
- `src/openpi/simulation/runtime/server.py` — `SimServer` binds wire
  callbacks → `scheduler.update / update_ack / reset_robot`, drains
  `batch_queue` into `SimGPU` after each `scheduler.schedule()`.
- `src/openpi/simulation/runtime/runtime.py` — `SimRuntime`. Constructs
  the whole graph. `add_robot(robot_id)` creates a real `ActionChunkBroker`
  with `start_receive_thread=False` + `SimWsClient`. Pre-seeds
  `observation_latency` and `action_latency` with `d_net_s` (mirrors
  warmup). `schedule_robot(robot_id, num_steps)` schedules
  `broker.infer(obs_i)` at `i / control_hz` and records each returned
  `Action` into `traces[robot_id]` as `RobotTraceEntry(step, sim_time_s,
  action)`.

### Modifications to existing files

- `packages/openpi-client/src/openpi_client/action_chunkers/action_chunk_broker.py`
  - Added `start_receive_thread: bool = True` constructor kwarg and gated
    the daemon thread on it.
  - Extracted `_on_response(infer_response)` from `_receive_actions` so
    the sim can inject responses synchronously from the event loop.
  - Fixed a pre-existing syntax bug in `_infer` (missing comma between
    `self.deadline_step` and `self._next_action_step`).
- `src/openpi/scheduling/__init__.py`
  - `RequestScheduler.__init__` takes `clock: Clock | None = None`
    (defaults to `default_clock()`). `collect_trace`, `schedule` now use
    `self._clock.time()`.
  - **Bug fix** in `update()` — logger used
    `self._deadline_steps[request.robot_id]` before inserting, which
    KeyErrored on first update. Now uses `.get(..., 0)`.
- `src/openpi/scheduling/baselines.py` — `GreedyDeadlineScheduler` uses
  `self._clock.time()` in `get_largest_batch_size`.
- `src/openpi/scheduling/lookahead.py` — clock kwarg, `self._clock.time()`.
- `src/openpi/scheduling/lookahead_actions.py` — `self._clock.time()`.
- `src/openpi/scheduling/receding_horizon_ilp.py` — clock kwarg, all
  `time.monotonic()` / `time.time()` → `self._clock.*`.
- `pyproject.toml` — removed the dangling `action-chunk-scheduling`
  workspace member (it pointed at `third_party/action-chunk-scheduling`
  which doesn't exist). `uv sync --no-install-workspace` now resolves
  dependencies; full sync still fails because `jax-cuda12-plugin` has no
  macOS wheel.

## Env setup (macOS)

```bash
# deps needed for the sim + broker only (skips jax/torch/etc.)
uv pip install numpy jaxtyping pytest pandas pyarrow requests typing_extensions msgpack msgpack-numpy websockets
uv pip install -e packages/openpi-client
```

Running anything:

```bash
source .venv/bin/activate
PYTHONPATH=src python ...
```

Full `uv sync` fails on macOS (no jax-cuda wheel). Don't fight it; the
piecewise install is sufficient for the sim.

## How to run a smoke test

```python
from openpi.simulation.runtime.runtime import SimRuntime
from openpi.scheduling.baselines import FixedSizeGreedyScheduler
import queue

sched = FixedSizeGreedyScheduler(queue.Queue(), max_batch_size=1)
rt = SimRuntime(
    scheduler=sched,
    latency_s_by_batch_size={1: 0.05},
    control_hz=20, action_horizon=10, action_dim=7, execution_horizon=5,
    d_net_s=0.0,
)
rt.add_robot("r0")
rt.schedule_robot("r0", num_steps=30)
rt.run_until(2.0)

for entry in rt.trace("r0")[:8]:
    print(entry.step, entry.sim_time_s, entry.action.action[0],
          entry.action.action_chunk_index, entry.action.index_in_chunk)
```

Expected shape of output (chunk_idx progresses, action[0] equals the
observation_step of the inference that produced it):

```
step=0 t=0.000 null
step=1 t=0.050 null
step=2 t=0.100 chunk 0 idx 0 action[0]=0.0
step=3 t=0.150 chunk 0 idx 1 action[0]=0.0
step=4 t=0.200 chunk 1 idx 1 action[0]=2.0
...
```

## Initial parity tests (task #6, done)

Landed in `src/openpi/simulation/runtime/tests/test_sim_runtime.py`:

- `TestDeterminism` — same config (1-robot and 2-robot) → identical trace
  fingerprints.
- `TestZeroLatencyGolden` — with `d_infer=0`, step N consumes the chunk
  built from obs N-1 (`action[0] == N-1`, `index_in_chunk == 0`).
- `TestNullPrefix` — null-action prefix length is
  `floor((2*d_net + d_infer) * hz) + 1` (the `+1` is the sim-time tie rule
  — a step and its triggering response-delivery land at the same sim time;
  step was scheduled first so it fires first → null).
- `TestChunkProvenance` — for every real action, `action[0]` equals the
  `observation_step` of the chunk the broker indexed into.
- `TestTwoRobotEDF` — both robots make progress with `max_batch_size=1`.
- `TestSchedulerState` — `_latest_requests` tracks the last observation,
  `latency_tracker.infer_latency[1]` stays pinned at the profiled latency,
  `in_flight == 0` after the run drains.
- `TestReset` — `broker.reset()` clears both broker and scheduler state
  (`_latest_requests`, `_latest_scheduled_requests`, `_deadline_steps`).

`tests/conftest.py` patches out `ServerMetadata.__post_init__` to skip the
ipinfo.io HTTP call (multi-second hang per `add_robot`).

### Ideas for follow-up tests (not written)

- `scheduler.deadline(robot_id)` numeric parity — currently blocked by the
  mixed-units bug in `RequestScheduler.schedule()` (writes a *timestamp*
  into `_deadline_steps`, then `deadline()` does `deadline_step -
  observation_step` which subtracts seconds from a step index).
- Multi-robot EDF ordering — actually compare the order of served batches
  against deadline-sorted candidates.
- Long-horizon starvation metric — `actions_left_history.count(0)` vs
  analytical expectation.

### Test infra notes

- Pre-seed scheduler `infer_latency` via `SimRuntime.__init__` (already
  done via `latency_tracker.update_infer` per batch size).
- `observation_latency` and `action_latency` are pre-seeded to `d_net_s`
  in `add_robot` so schedulers that call `total_latency` don't KeyError
  before the first ack round-trip.
- `ActionChunk.from_infer_response` uses wall-clock `time.time()` for
  `response_timestamp` — `SimWsClient.send_ack` overrides the ack's
  `receive_time` with `self._clock.time()` so `update_action_delivery`
  stays self-consistent. If any test wants sim-time `response_timestamp`
  on ActionChunk itself, we'd need to either monkey-patch `time.time` or
  add a clock seam to `ActionChunk.from_infer_response`.

## Known pre-existing issues not yet fixed

- `src/openpi/serving/server.py:424` builds `SlotRequest(deadline=...)` but
  `SlotRequest` has `deadline_step`, not `deadline`. The real server is
  broken on this path. The sim uses `deadline_step` directly.
- `src/openpi/scheduling/baselines.py` `GreedyDeadlineScheduler` calls
  `self._deadline(...)` (underscore) — method on the base is `deadline`.
  Scheduler is broken as-is.
- `RoundRobinScheduler.__init__` doesn't accept / thread a `clock` kwarg.
  Low priority; sim's FixedSizeGreedy path doesn't hit this.
- Deadline-step log warning in `RequestScheduler.update` is spammy under
  sim (fires on every obs). Consider demoting to debug.

## Open design questions (worth raising with the user)

- The sim derives "ground truth" directly from knowing `d_net_s` and
  `latency_s_by_batch_size`, so there isn't a separate shadow state
  object. Tests compare real-broker / real-scheduler state against
  first-principles calculations. If the user wanted a `GroundTruth`
  dataclass that *mirrors* broker/scheduler state at each tick (and
  diffs after the run), that would live in `runtime.py` — not yet built.
- `SimServer._schedule_and_drain` calls `scheduler.schedule()` after
  every state change. Real server calls it on a 1ms poll. For any
  scheduler whose batch decision depends on `clock.time()` crossing a
  threshold (e.g. `server_available_at` in Lookahead), this is
  equivalent because new state changes happen exactly when the threshold
  crosses. But if you add a scheduler that uses wallclock drift in a
  subtler way, it may diverge.
