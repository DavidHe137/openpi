"""Lookahead scheduler: DFS over H future inference rounds, minimising starvation.

Translates the discrete-step LookaheadScheduler from action-chunk-scheduling to
wall-clock time using the GPU batch-latency profile already available in openpi.

Key mapping from discrete → wall-clock:
  d_infer[k]          → batch_profile_ms[k] / 1000  (seconds)
  execution_horizon   → execution_horizon_s           (seconds, configurable)
  deadline            → SlotRequest.deadline          (wall-clock seconds)
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import multiprocessing as mp
import time

from openpi.scheduling import RequestScheduler
from openpi.serving.schemas import SlotRequest


@dataclass
class _SimState:
    server_free_at: float  # wall-clock time when server next becomes available
    robot_deadlines: dict[str, float] = field(default_factory=dict)  # robot_id -> deadline


class LookaheadScheduler(RequestScheduler):
    """MPC scheduler: DFS over H future rounds, picks the first action of the best trajectory.

    Candidate generation (same as discrete-step original):
      For each batch size k ∈ [1, max_batch_size], take the k robots with the
      earliest deadlines.  This gives N+1 candidates per step rather than 2^N.

    Cost: binary starvation — number of robots whose deadline has already passed
          when the server starts the batch.
    """

    def __init__(
        self,
        batch_queue: mp.Queue,
        max_batch_size: int = 1,
        batch_profile: dict[int, float] | None = None,
        horizon: int = 5,
        execution_horizon_s: float = 0.5,
    ):
        super().__init__(batch_queue, max_batch_size, batch_profile)
        self.horizon = horizon
        self.execution_horizon_s = execution_horizon_s
        self._server_free_at: float = 0.0

    # ------------------------------------------------------------------
    # Override schedule() to track when the server will next be free.
    # ------------------------------------------------------------------
    def schedule(self) -> None:
        batches = self.get_next_batches()
        for batch in batches:
            for request in batch:
                self._deadlines[request.robot_id] = request.deadline
                self._latest_scheduled_requests[request.robot_id] = request
            k = len(batch)
            if k in self._batch_profile_ms:
                self._server_free_at = time.time() + self._batch_profile_ms[k] / 1000.0
            self._batch_queue.put_nowait(batch)

    # ------------------------------------------------------------------
    # Core scheduling logic
    # ------------------------------------------------------------------
    def get_next_batches(self) -> list[list[SlotRequest]]:
        if self._batch_queue.qsize() > 0:
            return []

        candidates = self._get_schedulable_requests()
        if not candidates:
            return []

        now = time.time()
        req_by_id = {req.robot_id: req for req in candidates}
        initial_state = _SimState(
            server_free_at=max(self._server_free_at, now),
            robot_deadlines={req.robot_id: req.deadline for req in candidates},
        )

        best_cost: list[float] = [float("inf")]
        best_first_batch: list[list[SlotRequest] | None] = [None]

        def dfs(state: _SimState, depth: int, cost: float, first: list[SlotRequest] | None) -> None:
            if depth == 0 or not state.robot_deadlines:
                if cost < best_cost[0]:
                    best_cost[0] = cost
                    best_first_batch[0] = first
                return

            sorted_robots = sorted(state.robot_deadlines.items(), key=lambda x: x[1])
            max_k = min(len(sorted_robots), self._max_batch_size)

            for k in range(1, max_k + 1):
                if k not in self._batch_profile_ms:
                    continue

                batch_ids = [rid for rid, _ in sorted_robots[:k]]
                latency_s = self._batch_profile_ms[k] / 1000.0
                t_start = state.server_free_at
                t_end = t_start + latency_s

                step_cost = sum(1 for _, dl in state.robot_deadlines.items() if dl < t_start)

                new_deadlines = dict(state.robot_deadlines)
                for rid in batch_ids:
                    new_deadlines[rid] = t_end + self.execution_horizon_s

                new_state = _SimState(server_free_at=t_end, robot_deadlines=new_deadlines)

                first_batch = [req_by_id[rid] for rid in batch_ids if rid in req_by_id] if first is None else first
                dfs(new_state, depth - 1, cost + step_cost, first_batch)

        dfs(initial_state, self.horizon, 0.0, None)

        if best_first_batch[0] is None:
            # Fallback: greedy (earliest deadline, batch size 1)
            fallback = sorted(candidates, key=lambda r: r.deadline)
            return [fallback[:1]]

        return [best_first_batch[0]]
