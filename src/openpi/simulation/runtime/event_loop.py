"""Deterministic priority-queue event loop for the sim harness.

Events are ``(fire_at_s, seq, callback)``. Ties on ``fire_at_s`` are broken by
insertion order, so the loop is fully deterministic regardless of how many
events are scheduled at the same simulated instant.

Each pop advances the ``SimClock`` to the event's ``fire_at_s`` and invokes
the callback synchronously — callbacks may schedule further events at
``now + delta`` via :meth:`EventLoop.schedule`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import heapq
import itertools

from openpi.shared.clock import SimClock

Callback = Callable[[], None]


@dataclass(order=True)
class _Event:
    fire_at_s: float
    seq: int
    callback: Callback = field(compare=False)


class EventLoop:
    def __init__(self, start_s: float = 0.0) -> None:
        self._clock = SimClock(start_s=start_s)
        self._heap: list[_Event] = []
        self._seq = itertools.count()

    @property
    def clock(self) -> SimClock:
        return self._clock

    @property
    def now_s(self) -> float:
        return self._clock.now_s

    def schedule(self, delay_s: float, callback: Callback) -> None:
        """Run ``callback`` at ``now + delay_s`` (delay_s >= 0)."""
        self.schedule_at(self._clock.now_s + delay_s, callback)

    def schedule_at(self, fire_at_s: float, callback: Callback) -> None:
        """Run ``callback`` at absolute simulated time ``fire_at_s``."""
        assert fire_at_s >= self._clock.now_s, (
            f"Cannot schedule in the past: fire_at_s={fire_at_s} now={self._clock.now_s}"
        )
        heapq.heappush(self._heap, _Event(fire_at_s, next(self._seq), callback))

    def run_until(self, end_s: float) -> None:
        """Pop events in time order until the clock would pass ``end_s``."""
        while self._heap and self._heap[0].fire_at_s <= end_s:
            event = heapq.heappop(self._heap)
            self._clock.advance_to(event.fire_at_s)
            event.callback()
        if self._clock.now_s < end_s:
            self._clock.advance_to(end_s)

    def run_until_empty(self) -> None:
        """Pop all scheduled events in time order."""
        while self._heap:
            event = heapq.heappop(self._heap)
            self._clock.advance_to(event.fire_at_s)
            event.callback()

    def pending(self) -> int:
        return len(self._heap)
