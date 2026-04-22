"""Clock seam for testability.

Production code takes a ``Clock`` (default ``RealClock``) and calls
``clock.time()`` / ``clock.monotonic()`` instead of ``time.time()`` /
``time.monotonic()``. Tests pass a ``SimClock`` that advances under explicit
control, making time-dependent logic deterministic and fast.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import time as _time


class Clock(ABC):
    @abstractmethod
    def time(self) -> float:
        """Wall-clock seconds since epoch."""

    @abstractmethod
    def monotonic(self) -> float:
        """Monotonic seconds from an arbitrary epoch."""


class RealClock(Clock):
    def time(self) -> float:
        return _time.time()

    def monotonic(self) -> float:
        return _time.monotonic()


_default_clock = RealClock()


def default_clock() -> Clock:
    """Singleton RealClock — use as the default for non-test code."""
    return _default_clock


class SimClock(Clock):
    """A clock whose value is set explicitly by an event loop.

    ``time()`` and ``monotonic()`` are kept in lockstep — in sim they share the
    same origin, so code that mixes both sees a consistent timeline.
    """

    def __init__(self, start_s: float = 0.0) -> None:
        self._now_s = float(start_s)

    @property
    def now_s(self) -> float:
        return self._now_s

    def advance_to(self, now_s: float) -> None:
        assert now_s >= self._now_s, f"SimClock cannot go backwards ({self._now_s} -> {now_s})"
        self._now_s = float(now_s)

    def advance_by(self, delta_s: float) -> None:
        assert delta_s >= 0, f"SimClock delta must be non-negative (got {delta_s})"
        self._now_s += float(delta_s)

    def time(self) -> float:
        return self._now_s

    def monotonic(self) -> float:
        return self._now_s
