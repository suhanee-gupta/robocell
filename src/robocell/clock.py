"""Deterministic simulated clock driving asyncio actors in lock-step.

Each actor is an asyncio task that loops `await clock.wait_tick(name)` -> do one tick of
work. The driver (`SimClock.run`) waits until *every* registered actor is parked at
`wait_tick`, then advances the tick and wakes them in registration order. Nothing ever
awaits wall time (unless `pace` is set for a live viewer), so a run is a pure function of
its inputs and the actor order is the same on every run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class SimClock:
    def __init__(self, dt: float = 0.01) -> None:
        if dt <= 0:
            raise ValueError("tick length must be positive")
        self.dt = dt
        self.tick = 0
        self._actors: list[str] = []
        self._parked: dict[str, asyncio.Future[None]] = {}
        self._all_parked: asyncio.Event | None = None

    @property
    def time(self) -> float:
        return self.tick * self.dt

    def ticks(self, seconds: float) -> int:
        """Number of whole ticks covering `seconds` (at least 1 for positive durations)."""
        return max(1, round(seconds / self.dt)) if seconds > 0 else 0

    def register(self, name: str) -> None:
        if name in self._actors:
            raise ValueError(f"actor {name!r} already registered")
        self._actors.append(name)

    def unregister(self, name: str) -> None:
        self._actors.remove(name)
        fut = self._parked.pop(name, None)
        if fut is not None and not fut.done():
            fut.cancel()
        self._check_parked()

    def _event(self) -> asyncio.Event:
        if self._all_parked is None:
            self._all_parked = asyncio.Event()
        return self._all_parked

    def _check_parked(self) -> None:
        if len(self._parked) == len(self._actors):
            self._event().set()

    async def wait_tick(self, name: str) -> int:
        """Park until the next tick; returns the new tick number."""
        if name not in self._actors:
            raise RuntimeError(f"actor {name!r} is not registered")
        if name in self._parked:
            raise RuntimeError(f"actor {name!r} is already waiting")
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._parked[name] = fut
        self._check_parked()
        await fut
        return self.tick

    async def run(
        self,
        until: Callable[[], bool],
        max_ticks: int,
        pace: float | None = None,
    ) -> bool:
        """Advance ticks until `until()` holds (True) or `max_ticks` is reached (False).

        `until` is evaluated only when every actor has finished the current tick.
        `pace` > 0 sleeps dt / pace of wall time per tick (for a live viewer only).
        """
        event = self._event()
        self._check_parked()
        while True:
            await event.wait()
            if until():
                return True
            if self.tick >= max_ticks:
                return False
            event.clear()
            self.tick += 1
            parked, self._parked = self._parked, {}
            # Wake in registration order so every run interleaves actors identically.
            for name in self._actors:
                parked[name].set_result(None)
            if not self._actors:
                event.set()
            if pace:
                await asyncio.sleep(self.dt / pace)
