"""Shared zones: axis-aligned boxes that at most one arm may occupy at a time.

Locking protocol (deadlock-free by construction):
1. Before a command starts moving, the arm computes every zone its whole path may touch
   (`sweep`) and acquires those locks one by one in a single global order (zone index,
   i.e. sorted by name). It never moves while any of them is missing.
2. Each lock has a FIFO queue; an arm waits only on the lowest-index lock it still needs
   and holds only lower-index locks while waiting. A wait-for cycle would need a lock
   held by someone waiting on a *lower* index, which the ordering rules out.
3. A lock is released as soon as the remaining path can no longer touch that zone, and all
   locks are released when the command ends. Arms at rest are never inside a zone, so an
   idle arm never holds a lock.
4. A moving arm never waits for a lock (it already holds its whole path), so every holder
   eventually releases unless it is E-stopped, which is an operator decision.

"Occupies" means some part of the arm's links may be inside the box: a body sample point
(shoulder, points along both links, elbow, tip; see kinematics.body_points_batch) lies
within half the point spacing of it. Every point of a link is that close to a sample, so
the test is conservative for the continuous links, at rest as well as in motion.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from robocell.config import ZoneConfig
from robocell.kinematics import ArmGeometry, FloatArray, body_gap, body_points_batch
from robocell.trajectory import Trajectory

# Path samples per sim tick when sweeping a trajectory for zone contact.
SUBSTEPS = 4


@dataclass(frozen=True)
class Zone:
    name: str
    lo: FloatArray
    hi: FloatArray

    @classmethod
    def from_config(cls, cfg: ZoneConfig) -> Zone:
        return cls(cfg.name, np.array(cfg.min), np.array(cfg.max))

    def contains(self, points: FloatArray, margin: float = 0.0) -> NDArray[np.bool_]:
        """Which of the (N, 3) points lie inside the box grown by `margin` on every side."""
        pts = np.atleast_2d(points)
        inside: NDArray[np.bool_] = np.all(
            (pts >= self.lo - margin) & (pts <= self.hi + margin), axis=1
        )
        return inside


@dataclass
class ZoneLock:
    holder: str | None = None
    queue: deque[str] = field(default_factory=deque)


class ZoneManager:
    """Owns the zone locks. Zone index = position in name order = global lock order."""

    def __init__(self, zones: Sequence[ZoneConfig]) -> None:
        self.zones = [Zone.from_config(z) for z in sorted(zones, key=lambda z: z.name)]
        self._locks = [ZoneLock() for _ in self.zones]
        self._lo = np.array([z.lo for z in self.zones]).reshape(-1, 3)
        self._hi = np.array([z.hi for z in self.zones]).reshape(-1, 3)

    def __len__(self) -> int:
        return len(self.zones)

    @property
    def names(self) -> list[str]:
        return [z.name for z in self.zones]

    def holder(self, zone: int) -> str | None:
        return self._locks[zone].holder

    def waiting(self, zone: int) -> list[str]:
        return list(self._locks[zone].queue)

    def held_by(self, who: str) -> set[int]:
        return {i for i, lock in enumerate(self._locks) if lock.holder == who}

    def _try(self, zone: int, who: str) -> bool:
        lock = self._locks[zone]
        if lock.holder == who:
            return True
        if who not in lock.queue:
            lock.queue.append(who)
        if lock.holder is None and lock.queue[0] == who:
            lock.queue.popleft()
            lock.holder = who
            return True
        return False

    def acquire_in_order(self, who: str, needed: Iterable[int]) -> int | None:
        """Acquire `needed` locks in global order; returns the zone blocking progress, if any."""
        for zone in sorted(set(needed)):
            if not self._try(zone, who):
                return zone
        return None

    def release(self, zone: int, who: str) -> None:
        lock = self._locks[zone]
        if lock.holder != who:
            raise RuntimeError(f"{who} released zone {self.zones[zone].name} held by {lock.holder}")
        lock.holder = None

    def release_all(self, who: str) -> None:
        """Release every lock `who` holds and leave every queue it is waiting in."""
        for lock in self._locks:
            if lock.holder == who:
                lock.holder = None
            if who in lock.queue:
                lock.queue.remove(who)

    def occupied_mask(self, bodies: FloatArray, margin: float = 0.0) -> NDArray[np.bool_]:
        """For (N, P, 3) sets of body points: which of the N sets touch any zone grown by
        `margin`."""
        pts = bodies[:, :, None, :]  # (N, P, 1, 3) against (Z, 3) boxes
        inside = np.all((pts >= self._lo - margin) & (pts <= self._hi + margin), axis=3)
        mask: NDArray[np.bool_] = inside.any(axis=(1, 2))
        return mask

    def occupied(self, points: FloatArray, margin: float = 0.0) -> set[int]:
        """Zones containing any of the given points, boxes grown by `margin`."""
        pts = np.atleast_2d(points)[:, None, :]  # (P, 1, 3) against (Z, 3) boxes
        inside = np.all((pts >= self._lo - margin) & (pts <= self._hi + margin), axis=2)
        return {int(i) for i in np.flatnonzero(inside.any(axis=0))}


def sweep_margin(geom: ArmGeometry, max_vel: FloatArray, sample_dt: float) -> float:
    """Box growth that makes sampled zone checks conservative for the continuous arm body.

    Time: each joint rotates a body point about an axis at most l1 + l2 away, so
    |dp| <= (l1 + l2) * sum_i |dq_i| <= (l1 + l2) * sum_i vmax_i * dt, and any instant is
    within sample_dt / 2 of a sample.
    Space: any point of a link is within body_gap / 2 of a body sample point.
    """
    time_part = (geom.l1 + geom.l2) * float(np.sum(max_vel)) * sample_dt / 2
    return time_part + body_gap(geom) / 2 + 1e-9


def sweep(
    geom: ArmGeometry,
    segment: Trajectory,
    zones: ZoneManager,
    margin: float,
    dt: float,
) -> dict[int, float]:
    """Zones the segment may touch -> last sampled time (s) it may touch each.

    Samples every dt / SUBSTEPS (which includes every tick instant k * dt, where the
    arm is actually evaluated) plus the end point, and tests the body points against the
    zones grown by `margin`.
    """
    if not len(zones):
        return {}
    n = int(np.ceil(segment.duration * SUBSTEPS / dt))
    times = np.minimum((np.arange(n + 1) / SUBSTEPS) * dt, segment.duration)
    body = body_points_batch(geom, segment.positions(times))  # (N, P, 3)
    flat = body.reshape(-1, 3)
    touch: dict[int, float] = {}
    for i, zone in enumerate(zones.zones):
        hit = zone.contains(flat, margin).reshape(body.shape[:2]).any(axis=1)
        if np.any(hit):
            touch[i] = float(times[np.flatnonzero(hit)[-1]])
    return touch
