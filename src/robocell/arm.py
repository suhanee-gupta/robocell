"""A single simulated robot arm: state machine, motion execution, E-stop and faults.

The arm is driven one tick at a time by `step()`; `run()` wraps that in an asyncio actor
that advances on SimClock ticks. Commands (assign, estop, reset, inject_fault,
clear_fault) take effect immediately on the arm's state, and motion only ever happens
inside `step()`, so an E-stop issued at any point stops the arm before its next tick.

Zone handling follows the protocol in zones.py: all zones of a command are acquired
before the first motion tick and released as soon as the remaining path cannot touch them.
After an abrupt stop the arm keeps the locks for whatever remains of its path; resuming
only ever re-times the *same* joint-space line, so it never needs a lock it does not hold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise

import numpy as np

from robocell import kinematics as kin
from robocell import trajectory as traj
from robocell.clock import SimClock
from robocell.config import ArmConfig
from robocell.kinematics import ArmGeometry, FloatArray
from robocell.zones import SUBSTEPS, ZoneManager, sweep, sweep_margin


class ArmState(Enum):
    IDLE = "idle"
    MOVING = "moving"  # executing a command (including waiting for zone clearance)
    FAULT = "fault"
    ESTOPPED = "estopped"


ALLOWED_TRANSITIONS: dict[ArmState, frozenset[ArmState]] = {
    ArmState.IDLE: frozenset({ArmState.MOVING, ArmState.FAULT, ArmState.ESTOPPED}),
    ArmState.MOVING: frozenset({ArmState.IDLE, ArmState.FAULT, ArmState.ESTOPPED}),
    ArmState.FAULT: frozenset({ArmState.IDLE, ArmState.ESTOPPED}),
    # Reset returns to FAULT if the arm was faulted before the E-stop: only clear_fault clears it.
    ArmState.ESTOPPED: frozenset({ArmState.IDLE, ArmState.FAULT}),
}


# Samples per segment when searching a path for the first pose clear of all zones.
CLEAR_SAMPLES = 400


class InvalidTransitionError(RuntimeError):
    pass


class UnreachableTargetError(ValueError):
    pass


@dataclass
class Plan:
    """Remaining motion of one command: straight joint-space segments, executed in order."""

    segments: list[traj.Trajectory]
    # Per segment: zone index -> last time (s) into that segment it may touch the zone.
    touch: list[dict[int, float]]
    task_id: int | None
    # Index into `segments` whose end completes the task (the arm is then at the target).
    task_segment: int
    k: int = 0  # ticks elapsed in segments[0]
    started: bool = False  # False until motion begins (e.g. while waiting for zones)
    replan: bool = False  # segments[0] must be re-timed from rest before continuing

    @property
    def zones(self) -> set[int]:
        return {z for touch in self.touch for z in touch}

    def still_needs(self, zone: int, t: float) -> bool:
        """Can the path from time t into segments[0] onwards still touch `zone`?"""
        # t is always a sample instant, so any later contact shows up at a sample >= t.
        if self.touch[0].get(zone, -1.0) >= t:
            return True
        return any(zone in touch for touch in self.touch[1:])


@dataclass
class Transition:
    tick: int
    old: ArmState
    new: ArmState


def _noop(arm: Arm, value: int) -> None:
    pass


@dataclass(eq=False)
class Arm:
    cfg: ArmConfig
    dt: float
    on_done: Callable[[Arm, int], None] = _noop
    on_abort: Callable[[Arm, int], None] = _noop
    on_zone_wait: Callable[[Arm, int], None] = _noop  # called each tick blocked on a zone
    zones: ZoneManager = field(default_factory=lambda: ZoneManager([]))
    state: ArmState = field(init=False, default=ArmState.IDLE)
    q: FloatArray = field(init=False)
    plan: Plan | None = field(init=False, default=None)
    tick: int = field(init=False, default=0)
    transitions: list[Transition] = field(init=False, default_factory=list)
    stop_count: int = field(init=False, default=0)  # E-stops + faults (abrupt stops) so far
    fault_latched: bool = field(init=False, default=False)  # set by a fault until clear_fault

    def __post_init__(self) -> None:
        self.geom = ArmGeometry.from_config(self.cfg)
        self.max_vel = np.array(self.cfg.max_vel)
        self.max_acc = np.array(self.cfg.max_acc)
        self.home = np.array(self.cfg.home)
        if not self.geom.within_limits(self.home):
            raise ValueError(f"arm {self.name!r}: home pose violates joint limits")
        self.q = self.home.copy()
        self._occupancy_cache: tuple[bytes, frozenset[int]] = (b"", frozenset())
        self.body_margin = kin.body_gap(self.geom) / 2 + 1e-9
        self.margin = sweep_margin(self.geom, self.max_vel, self.dt / SUBSTEPS)

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def actor_name(self) -> str:
        return f"arm:{self.name}"

    @property
    def available(self) -> bool:
        """Free to take a new task."""
        return self.state is ArmState.IDLE and self.plan is None

    @property
    def task_id(self) -> int | None:
        return self.plan.task_id if self.plan else None

    @property
    def pose(self) -> kin.ArmPose:
        return kin.forward(self.geom, self.q)

    def occupied_zones(self, q: FloatArray | None = None) -> set[int]:
        """Zones the arm's links may be inside at pose q (default: now); conservative, see
        zones.py."""
        pose = self.q if q is None else q
        key = pose.tobytes()
        # Checked every tick by the monitor; arms at rest keep the same pose for long spans.
        if self._occupancy_cache[0] != key:
            occupied = self.zones.occupied(kin.body_points(self.geom, pose), self.body_margin)
            self._occupancy_cache = (key, frozenset(occupied))
        return set(self._occupancy_cache[1])

    @property
    def held_zones(self) -> set[int]:
        return self.zones.held_by(self.name)

    # ---- planning -------------------------------------------------------------------

    def solve(self, target: FloatArray | tuple[float, float, float]) -> kin.IKSolution | None:
        sol = kin.inverse(self.geom, target, yaw_hint=float(self.q[0]))
        return sol if isinstance(sol, kin.IKSolution) else None

    def can_reach(self, target: FloatArray | tuple[float, float, float]) -> bool:
        return isinstance(kin.inverse(self.geom, target), kin.IKSolution)

    def trajectory_to(self, goal: FloatArray, start: FloatArray | None = None) -> traj.Trajectory:
        return traj.plan(self.q if start is None else start, goal, self.max_vel, self.max_acc)

    def estimate(self, target: FloatArray | tuple[float, float, float]) -> float | None:
        """Estimated time (s) to move from the current pose to `target`, None if unreachable."""
        sol = self.solve(target)
        return None if sol is None else self.trajectory_to(sol.q).duration

    def _sweep(self, segment: traj.Trajectory) -> dict[int, float]:
        return sweep(self.geom, segment, self.zones, self.margin, self.dt)

    def _build_plan(self, task_id: int, goal_q: FloatArray) -> Plan:
        segments = [self.trajectory_to(goal_q)]
        # Never come to rest inside a shared zone: that would hold its lock indefinitely.
        if self.occupied_zones(goal_q):
            segments.append(self.trajectory_to(self.home, start=goal_q))
        return Plan(segments, [self._sweep(s) for s in segments], task_id, task_segment=0)

    # ---- commands -------------------------------------------------------------------

    def _transition(self, new: ArmState) -> None:
        if new not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidTransitionError(f"{self.name}: {self.state.value} -> {new.value}")
        self.transitions.append(Transition(self.tick, self.state, new))
        self.state = new

    def assign(self, task_id: int, target: FloatArray | tuple[float, float, float]) -> None:
        if not self.available:
            raise InvalidTransitionError(f"{self.name} is not available ({self.state.value})")
        sol = self.solve(target)
        if sol is None:
            raise UnreachableTargetError(f"{self.name} cannot reach {tuple(target)}")
        self.plan = self._build_plan(task_id, sol.q)
        self._transition(ArmState.MOVING)

    def estop(self) -> None:
        """Halt immediately; motion stays frozen until reset()."""
        if self.state is ArmState.ESTOPPED:
            return
        self._transition(ArmState.ESTOPPED)
        self.stop_count += 1
        self._on_abrupt_stop()

    def reset(self) -> None:
        """Clear an E-stop. A command in progress resumes (re-timed from rest) next tick.
        A fault that was active before the E-stop is still active afterwards."""
        if self.state is not ArmState.ESTOPPED:
            raise InvalidTransitionError(f"{self.name}: reset while {self.state.value}")
        self._transition(ArmState.FAULT if self.fault_latched else ArmState.IDLE)

    def inject_fault(self) -> None:
        """Simulate a drive fault: stop now and hand the current task back via on_abort."""
        if self.state not in (ArmState.IDLE, ArmState.MOVING):
            raise InvalidTransitionError(f"{self.name}: fault while {self.state.value}")
        self._transition(ArmState.FAULT)
        self.fault_latched = True
        self.stop_count += 1
        self._on_abrupt_stop()
        if self.plan is None:
            return
        task_id = self.plan.task_id
        self.plan.task_id = None
        self._abandon_plan()
        if task_id is not None:
            self.on_abort(self, task_id)

    def clear_fault(self) -> None:
        if self.state is not ArmState.FAULT:
            raise InvalidTransitionError(f"{self.name}: clear_fault while {self.state.value}")
        self.fault_latched = False
        self._transition(ArmState.IDLE)

    def _on_abrupt_stop(self) -> None:
        if self.plan is None:
            return
        if self.plan.started:
            self.plan.replan = True
        else:
            # Still waiting for zones and outside all of them: give up locks and queue spots
            # so a stopped arm cannot block others; they are re-acquired on resume.
            self.zones.release_all(self.name)

    def _abandon_plan(self) -> None:
        """After a fault the task is gone. If the arm stopped outside every zone, drop the
        motion. If it stopped inside one, keep the locks that path needs and shorten it to
        the first pose along it that is clear of all zones: once the fault is cleared it
        moves only that far. (If the abandoned target is itself inside a zone, that first
        clear pose lies on the retreat leg, so the arm passes the target, still under lock.)"""
        plan = self.plan
        assert plan is not None
        if not plan.started or not self.occupied_zones():
            self.zones.release_all(self.name)
            self.plan = None
            return
        waypoints = self._clearing_waypoints([self.q, *(seg.goal for seg in plan.segments)])
        held = self.held_zones
        plan.segments = [self.trajectory_to(b, start=a) for a, b in pairwise(waypoints)]
        plan.touch = [{z: t for z, t in self._sweep(g).items() if z in held} for g in plan.segments]
        # Locks for the abandoned rest of the path would only block other arms meanwhile.
        for zone in held - plan.zones - self.occupied_zones():
            self.zones.release(zone, self.name)
        plan.task_segment = -1
        plan.k = 0
        plan.replan = False  # the new segments already start from rest at the current pose

    def _clearing_waypoints(self, path: list[FloatArray]) -> list[FloatArray]:
        """Truncate a joint-space polyline at its first pose clear of every zone."""
        for i, (a, b) in enumerate(pairwise(path)):
            s = np.linspace(0.0, 1.0, CLEAR_SAMPLES + 1)[1:, None]
            qs = a + s * (b - a)
            bodies = kin.body_points_batch(self.geom, qs)
            clear = ~self.zones.occupied_mask(bodies, self.body_margin)
            if np.any(clear):
                return [*path[: i + 1], qs[int(np.argmax(clear))]]
        raise RuntimeError(f"{self.name}: planned path never leaves the shared zones")

    # ---- execution ------------------------------------------------------------------

    def step(self, tick: int) -> None:
        """Advance one tick."""
        self.tick = tick
        if self.state in (ArmState.FAULT, ArmState.ESTOPPED) or self.plan is None:
            return
        if self.state is ArmState.IDLE:
            self._transition(ArmState.MOVING)
        plan = self.plan
        if not plan.started:
            blocked = self.zones.acquire_in_order(self.name, plan.zones)
            if blocked is not None:
                self.on_zone_wait(self, blocked)
                return
            plan.started = True
        if plan.replan:
            self._replan(plan)
        self._advance(plan)

    def _replan(self, plan: Plan) -> None:
        """Re-time segments[0] from rest at the current pose after an abrupt stop.

        The arm stopped on the straight joint-space line of segments[0], so the new
        segment is the rest of that same line: it can only touch zones the original
        path touched and that are still held.
        """
        held = self.held_zones
        seg = self.trajectory_to(plan.segments[0].goal)
        plan.segments[0] = seg
        plan.touch[0] = {z: t for z, t in self._sweep(seg).items() if z in held}
        plan.k = 0
        plan.replan = False

    def _advance(self, plan: Plan) -> None:
        seg = plan.segments[0]
        plan.k += 1
        t = plan.k * self.dt
        finished = t >= seg.duration
        self.q = seg.position(t)
        for zone in self.held_zones:
            if not plan.still_needs(zone, t):
                self.zones.release(zone, self.name)
        if not finished:
            return
        if plan.task_id is not None and plan.task_segment == 0:
            task_id, plan.task_id = plan.task_id, None
            self.on_done(self, task_id)
        plan.segments.pop(0)
        plan.touch.pop(0)
        plan.task_segment -= 1
        plan.k = 0
        if not plan.segments:
            self.zones.release_all(self.name)
            self.plan = None
            self._transition(ArmState.IDLE)

    async def run(self, clock: SimClock) -> None:
        """Actor loop: one step per clock tick, forever (cancelled by the owner)."""
        while True:
            tick = await clock.wait_tick(self.actor_name)
            self.step(tick)
