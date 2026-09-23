"""A single simulated robot arm: state machine, motion execution, E-stop and faults.

The arm is driven one tick at a time by `step()`; `run()` wraps that in an asyncio actor
that advances on SimClock ticks. Commands (assign, estop, reset, inject_fault,
clear_fault) take effect immediately on the arm's state, and motion only ever happens
inside `step()`, so an E-stop issued at any point stops the arm before its next tick.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from robocell import kinematics as kin
from robocell import trajectory as traj
from robocell.clock import SimClock
from robocell.config import ArmConfig
from robocell.kinematics import ArmGeometry, FloatArray


class ArmState(Enum):
    IDLE = "idle"
    MOVING = "moving"  # executing a command (including waiting for zone clearance)
    FAULT = "fault"
    ESTOPPED = "estopped"


ALLOWED_TRANSITIONS: dict[ArmState, frozenset[ArmState]] = {
    ArmState.IDLE: frozenset({ArmState.MOVING, ArmState.FAULT, ArmState.ESTOPPED}),
    ArmState.MOVING: frozenset({ArmState.IDLE, ArmState.FAULT, ArmState.ESTOPPED}),
    ArmState.FAULT: frozenset({ArmState.IDLE, ArmState.ESTOPPED}),
    ArmState.ESTOPPED: frozenset({ArmState.IDLE}),
}


class InvalidTransitionError(RuntimeError):
    pass


class UnreachableTargetError(ValueError):
    pass


@dataclass
class Plan:
    """Remaining motion of one command: straight joint-space segments, executed in order."""

    segments: list[traj.Trajectory]
    task_id: int | None
    # Index into `segments` whose end completes the task (the arm is then at the target).
    task_segment: int
    k: int = 0  # ticks elapsed in segments[0]
    started: bool = False  # False until motion begins (e.g. while waiting for zones)
    replan: bool = False  # segments[0] must be re-timed from rest before continuing


@dataclass
class Transition:
    tick: int
    old: ArmState
    new: ArmState


def _noop(arm: Arm, task_id: int) -> None:
    pass


@dataclass(eq=False)
class Arm:
    cfg: ArmConfig
    dt: float
    on_done: Callable[[Arm, int], None] = _noop
    on_abort: Callable[[Arm, int], None] = _noop
    state: ArmState = field(init=False, default=ArmState.IDLE)
    q: FloatArray = field(init=False)
    plan: Plan | None = field(init=False, default=None)
    tick: int = field(init=False, default=0)
    transitions: list[Transition] = field(init=False, default_factory=list)
    stop_count: int = field(init=False, default=0)  # E-stops + faults (abrupt stops) so far

    def __post_init__(self) -> None:
        self.geom = ArmGeometry.from_config(self.cfg)
        self.max_vel = np.array(self.cfg.max_vel)
        self.max_acc = np.array(self.cfg.max_acc)
        self.home = np.array(self.cfg.home)
        self.q = self.home.copy()

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

    def _build_segments(self, goal_q: FloatArray) -> list[traj.Trajectory]:
        return [self.trajectory_to(goal_q)]

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
        self.plan = Plan(self._build_segments(sol.q), task_id=task_id, task_segment=0)
        self._transition(ArmState.MOVING)

    def estop(self) -> None:
        """Halt immediately; motion stays frozen until reset()."""
        if self.state is ArmState.ESTOPPED:
            return
        self._transition(ArmState.ESTOPPED)
        self.stop_count += 1
        self._on_abrupt_stop()

    def reset(self) -> None:
        """Clear an E-stop. A command in progress resumes (re-timed from rest) next tick."""
        if self.state is not ArmState.ESTOPPED:
            raise InvalidTransitionError(f"{self.name}: reset while {self.state.value}")
        self._transition(ArmState.IDLE)

    def inject_fault(self) -> None:
        """Simulate a drive fault: stop now and hand the current task back via on_abort."""
        if self.state not in (ArmState.IDLE, ArmState.MOVING):
            raise InvalidTransitionError(f"{self.name}: fault while {self.state.value}")
        self._transition(ArmState.FAULT)
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
        self._transition(ArmState.IDLE)

    def _on_abrupt_stop(self) -> None:
        if self.plan is not None and self.plan.started:
            self.plan.replan = True

    def _abandon_plan(self) -> None:
        """After a fault the task is gone; drop the remaining motion."""
        self.plan = None

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
            plan.started = True
        if plan.replan:
            seg = plan.segments[0]
            plan.segments[0] = self.trajectory_to(seg.goal)
            plan.k = 0
            plan.replan = False
        self._advance(plan)

    def _advance(self, plan: Plan) -> None:
        seg = plan.segments[0]
        plan.k += 1
        t = plan.k * self.dt
        finished = t >= seg.duration
        self.q = seg.position(t)
        if not finished:
            return
        if plan.task_id is not None and plan.task_segment == 0:
            task_id, plan.task_id = plan.task_id, None
            self.on_done(self, task_id)
        plan.segments.pop(0)
        plan.task_segment -= 1
        plan.k = 0
        if not plan.segments:
            self.plan = None
            self._transition(ArmState.IDLE)

    async def run(self, clock: SimClock) -> None:
        """Actor loop: one step per clock tick, forever (cancelled by the owner)."""
        while True:
            tick = await clock.wait_tick(self.actor_name)
            self.step(tick)
