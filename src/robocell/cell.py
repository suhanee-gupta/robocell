"""The factory cell: arms, scheduler and safety monitor wired to one SimClock.

Actor order within every tick (fixed, so runs are deterministic):
  1. control  - scheduled events (E-stop presses/resets, fault injection/recovery), then
                admit arriving tasks and dispatch queued tasks to idle arms
  2. arms     - each arm advances one tick, in config order
  3. monitor  - checks safety invariants on the resulting state and samples metrics
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from robocell.arm import Arm, ArmState
from robocell.clock import SimClock
from robocell.config import CellConfig, ConfigError, TaskGenConfig
from robocell.kinematics import FloatArray, tip_position
from robocell.metrics import Metrics
from robocell.scheduler import Assignment, Scheduler, Task
from robocell.zones import ZoneManager

# Floating point slack for the per-tick limit checks (not modelling slack).
LIMIT_RTOL = 1e-9
ACC_ATOL = 1e-6


def generate_tasks(cfg: TaskGenConfig, n: int, rng: np.random.Generator, dt: float) -> list[Task]:
    """n tasks with Poisson arrivals and uniformly random targets (some unreachable)."""
    gaps = rng.exponential(1.0 / cfg.arrival_rate_hz, size=n)
    arrivals = np.cumsum(gaps)
    lo = np.array([cfg.x[0], cfg.y[0], cfg.z[0]])
    hi = np.array([cfg.x[1], cfg.y[1], cfg.z[1]])
    targets = rng.uniform(lo, hi, size=(n, 3))
    weights = np.array(cfg.priority_weights) / sum(cfg.priority_weights)
    priorities = rng.choice(len(weights), size=n, p=weights)
    return [
        Task(
            id=i,
            target=(float(targets[i, 0]), float(targets[i, 1]), float(targets[i, 2])),
            priority=int(priorities[i]),
            arrival_tick=max(1, int(np.ceil(arrivals[i] / dt))),
        )
        for i in range(n)
    ]


@dataclass
class RunResult:
    finished: bool
    metrics: Metrics


class Cell:
    def __init__(
        self, config: CellConfig, tasks: Sequence[Task], rng: np.random.Generator | None = None
    ) -> None:
        self.config = config
        self.clock = SimClock(config.sim.tick_s)
        dt = self.clock.dt
        self.zones = ZoneManager(config.zones)
        self.arms = [Arm(cfg, dt, zones=self.zones) for cfg in config.arms]
        for arm in self.arms:
            if inside := arm.occupied_zones(arm.home):
                names = [self.zones.zones[z].name for z in sorted(inside)]
                raise ConfigError(f"arm {arm.name!r}: home pose is inside shared zone(s) {names}")
        self.metrics = Metrics(
            dt=dt, arm_names=[a.name for a in self.arms], zone_names=self.zones.names
        )
        self.scheduler = Scheduler(self.arms, self.metrics)
        for arm in self.arms:
            arm.on_done = self._on_done
            arm.on_abort = self._on_abort
            arm.on_zone_wait = self._on_zone_wait
        self.tasks = sorted(tasks, key=lambda t: (t.arrival_tick, t.id))
        self.metrics.generated = len(self.tasks)
        self._next_task = 0
        self._events: dict[int, list[Callable[[], None]]] = {}
        # Called at the end of every tick with the settled state (e.g. the viewer's recorder).
        self.observers: list[Callable[[Cell, int], None]] = []
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # Monitor state: last two joint samples and the stop count seen with them.
        self._history: dict[str, list[tuple[FloatArray, int]]] = {a.name: [] for a in self.arms}
        self._estop_reset_due = 0
        for at_s in config.estop.global_at_s:
            self.at(max(1, self.clock.ticks(at_s)), self._global_estop_press)

    # ---- arm callbacks ------------------------------------------------------------

    def _on_done(self, arm: Arm, task_id: int) -> None:
        target = np.array(self.scheduler.tasks[task_id].target)
        error = float(np.linalg.norm(tip_position(arm.geom, arm.q) - target))
        self.scheduler.complete(task_id, arm, error)

    def _on_abort(self, arm: Arm, task_id: int) -> None:
        self.scheduler.requeue(task_id, self.clock.tick)

    def _on_zone_wait(self, arm: Arm, zone: int) -> None:
        self.metrics.zone_wait(self.zones.zones[zone].name)

    # ---- events -------------------------------------------------------------------

    def at(self, tick: int, action: Callable[[], None]) -> None:
        """Run `action` at the start of `tick` (before arrivals and dispatch)."""
        if tick <= self.clock.tick:
            raise ValueError(f"tick {tick} is not in the future")
        self._events.setdefault(tick, []).append(action)

    # ---- E-stop and faults ----------------------------------------------------------

    def estop_all(self) -> None:
        """Global E-stop: every arm halts before its next tick."""
        self.metrics.estops += 1
        for arm in self.arms:
            arm.estop()

    def reset_all(self) -> None:
        for arm in self.arms:
            if arm.state is ArmState.ESTOPPED:
                arm.reset()

    def _global_estop_press(self) -> None:
        self.estop_all()
        # Overlapping presses: only the reset due after the *latest* press may release arms.
        due = self.clock.tick + self.clock.ticks(self.config.estop.reset_after_s)
        self._estop_reset_due = max(self._estop_reset_due, due)
        self.at(due, self._global_reset_if_due)

    def _global_reset_if_due(self) -> None:
        if self.clock.tick >= self._estop_reset_due:
            self.reset_all()

    def _plan_faults(self, assignments: Sequence[Assignment]) -> None:
        """Decide (seeded) whether each new execution faults, and when."""
        for a in assignments:
            # Always draw both numbers so the random stream does not depend on outcomes.
            roll, when = self.rng.random(), self.rng.random()
            if roll >= self.config.faults.probability:
                continue
            assert a.arm.plan is not None
            span = max(1, self.clock.ticks(a.arm.plan.segments[0].duration))
            tick = a.tick + 1 + int(when * span)
            self.at(tick, functools.partial(self._fault, a.arm, a.task.id))

    def _fault(self, arm: Arm, task_id: int) -> None:
        if arm.task_id != task_id or arm.state is not ArmState.MOVING:
            return  # finished, E-stopped or already faulted meanwhile: nothing to break
        self.metrics.faults += 1
        arm.inject_fault()  # hands the task back through _on_abort -> requeue
        self._schedule_recovery(arm, self.clock.ticks(self.config.faults.recovery_s))

    def _schedule_recovery(self, arm: Arm, after: int) -> None:
        self.at(self.clock.tick + after, functools.partial(self._recover, arm))

    def _recover(self, arm: Arm) -> None:
        if arm.state is ArmState.FAULT:
            arm.clear_fault()
        elif arm.fault_latched:  # E-stopped while faulted: clear once the E-stop is reset
            self._schedule_recovery(arm, 1)

    # ---- actors -------------------------------------------------------------------

    def control_step(self, tick: int) -> None:
        for action in self._events.pop(tick, []):
            action()
        while (
            self._next_task < len(self.tasks) and self.tasks[self._next_task].arrival_tick <= tick
        ):
            self.scheduler.submit(self.tasks[self._next_task], tick)
            self._next_task += 1
        self._plan_faults(self.scheduler.dispatch(tick))

    def zone_violations(self) -> int:
        """Safety invariant: each shared zone holds at most one arm, and only its lock holder.
        Also flags locks held by arms at rest (a leaked lock would block others forever)."""
        occupants: dict[int, list[str]] = {}
        violations = 0
        for arm in self.arms:
            for zone in arm.occupied_zones():
                occupants.setdefault(zone, []).append(arm.name)
            if arm.available and arm.held_zones:
                violations += 1
        for zone, names in occupants.items():
            if len(names) > 1 or self.zones.holder(zone) != names[0]:
                violations += 1
        return violations

    def monitor_step(self, tick: int) -> None:
        self._check(tick)
        for observer in self.observers:
            observer(self, tick)

    def _check(self, tick: int) -> None:
        self.metrics.ticks = tick
        self.metrics.zone_violations += self.zone_violations()
        dt = self.clock.dt
        for arm in self.arms:
            if arm.state is ArmState.MOVING:
                self.metrics.arm_busy(arm.name)
            if not arm.geom.within_limits(arm.q):
                self.metrics.limit_violations += 1
            hist = self._history[arm.name]
            hist.append((arm.q.copy(), arm.stop_count))
            if len(hist) > 3:
                hist.pop(0)
            if len(hist) >= 2:
                vel = np.abs(hist[-1][0] - hist[-2][0]) / dt
                if np.any(vel > arm.max_vel * (1 + LIMIT_RTOL)):
                    self.metrics.limit_violations += 1
            # An E-stop or fault is a deliberate instantaneous stop: skip the acceleration
            # check for samples that straddle one.
            if len(hist) == 3 and hist[0][1] == hist[2][1]:
                acc = np.abs(hist[2][0] - 2 * hist[1][0] + hist[0][0]) / (dt * dt)
                if np.any(acc > arm.max_acc * (1 + LIMIT_RTOL) + ACC_ATOL):
                    self.metrics.limit_violations += 1

    def done(self) -> bool:
        """All tasks resolved and every arm at rest. Pending events (e.g. a later E-stop
        press) do not keep the run alive; recoveries do, via the arms' availability."""
        return (
            self._next_task == len(self.tasks)
            and self.scheduler.idle
            and all(a.available for a in self.arms)
        )

    async def _loop(self, name: str, step: Callable[[int], None]) -> None:
        while True:
            tick = await self.clock.wait_tick(name)
            step(tick)

    async def run_async(self, pace: float | None = None) -> RunResult:
        clock = self.clock
        clock.register("control")
        for arm in self.arms:
            clock.register(arm.actor_name)
        clock.register("monitor")
        max_ticks = round(self.config.sim.max_time_s / clock.dt)
        async with asyncio.TaskGroup() as tg:
            actors = [tg.create_task(self._loop("control", self.control_step))]
            actors += [tg.create_task(arm.run(clock)) for arm in self.arms]
            actors.append(tg.create_task(self._loop("monitor", self.monitor_step)))
            finished = await clock.run(self.done, max_ticks, pace=pace)
            for actor in actors:
                actor.cancel()
        return RunResult(finished=finished, metrics=self.metrics)

    def run(self, pace: float | None = None) -> RunResult:
        return asyncio.run(self.run_async(pace))


def build_cell(config: CellConfig, n_tasks: int, seed: int) -> Cell:
    """A cell with `n_tasks` generated from `seed`; running it is fully deterministic."""
    task_seed, cell_seed = np.random.SeedSequence(seed).spawn(2)
    tasks = generate_tasks(
        config.tasks, n_tasks, np.random.default_rng(task_seed), config.sim.tick_s
    )
    return Cell(config, tasks, np.random.default_rng(cell_seed))


def simulate(config: CellConfig, n_tasks: int, seed: int, pace: float | None = None) -> RunResult:
    return build_cell(config, n_tasks, seed).run(pace)
