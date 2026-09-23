"""Task queue and arm assignment.

Policy:
- Tasks are ordered by priority (higher first), then FIFO by task id (= arrival order).
  A requeued task keeps its original place in that order.
- On arrival a task is rejected outright if no arm can reach its target.
- Each tick, tasks are considered in queue order; a task goes to the idle arm that can
  reach it with the lowest estimated move time (ties: lower arm index). A task whose
  capable arms are all busy waits, and lower-priority tasks may still use arms it cannot.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

from robocell.arm import Arm
from robocell.metrics import Metrics


@dataclass(frozen=True)
class Task:
    id: int
    target: tuple[float, float, float]
    priority: int = 0
    arrival_tick: int = 0

    @property
    def order(self) -> tuple[int, int]:
        return (-self.priority, self.id)


@dataclass(frozen=True)
class Assignment:
    task: Task
    arm: Arm
    tick: int


class Scheduler:
    def __init__(self, arms: Sequence[Arm], metrics: Metrics) -> None:
        self.arms = list(arms)
        self.metrics = metrics
        self._queue: list[tuple[tuple[int, int], Task]] = []
        self._capable: dict[int, list[Arm]] = {}
        self.tasks: dict[int, Task] = {}
        self.running: dict[int, Arm] = {}
        self.assignments: list[Assignment] = []

    @property
    def queued(self) -> list[Task]:
        return [t for _, t in self._queue]

    @property
    def idle(self) -> bool:
        """No task waiting or in progress."""
        return not self._queue and not self.running

    def capable_arms(self, task_id: int) -> list[Arm]:
        return list(self._capable[task_id])

    def _enqueue(self, task: Task) -> None:
        bisect.insort(self._queue, (task.order, task), key=lambda e: e[0])

    def submit(self, task: Task, tick: int) -> bool:
        """Accept a new task (True) or reject it because no arm can reach it (False)."""
        if task.id in self.tasks:
            raise ValueError(f"duplicate task id {task.id}")
        self.tasks[task.id] = task
        capable = [a for a in self.arms if a.can_reach(task.target)]
        if not capable:
            self.metrics.task_rejected()
            return False
        self._capable[task.id] = capable
        self.metrics.task_queued(task.id, tick)
        self._enqueue(task)
        return True

    def requeue(self, task_id: int, tick: int) -> None:
        """Put a task whose arm failed back in the queue (at its original position)."""
        self.running.pop(task_id)
        self.metrics.task_queued(task_id, tick, requeue=True)
        self._enqueue(self.tasks[task_id])

    def complete(self, task_id: int, arm: Arm, error: float) -> None:
        owner = self.running.pop(task_id)
        if owner is not arm:
            raise RuntimeError(f"task {task_id} completed by {arm.name}, owned by {owner.name}")
        self.metrics.task_completed(task_id, arm.name, error)

    def dispatch(self, tick: int) -> list[Assignment]:
        made: list[Assignment] = []
        if not self._queue or not any(a.available for a in self.arms):
            return made
        waiting: list[tuple[tuple[int, int], Task]] = []
        for entry in self._queue:
            task = entry[1]
            best: tuple[float, int, Arm] | None = None
            for idx, arm in enumerate(self._capable[task.id]):
                if not arm.available:
                    continue
                est = arm.estimate(task.target)
                if est is None:  # capable_arms is computed with IK, so this cannot happen
                    raise RuntimeError(f"{arm.name} lost reach to task {task.id}")
                if best is None or (est, idx) < best[:2]:
                    best = (est, idx, arm)
            if best is None:
                waiting.append(entry)
                continue
            arm = best[2]
            arm.assign(task.id, task.target)
            self.running[task.id] = arm
            self.metrics.task_assigned(task.id, tick)
            made.append(Assignment(task, arm, tick))
        self._queue = waiting
        self.assignments.extend(made)
        return made
