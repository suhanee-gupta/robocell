"""Run metrics: task throughput, wait times, arm utilisation, zone waits, safety violations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Metrics:
    dt: float
    arm_names: list[str]
    zone_names: list[str]
    generated: int = 0
    completed: int = 0
    rejected: int = 0
    requeued: int = 0
    faults: int = 0
    estops: int = 0
    ticks: int = 0
    zone_violations: int = 0
    limit_violations: int = 0
    max_target_error: float = 0.0
    # Ticks each completed task spent queued (summed over requeues).
    wait_ticks: list[int] = field(default_factory=list)
    busy_ticks: dict[str, int] = field(init=False)
    zone_wait_ticks: dict[str, int] = field(init=False)
    completed_by: dict[str, int] = field(init=False)
    _queued_since: dict[int, int] = field(default_factory=dict)
    _waited: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.busy_ticks = dict.fromkeys(self.arm_names, 0)
        self.completed_by = dict.fromkeys(self.arm_names, 0)
        self.zone_wait_ticks = dict.fromkeys(self.zone_names, 0)

    # ---- task lifecycle -----------------------------------------------------------

    def task_queued(self, task_id: int, tick: int, *, requeue: bool = False) -> None:
        if requeue:
            self.requeued += 1
        self._queued_since[task_id] = tick

    def task_assigned(self, task_id: int, tick: int) -> None:
        since = self._queued_since.pop(task_id)
        self._waited[task_id] = self._waited.get(task_id, 0) + tick - since

    def task_completed(self, task_id: int, arm: str, error: float) -> None:
        self.completed += 1
        self.completed_by[arm] += 1
        self.wait_ticks.append(self._waited.pop(task_id, 0))
        self.max_target_error = max(self.max_target_error, error)

    def task_rejected(self) -> None:
        self.rejected += 1

    # ---- per-tick samples ---------------------------------------------------------

    def arm_busy(self, arm: str) -> None:
        self.busy_ticks[arm] += 1

    def zone_wait(self, zone: str) -> None:
        self.zone_wait_ticks[zone] += 1

    # ---- report -------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        waits = np.array(self.wait_ticks, dtype=np.float64) * self.dt
        ticks = max(self.ticks, 1)

        def r(x: float) -> float:
            return round(float(x), 6)

        return {
            "tasks_generated": self.generated,
            "tasks_completed": self.completed,
            "tasks_rejected": self.rejected,
            "tasks_requeued": self.requeued,
            "faults": self.faults,
            "estops": self.estops,
            "wait_time_mean_s": r(waits.mean()) if len(waits) else 0.0,
            "wait_time_p95_s": r(np.percentile(waits, 95)) if len(waits) else 0.0,
            "arm_utilization": {a: r(t / ticks) for a, t in self.busy_ticks.items()},
            "tasks_by_arm": dict(self.completed_by),
            "zone_wait_time_s": {z: r(t * self.dt) for z, t in self.zone_wait_ticks.items()},
            "zone_wait_time_total_s": r(sum(self.zone_wait_ticks.values()) * self.dt),
            "zone_violations": self.zone_violations,
            "limit_violations": self.limit_violations,
            "max_target_error_m": float(f"{self.max_target_error:.3e}"),
            "sim_time_s": r(self.ticks * self.dt),
            "ticks": self.ticks,
        }
