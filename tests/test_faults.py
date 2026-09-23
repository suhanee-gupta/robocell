import dataclasses

import numpy as np
import pytest
from helpers import two_arm_config

from robocell.arm import ArmState
from robocell.cell import Cell, generate_tasks, simulate
from robocell.config import CellConfig, EStopConfig, FaultConfig
from robocell.metrics import Metrics
from robocell.scheduler import Task

BOTH_REACH = (0.6, 0.2, 0.5)  # both arms of two_arm_config can reach this; a is closer


def busy_cell(cfg: CellConfig, n: int = 60, seed: int = 4) -> Cell:
    tasks = generate_tasks(cfg.tasks, n, np.random.default_rng(seed), cfg.sim.tick_s)
    return Cell(cfg, tasks)


def test_global_estop_stops_every_arm_within_one_tick(default_config: CellConfig) -> None:
    cfg = dataclasses.replace(default_config, faults=FaultConfig(), estop=EStopConfig())
    cell = busy_cell(cfg)
    frozen: dict[str, np.ndarray] = {}

    def press() -> None:
        assert sum(a.state is ArmState.MOVING for a in cell.arms) >= 2
        frozen.update({a.name: a.q.copy() for a in cell.arms})
        cell.estop_all()

    def check() -> None:
        for arm in cell.arms:
            assert arm.state is ArmState.ESTOPPED
            assert np.array_equal(arm.q, frozen[arm.name]), arm.name

    cell.at(1500, press)
    for t in (1501, 1502, 1600):
        cell.at(t, check)
    cell.at(1601, cell.reset_all)
    result = cell.run()
    m = result.metrics
    assert result.finished
    assert m.estops == 1
    assert m.completed + m.rejected == m.generated
    assert m.zone_violations == 0
    assert m.limit_violations == 0


def test_single_arm_estop_leaves_others_running(default_config: CellConfig) -> None:
    cfg = dataclasses.replace(default_config, faults=FaultConfig(), estop=EStopConfig())
    cell = busy_cell(cfg)
    before: dict[str, np.ndarray] = {}
    stopped = cell.arms[0]

    def press() -> None:
        before.update({a.name: a.q.copy() for a in cell.arms})
        stopped.estop()

    def check() -> None:
        assert np.array_equal(stopped.q, before[stopped.name])
        moved = [a for a in cell.arms[1:] if not np.array_equal(a.q, before[a.name])]
        assert moved, "other arms should keep working"

    cell.at(1500, press)
    cell.at(1550, check)
    cell.at(1800, stopped.reset)
    result = cell.run()
    assert result.finished
    assert result.metrics.completed + result.metrics.rejected == result.metrics.generated


def test_faulted_arms_task_is_requeued_and_done_by_another_arm() -> None:
    cell = Cell(two_arm_config(), [Task(0, BOTH_REACH, arrival_tick=1)])
    a = cell.arms[0]
    cell.at(30, a.inject_fault)
    cell.at(5000, a.clear_fault)  # a stays faulted long after b has finished
    result = cell.run()
    m = result.metrics
    assert result.finished
    assert [(x.task.id, x.arm.name) for x in cell.scheduler.assignments] == [(0, "a"), (0, "b")]
    assert m.requeued == 1
    assert m.completed == 1
    assert m.completed_by == {"a": 0, "b": 1}
    assert m.max_target_error < 1e-9


def test_task_only_faulted_arm_can_reach_waits_for_recovery() -> None:
    only_a = (-0.5, 0.3, 0.6)
    cell = Cell(two_arm_config(), [Task(0, only_a, arrival_tick=1)])
    a, b = cell.arms
    assert not b.can_reach(only_a)
    cell.at(30, a.inject_fault)
    cell.at(300, a.clear_fault)
    result = cell.run()
    assert result.finished
    assert [x.arm.name for x in cell.scheduler.assignments] == ["a", "a"]
    assert cell.scheduler.assignments[1].tick >= 300
    assert result.metrics.completed == 1


@pytest.mark.parametrize("seed", [1, 2])
def test_random_faults_and_frequent_estops(default_config: CellConfig, seed: int) -> None:
    cfg = dataclasses.replace(
        default_config,
        faults=FaultConfig(probability=0.25, recovery_s=1.0),
        estop=EStopConfig(global_at_s=tuple(float(t) for t in range(5, 200, 7)), reset_after_s=0.8),
    )
    result = simulate(cfg, 200, seed)
    m = result.metrics
    assert result.finished
    assert m.faults > 10
    assert m.requeued == m.faults
    assert m.estops > 5
    assert m.completed + m.rejected == m.generated == 200
    assert m.zone_violations == 0
    assert m.limit_violations == 0
    assert m.max_target_error < 1e-9


def test_faults_and_estops_are_deterministic(default_config: CellConfig) -> None:
    cfg = dataclasses.replace(default_config, faults=FaultConfig(probability=0.3, recovery_s=0.5))
    assert simulate(cfg, 80, 5).metrics.summary() == simulate(cfg, 80, 5).metrics.summary()


def test_metrics_wait_statistics() -> None:
    m = Metrics(dt=0.5, arm_names=["a"], zone_names=["z"])
    for i, wait in enumerate([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]):
        m.task_queued(i, 0)
        m.task_assigned(i, wait)
        m.task_completed(i, "a", 0.0)
    m.ticks = 100
    m.busy_ticks["a"] = 25
    m.zone_wait("z")
    s = m.summary()
    assert s["wait_time_mean_s"] == pytest.approx(5.0)  # mean 10 ticks * 0.5 s
    assert s["wait_time_p95_s"] == pytest.approx(9.5)  # 95th pct of 0..20 ticks = 19 ticks
    assert s["arm_utilization"] == {"a": 0.25}
    assert s["zone_wait_time_s"] == {"z": 0.5}


def test_requeue_wait_is_accumulated() -> None:
    m = Metrics(dt=1.0, arm_names=["a"], zone_names=[])
    m.task_queued(1, 0)
    m.task_assigned(1, 3)
    m.task_queued(1, 10, requeue=True)
    m.task_assigned(1, 14)
    m.task_completed(1, "a", 0.0)
    assert m.wait_ticks == [7]
    assert m.requeued == 1


def test_overlapping_global_estops_hold_until_last_reset(default_config: CellConfig) -> None:
    """Regression (review 2 #2): the first press's reset used to release a later press early."""
    cfg = dataclasses.replace(
        default_config,
        faults=FaultConfig(),
        estop=EStopConfig(global_at_s=(5.0, 5.5), reset_after_s=1.0),
    )
    cell = busy_cell(cfg)
    held: list[bool] = []
    cell.at(620, lambda: held.append(all(a.state is ArmState.ESTOPPED for a in cell.arms)))
    cell.at(660, lambda: held.append(any(a.state is ArmState.ESTOPPED for a in cell.arms)))
    result = cell.run()
    assert held == [True, False]  # still stopped at 6.2 s, released after 6.5 s
    assert result.metrics.estops == 2
    assert result.finished
