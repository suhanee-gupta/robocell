import numpy as np
import pytest
from helpers import two_arm_config

from robocell.cell import Cell, generate_tasks, simulate
from robocell.config import CellConfig
from robocell.scheduler import Task

NEAR_A = (0.6, 0.2, 0.5)  # both arms reach it, a is closer
NEAR_B = (1.0, -0.2, 0.5)
FAR_FROM_B = (-0.5, 0.3, 0.6)
UNREACHABLE = (0.8, 0.0, 3.0)


def test_fixture_reachability() -> None:
    cell = Cell(two_arm_config(), [])
    a, b = cell.arms
    assert a.can_reach(NEAR_A)
    assert b.can_reach(NEAR_A)  # both can, a is closer
    assert a.can_reach(FAR_FROM_B)
    assert not b.can_reach(FAR_FROM_B)
    assert not a.can_reach(UNREACHABLE)
    assert not b.can_reach(UNREACHABLE)


def test_unreachable_task_rejected() -> None:
    cell = Cell(two_arm_config(), [Task(0, UNREACHABLE, arrival_tick=1)])
    result = cell.run()
    assert result.finished
    assert result.metrics.rejected == 1
    assert result.metrics.completed == 0
    assert cell.scheduler.assignments == []


def test_lowest_estimated_move_time_wins() -> None:
    cell = Cell(
        two_arm_config(), [Task(0, NEAR_B, arrival_tick=1), Task(1, NEAR_A, arrival_tick=1)]
    )
    a, b = cell.arms
    assert b.estimate(NEAR_B) < a.estimate(NEAR_B)  # type: ignore[operator]
    assert a.estimate(NEAR_A) < b.estimate(NEAR_A)  # type: ignore[operator]
    cell.run()
    assigned = {x.task.id: x.arm.name for x in cell.scheduler.assignments}
    assert assigned == {0: "b", 1: "a"}


def test_priority_then_fifo_order() -> None:
    # Only arm a can reach FAR_FROM_B, so these tasks are served one at a time by a.
    prios = [0, 2, 1, 2, 0]
    tasks = [Task(i, FAR_FROM_B, priority=p, arrival_tick=1) for i, p in enumerate(prios)]
    cell = Cell(two_arm_config(), tasks)
    cell.run()
    order = [x.task.id for x in cell.scheduler.assignments]
    assert order == [1, 3, 2, 0, 4]
    assert all(x.arm.name == "a" for x in cell.scheduler.assignments)


def test_task_waits_while_capable_arm_busy_but_others_proceed() -> None:
    tasks = [
        Task(0, FAR_FROM_B, priority=0, arrival_tick=1),
        Task(1, FAR_FROM_B, priority=5, arrival_tick=2),  # high priority, only a can do it
        Task(2, NEAR_B, priority=0, arrival_tick=2),  # low priority, b is free
    ]
    cell = Cell(two_arm_config(), tasks)
    cell.run()
    by_id = {x.task.id: x for x in cell.scheduler.assignments}
    assert by_id[2].arm.name == "b"
    assert by_id[2].tick == 2  # not blocked behind the higher-priority task
    assert by_id[1].arm.name == "a"
    assert by_id[1].tick > 2  # waited for a to finish task 0
    assert cell.metrics.completed == 3


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_never_assigns_unreachable(default_config: CellConfig, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tasks = generate_tasks(default_config.tasks, 150, rng, default_config.sim.tick_s)
    cell = Cell(default_config, tasks)
    result = cell.run()
    assert result.finished
    for x in cell.scheduler.assignments:
        assert x.arm.can_reach(x.task.target)
    for task in tasks:
        if not any(a.can_reach(task.target) for a in cell.arms):
            assert task.id not in {x.task.id for x in cell.scheduler.assignments}
    m = result.metrics
    assert m.completed + m.rejected == m.generated == 150
    assert m.max_target_error < 1e-9
    assert m.limit_violations == 0


def test_same_seed_same_result(default_config: CellConfig) -> None:
    a = simulate(default_config, 100, seed=7).metrics.summary()
    b = simulate(default_config, 100, seed=7).metrics.summary()
    c = simulate(default_config, 100, seed=8).metrics.summary()
    assert a == b
    assert a != c


def test_duplicate_task_id_rejected() -> None:
    cell = Cell(two_arm_config(), [])
    cell.scheduler.submit(Task(1, NEAR_A), 0)
    with pytest.raises(ValueError, match="duplicate"):
        cell.scheduler.submit(Task(1, NEAR_B), 0)


def test_event_hook_must_be_in_future() -> None:
    cell = Cell(two_arm_config(), [])
    with pytest.raises(ValueError, match="future"):
        cell.at(0, lambda: None)
