import asyncio
import math

import numpy as np
import pytest

from robocell.arm import (
    ALLOWED_TRANSITIONS,
    Arm,
    ArmState,
    InvalidTransitionError,
    UnreachableTargetError,
)
from robocell.clock import SimClock
from robocell.config import ArmConfig
from robocell.kinematics import tip_position

D = math.radians
DT = 0.01
CFG = ArmConfig(
    name="a",
    base=(0.0, 0.0),
    base_height=0.4,
    links=(0.6, 0.5),
    joint_min=(D(-170), D(-20), D(-150)),
    joint_max=(D(170), D(110), D(150)),
    max_vel=(D(90), D(70), D(100)),
    max_acc=(D(180), D(140), D(220)),
    home=(D(-135), D(60), D(-100)),
)
TARGET = (0.7, 0.3, 0.5)


class Recorder:
    def __init__(self) -> None:
        self.done: list[tuple[str, int, int]] = []
        self.aborted: list[tuple[str, int, int]] = []

    def on_done(self, arm: Arm, task_id: int) -> None:
        self.done.append((arm.name, task_id, arm.tick))

    def on_abort(self, arm: Arm, task_id: int) -> None:
        self.aborted.append((arm.name, task_id, arm.tick))


def make_arm() -> tuple[Arm, Recorder]:
    rec = Recorder()
    return Arm(CFG, DT, on_done=rec.on_done, on_abort=rec.on_abort), rec


def run_ticks(arm: Arm, start: int, n: int) -> list[np.ndarray]:
    qs = []
    for tick in range(start, start + n):
        arm.step(tick)
        qs.append(arm.q.copy())
    return qs


def assert_transitions_valid(arm: Arm) -> None:
    for tr in arm.transitions:
        assert tr.new in ALLOWED_TRANSITIONS[tr.old], tr
    for a, b in zip(arm.transitions, arm.transitions[1:], strict=False):
        assert a.new is b.old


def test_moves_to_target_and_reports_done() -> None:
    arm, rec = make_arm()
    arm.assign(7, TARGET)
    duration = arm.plan.segments[0].duration  # type: ignore[union-attr]
    n = math.ceil(duration / DT - 1e-9)
    qs = run_ticks(arm, 1, n + 5)
    assert rec.done == [("a", 7, n)]
    assert arm.state is ArmState.IDLE
    assert arm.available
    np.testing.assert_allclose(tip_position(arm.geom, arm.q), TARGET, atol=1e-9)
    # Tick-to-tick joint motion never exceeds max velocity.
    steps = np.abs(np.diff(np.array([arm.home, *qs]), axis=0)) / DT
    assert np.all(steps <= arm.max_vel * (1 + 1e-9))
    assert [(t.old, t.new) for t in arm.transitions] == [
        (ArmState.IDLE, ArmState.MOVING),
        (ArmState.MOVING, ArmState.IDLE),
    ]


def test_estop_during_motion_stops_within_one_tick() -> None:
    arm, rec = make_arm()
    arm.assign(1, TARGET)
    run_ticks(arm, 1, 30)
    frozen = arm.q.copy()
    assert np.any(frozen != arm.home)
    arm.estop()
    assert arm.state is ArmState.ESTOPPED
    run_ticks(arm, 31, 50)
    assert np.array_equal(arm.q, frozen)  # not a single tick of further motion
    assert not rec.done
    assert not arm.available
    assert_transitions_valid(arm)


def test_estop_requires_reset_then_resumes_task() -> None:
    arm, rec = make_arm()
    arm.assign(1, TARGET)
    run_ticks(arm, 1, 30)
    arm.estop()
    arm.estop()  # idempotent
    run_ticks(arm, 31, 10)
    arm.reset()
    assert arm.state is ArmState.IDLE
    assert not arm.available
    qs = run_ticks(arm, 41, 600)
    assert [d[1] for d in rec.done] == [1]
    np.testing.assert_allclose(tip_position(arm.geom, arm.q), TARGET, atol=1e-9)
    # Resuming re-times the motion from rest, so velocity limits still hold.
    steps = np.abs(np.diff(np.array(qs), axis=0)) / DT
    assert np.all(steps <= arm.max_vel * (1 + 1e-9))
    assert_transitions_valid(arm)


def test_fault_hands_task_back_and_stops() -> None:
    arm, rec = make_arm()
    arm.assign(3, TARGET)
    run_ticks(arm, 1, 20)
    frozen = arm.q.copy()
    arm.inject_fault()
    assert arm.state is ArmState.FAULT
    assert rec.aborted == [("a", 3, 20)]
    run_ticks(arm, 21, 20)
    assert np.array_equal(arm.q, frozen)
    assert not arm.available
    arm.clear_fault()
    assert arm.available
    run_ticks(arm, 41, 300)
    assert not rec.done
    assert_transitions_valid(arm)


def test_estop_while_faulted_then_reset() -> None:
    arm, _ = make_arm()
    arm.inject_fault()
    arm.estop()
    assert arm.state is ArmState.ESTOPPED
    arm.reset()
    assert arm.available
    assert_transitions_valid(arm)


@pytest.mark.parametrize("action", ["reset", "clear_fault"])
def test_invalid_commands_rejected(action: str) -> None:
    arm, _ = make_arm()
    with pytest.raises(InvalidTransitionError):
        getattr(arm, action)()
    assert arm.state is ArmState.IDLE
    assert arm.transitions == []


def test_fault_while_estopped_rejected() -> None:
    arm, _ = make_arm()
    arm.estop()
    with pytest.raises(InvalidTransitionError):
        arm.inject_fault()


def test_assign_while_busy_rejected() -> None:
    arm, _ = make_arm()
    arm.assign(1, TARGET)
    with pytest.raises(InvalidTransitionError):
        arm.assign(2, TARGET)


def test_assign_unreachable_rejected() -> None:
    arm, _ = make_arm()
    with pytest.raises(UnreachableTargetError):
        arm.assign(1, (5.0, 0.0, 0.0))
    assert arm.available


def test_estimate_matches_executed_duration() -> None:
    arm, rec = make_arm()
    est = arm.estimate(TARGET)
    assert est is not None
    arm.assign(1, TARGET)
    run_ticks(arm, 1, 1000)
    assert rec.done[0][2] == math.ceil(est / DT - 1e-9)
    assert arm.estimate((5.0, 0.0, 0.0)) is None


def test_arm_actor_runs_on_sim_clock() -> None:
    async def main() -> tuple[Arm, Recorder, int]:
        clock = SimClock(DT)
        arm, rec = make_arm()
        clock.register(arm.name)
        arm.assign(1, TARGET)
        async with asyncio.TaskGroup() as tg:
            task = tg.create_task(arm.run(clock))
            await clock.run(lambda: arm.available, max_ticks=10_000)
            task.cancel()
        return arm, rec, clock.tick

    arm, rec, tick = asyncio.run(main())
    assert rec.done == [("a", 1, tick)]
    np.testing.assert_allclose(tip_position(arm.geom, arm.q), TARGET, atol=1e-9)
