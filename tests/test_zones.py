import math

import numpy as np
import pytest
from helpers import arm_cfg, two_arm_config
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from robocell.arm import Arm, ArmState
from robocell.cell import Cell, generate_tasks
from robocell.config import (
    CellConfig,
    ConfigError,
    EStopConfig,
    FaultConfig,
    SimConfig,
    TaskGenConfig,
    ZoneConfig,
)
from robocell.kinematics import ArmGeometry, body_points_batch, tip_position
from robocell.scheduler import Task
from robocell.trajectory import plan
from robocell.zones import SUBSTEPS, Zone, ZoneManager, sweep, sweep_margin

DT = 0.01
# Two zones between arms a (x=0) and b (x=1.6). Reaching the targets below, a's path enters
# z2 before z1 and b's enters z1 before z2: locking on entry would deadlock.
CROSS_ZONES = (
    ZoneConfig("z1", (0.5, -0.6, 0.0), (0.7, 0.6, 1.4)),
    ZoneConfig("z2", (0.9, -0.6, 0.0), (1.1, 0.6, 1.4)),
)
A_TARGET = (1.0, 0.3, 0.6)
B_TARGET = (0.6, -0.3, 0.6)


def zm(*boxes: tuple[str, tuple[float, float, float], tuple[float, float, float]]) -> ZoneManager:
    return ZoneManager([ZoneConfig(n, lo, hi) for n, lo, hi in boxes])


def no_locks_left(zones: ZoneManager) -> bool:
    return all(zones.holder(i) is None and not zones.waiting(i) for i in range(len(zones)))


# ---- geometry ----------------------------------------------------------------------


def test_contains_with_margin() -> None:
    z = Zone("z", np.zeros(3), np.ones(3))
    pts = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.05, 0.5, 0.5], [-0.2, 0.5, 0.5]])
    assert z.contains(pts).tolist() == [True, True, False, False]
    assert z.contains(pts, margin=0.1).tolist() == [True, True, True, False]


# ---- lock manager ------------------------------------------------------------------


def test_zones_ordered_by_name() -> None:
    m = zm(("b", (0, 0, 0), (1, 1, 1)), ("a", (2, 2, 2), (3, 3, 3)))
    assert m.names == ["a", "b"]


def test_acquire_in_global_order_and_report_blocker() -> None:
    m = zm(("a", (0, 0, 0), (1, 1, 1)), ("b", (0, 0, 0), (1, 1, 1)), ("c", (0, 0, 0), (1, 1, 1)))
    assert m.acquire_in_order("x", [1]) is None
    # y needs {0, 1, 2}: gets 0, blocks on 1, and must not grab 2 out of order.
    assert m.acquire_in_order("y", [2, 0, 1]) == 1
    assert m.held_by("y") == {0}
    assert m.holder(2) is None
    assert m.waiting(1) == ["y"]
    m.release(1, "x")
    assert m.acquire_in_order("y", [0, 1, 2]) is None
    assert m.held_by("y") == {0, 1, 2}


def test_lock_queue_is_fifo() -> None:
    m = zm(("a", (0, 0, 0), (1, 1, 1)))
    assert m.acquire_in_order("x", [0]) is None
    assert m.acquire_in_order("y", [0]) == 0
    assert m.acquire_in_order("z", [0]) == 0
    m.release(0, "x")
    # z asks first after the release, but y has been waiting longer.
    assert m.acquire_in_order("z", [0]) == 0
    assert m.acquire_in_order("y", [0]) is None


def test_release_by_non_holder_raises() -> None:
    m = zm(("a", (0, 0, 0), (1, 1, 1)))
    m.acquire_in_order("x", [0])
    with pytest.raises(RuntimeError, match="held by x"):
        m.release(0, "y")


def test_release_all_leaves_queues() -> None:
    m = zm(("a", (0, 0, 0), (1, 1, 1)), ("b", (0, 0, 0), (1, 1, 1)))
    m.acquire_in_order("x", [1])
    assert m.acquire_in_order("y", [0, 1]) == 1
    m.release_all("y")
    assert m.held_by("y") == set()
    assert m.waiting(1) == []


# ---- path sweep --------------------------------------------------------------------

GEOM = ArmGeometry((0.0, 0.0), 0.4, 0.6, 0.5, (-3.0, -0.5, -2.6), (3.0, 2.0, 2.6))
VMAX = np.array([2.0, 1.5, 2.5])
AMAX = np.array([4.0, 3.0, 5.0])


@settings(max_examples=150, deadline=None)
@given(
    arrays(np.float64, 3, elements=st.floats(-1.0, 1.0)),
    arrays(np.float64, 3, elements=st.floats(-1.0, 1.0)),
    arrays(np.float64, 3, elements=st.floats(-1.2, 1.2)),
    arrays(np.float64, 3, elements=st.floats(0.005, 0.4)),
)
def test_sweep_never_misses_contact(
    q0: np.ndarray, q1: np.ndarray, corner: np.ndarray, size: np.ndarray
) -> None:
    """Compare against 64x finer time sampling of densely sampled links: every contact is
    found (including a link passing through with elbow and tip outside), and the
    reported last-contact time is never earlier than a real contact (minus half a sample)."""
    corner = corner + np.array([0.0, 0.0, 0.4])
    zones = zm(("z", tuple(corner), tuple(corner + size)))  # type: ignore[arg-type]
    seg = plan(q0, q1, VMAX, AMAX)
    spacing = DT / SUBSTEPS
    touch = sweep(GEOM, seg, zones, sweep_margin(GEOM, VMAX, spacing), DT)
    fine = np.linspace(0.0, seg.duration, max(2, int(seg.duration / spacing) * 64 + 1))
    body = body_points_batch(GEOM, seg.positions(fine))
    # Dense points along the links (4x the body sampling) approximate the continuous arm.
    a, b = body[:, :-1, :], body[:, 1:, :]
    dense = np.concatenate([a + f * (b - a) for f in np.linspace(0, 1, 5)], axis=1)
    hit = zones.zones[0].contains(dense.reshape(-1, 3)).reshape(dense.shape[:2]).any(axis=1)
    if np.any(hit):
        assert 0 in touch
        assert fine[np.flatnonzero(hit)[-1]] <= touch[0] + spacing / 2 + 1e-12


def test_sweep_catches_fast_crossing_of_thin_zone() -> None:
    # The tip swings 180 deg around the base in well under a second and crosses a 1 mm slab.
    seg = plan(np.array([-1.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0]), VMAX, AMAX)
    zones = zm(("slab", (1.0, -0.0005, 0.3), (1.2, 0.0005, 0.5)))
    touch = sweep(GEOM, seg, zones, sweep_margin(GEOM, VMAX, DT / SUBSTEPS), DT)
    assert 0 in touch


def test_link_through_zone_counts_as_occupied() -> None:
    """Regression (review #5): only elbow and tip were checked, so a link could pass
    through a zone unnoticed."""
    zones = zm(("small", (0.2, -0.05, 0.35), (0.3, 0.05, 0.45)))
    arm = Arm(arm_cfg("a", (0.0, 0.0), 150.0), DT, zones=zones)
    q = np.array([0.0, 0.0, 0.0])  # link 1 runs straight through the box along +x
    pose_pts = np.vstack([tip_position(arm.geom, q), [0.6, 0.0, 0.4]])
    assert not zones.occupied(pose_pts)  # neither elbow nor tip is inside
    assert arm.occupied_zones(q) == {0}
    # And a sweep through that pose reports the zone.
    seg = plan(np.array([-0.5, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), VMAX, AMAX)
    assert 0 in sweep(arm.geom, seg, zones, arm.margin, DT)


def test_home_inside_zone_rejected() -> None:
    cfg = two_arm_config((ZoneConfig("bad", (-1.0, -1.0, 0.0), (0.0, 1.0, 2.0)),))
    with pytest.raises(ConfigError, match="home pose is inside"):
        Cell(cfg, [])


# ---- arms and locks ----------------------------------------------------------------


def test_target_in_zone_retreats_home_and_releases() -> None:
    cell = Cell(two_arm_config(CROSS_ZONES), [Task(0, A_TARGET, arrival_tick=1)])
    result = cell.run()
    a = cell.arms[0]
    assert result.metrics.completed == 1
    np.testing.assert_allclose(a.q, a.home)
    assert not a.occupied_zones()
    assert no_locks_left(cell.zones)


def test_crossing_paths_do_not_deadlock() -> None:
    tasks = [Task(0, A_TARGET, arrival_tick=1), Task(1, B_TARGET, arrival_tick=1)]
    cell = Cell(two_arm_config(CROSS_ZONES), tasks)
    a, b = cell.arms
    assert a.estimate(A_TARGET) < b.estimate(A_TARGET)  # type: ignore[operator]
    assert b.estimate(B_TARGET) < a.estimate(B_TARGET)  # type: ignore[operator]
    result = cell.run()
    assert result.finished
    assert {x.task.id: x.arm.name for x in cell.scheduler.assignments} == {0: "a", 1: "b"}
    assert result.metrics.completed == 2
    assert result.metrics.zone_violations == 0
    assert sum(result.metrics.zone_wait_ticks.values()) > 0  # one really waited
    assert no_locks_left(cell.zones)


def test_lock_released_after_leaving_zone_not_at_end() -> None:
    zones = ZoneManager([ZoneConfig("near", (0.3, -0.5, 0.0), (0.7, 0.5, 1.5))])
    a = Arm(arm_cfg("a", (0.0, 0.0), 150.0), DT, zones=zones)
    # Path from inside-reach of the zone out to the far side: passes the zone, ends away.
    a.q = a.solve((0.4, 0.8, 0.9)).q  # type: ignore[union-attr]
    assert not a.occupied_zones()
    a.assign(0, (0.2, -0.9, 0.5))
    assert a.plan is not None
    assert a.plan.zones == {0}
    released_while_moving = False
    for tick in range(1, 2000):
        a.step(tick)
        if a.state is ArmState.MOVING and a.plan and a.plan.started and not a.held_zones:
            released_while_moving = True
            assert not a.occupied_zones()
        if a.available:
            break
    assert released_while_moving
    assert a.available
    assert no_locks_left(zones)


def test_estop_while_waiting_leaves_queue() -> None:
    tasks = [Task(0, A_TARGET, arrival_tick=1), Task(1, B_TARGET, arrival_tick=1)]
    cell = Cell(two_arm_config(CROSS_ZONES), tasks)
    a, b = cell.arms
    waiter: list[Arm] = []

    def stop_waiter() -> None:
        w = next(arm for arm in (a, b) if arm.plan is not None and not arm.plan.started)
        waiter.append(w)
        w.estop()
        assert not w.held_zones
        assert all(w.name not in cell.zones.waiting(i) for i in range(len(cell.zones)))

    cell.at(10, stop_waiter)
    cell.at(700, lambda: waiter[0].reset())
    result = cell.run()
    assert result.finished
    assert result.metrics.completed == 2
    assert result.metrics.zone_violations == 0
    assert no_locks_left(cell.zones)


def test_estop_inside_zone_keeps_lock_until_reset() -> None:
    tasks = [Task(0, A_TARGET, arrival_tick=1), Task(1, B_TARGET, arrival_tick=1)]
    cell = Cell(two_arm_config(CROSS_ZONES), tasks)
    holder: list[Arm] = []
    seen: list[set[int]] = []

    def stop_mover() -> None:
        mover = next(arm for arm in cell.arms if arm.plan is not None and arm.plan.started)
        assert mover.occupied_zones()  # stopped inside a shared zone
        holder.append(mover)
        mover.estop()

    def check_still_held() -> None:
        seen.append(holder[0].held_zones)

    # Find when the first mover is inside a zone: its target is inside z1/z2.
    cell.at(190, stop_mover)
    cell.at(600, check_still_held)
    cell.at(601, lambda: holder[0].reset())
    result = cell.run()
    assert result.finished
    assert seen[0]  # kept its locks the whole time it was stopped
    assert result.metrics.completed == 2
    assert result.metrics.zone_violations == 0
    assert no_locks_left(cell.zones)


def test_fault_inside_zone_clears_out_and_releases() -> None:
    tasks = [Task(0, A_TARGET, arrival_tick=1), Task(1, B_TARGET, arrival_tick=1)]
    cell = Cell(two_arm_config(CROSS_ZONES), tasks)
    faulted: list[Arm] = []

    def fault_mover() -> None:
        mover = next(arm for arm in cell.arms if arm.plan is not None and arm.plan.started)
        assert mover.occupied_zones()
        faulted.append(mover)
        target = np.array(cell.scheduler.tasks[mover.task_id].target)  # type: ignore[index]
        mover.inject_fault()
        plan = mover.plan
        assert plan is not None  # keeps a path to get out later...
        assert mover.held_zones
        assert plan.zones <= mover.held_zones  # ...that needs no new locks...
        final = plan.segments[-1].goal
        assert not mover.occupied_zones(final)  # ...ends clear of every zone...
        # ...and does not drive on to the abandoned target (regression, review #3).
        assert np.linalg.norm(tip_position(mover.geom, final) - target) > 0.01

    cell.at(190, fault_mover)
    cell.at(400, lambda: faulted[0].clear_fault())
    result = cell.run()
    assert result.finished
    assert result.metrics.requeued == 1
    assert result.metrics.completed == 2
    assert result.metrics.zone_violations == 0
    assert not faulted[0].occupied_zones()
    assert no_locks_left(cell.zones)


# ---- long runs ---------------------------------------------------------------------


def ring_config(n_arms: int) -> CellConfig:
    """n arms on a circle around a crowded center, with a center zone and one zone
    between each pair of neighbours."""
    radius = 1.3
    arms = []
    zones = [ZoneConfig("center", (-0.35, -0.35, 0.0), (0.35, 0.35, 1.2))]
    for i in range(n_arms):
        # Offset by half a step so no arm faces exactly -x (outside the +/-170 deg yaw limit).
        ang = 2 * math.pi * (i + 0.5) / n_arms
        base = (radius * math.cos(ang), radius * math.sin(ang))
        home_yaw = math.degrees(ang)  # facing outwards
        home_yaw = (home_yaw + 180) % 360 - 180
        arms.append(arm_cfg(f"r{i}", base, home_yaw))
        mid = ang + math.pi / n_arms
        cx, cy = 0.95 * math.cos(mid), 0.95 * math.sin(mid)
        zones.append(ZoneConfig(f"gap{i}", (cx - 0.2, cy - 0.2, 0.0), (cx + 0.2, cy + 0.2, 1.1)))
    return CellConfig(
        sim=SimConfig(max_time_s=2000.0),
        tasks=TaskGenConfig(arrival_rate_hz=6.0, x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.1, 1.0)),
        faults=FaultConfig(),
        estop=EStopConfig(),
        arms=tuple(arms),
        zones=tuple(zones),
    )


class TickChecker:
    """Checks the zone invariant from outside the cell after every tick."""

    def __init__(self, cell: Cell) -> None:
        self.cell = cell
        self.ticks = 0
        inner = cell.monitor_step

        def wrapped(tick: int) -> None:
            inner(tick)
            self.check()

        cell.monitor_step = wrapped  # type: ignore[method-assign]

    def check(self) -> None:
        self.ticks += 1
        inside: dict[int, list[str]] = {}
        for arm in self.cell.arms:
            for z in arm.occupied_zones():
                inside.setdefault(z, []).append(arm.name)
        for z, names in inside.items():
            assert len(names) == 1, f"tick {self.cell.clock.tick}: {names} in zone {z}"
            assert self.cell.zones.holder(z) == names[0]


@pytest.mark.parametrize(("n_arms", "seed"), [(4, 1), (6, 2), (8, 3)])
def test_many_arms_contending_no_deadlock_no_violation(n_arms: int, seed: int) -> None:
    cfg = ring_config(n_arms)
    tasks = generate_tasks(cfg.tasks, 200, np.random.default_rng(seed), DT)
    cell = Cell(cfg, tasks)
    checker = TickChecker(cell)
    result = cell.run()
    m = result.metrics
    assert result.finished, "deadlock or livelock: run did not complete"
    assert checker.ticks == m.ticks
    assert m.completed + m.rejected == 200
    assert m.completed > 150
    assert sum(m.zone_wait_ticks.values()) > 100  # there really was contention
    assert m.zone_violations == 0
    assert m.limit_violations == 0
    assert no_locks_left(cell.zones)


def test_default_cell_long_run_zone_invariant(default_config: CellConfig) -> None:
    tasks = generate_tasks(default_config.tasks, 400, np.random.default_rng(11), DT)
    cell = Cell(default_config, tasks)
    checker = TickChecker(cell)
    result = cell.run()
    assert result.finished
    assert checker.ticks == result.metrics.ticks > 10_000
    assert result.metrics.zone_violations == 0
    assert no_locks_left(cell.zones)
