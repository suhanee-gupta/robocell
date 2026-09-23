import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from robocell.trajectory import plan

angles = arrays(np.float64, 3, elements=st.floats(-3.0, 3.0))
limits = arrays(np.float64, 3, elements=st.floats(0.2, 5.0))
# Tolerances below only absorb floating point rounding, not modelling error.
FP = 1e-9


@settings(max_examples=300)
@given(angles, angles, limits, limits)
def test_limits_respected_and_target_hit_exactly(
    start: np.ndarray, goal: np.ndarray, vmax: np.ndarray, amax: np.ndarray
) -> None:
    tr = plan(start, goal, vmax, amax)
    _, vel, acc = tr.sample(np.linspace(0.0, tr.duration, 2001))
    assert np.all(np.abs(vel) <= vmax * (1 + FP))
    assert np.all(np.abs(acc) <= amax * (1 + FP))
    # Sampled positions: finite differences obey the same limits (mean value theorem).
    h = max(tr.duration, 1e-3) / 3000
    qs = tr.positions(np.arange(-2 * h, tr.duration + 3 * h, h))
    fd_vel = np.diff(qs, axis=0) / h
    fd_acc = np.diff(qs, n=2, axis=0) / (h * h)
    assert np.all(np.abs(fd_vel) <= vmax * (1 + FP) + 1e-9)
    # Second differences amplify rounding by 1/h^2; allow for that, nothing more.
    assert np.all(np.abs(fd_acc) <= amax * (1 + FP) + 1e-15 / (h * h) * 10)
    assert np.array_equal(tr.position(tr.duration), goal)
    assert np.array_equal(tr.position(tr.duration + 1.0), goal)
    if tr.duration > 0:
        assert np.array_equal(tr.position(0.0), start)
    else:  # sub-MOVE_EPS move: a zero-length trajectory that is already at the goal
        assert np.all(np.abs(goal - start) <= 1e-12)
    np.testing.assert_array_equal(tr.velocity(tr.duration), 0.0)


@settings(max_examples=300)
@given(angles, angles, limits, limits)
def test_joints_finish_together(
    start: np.ndarray, goal: np.ndarray, vmax: np.ndarray, amax: np.ndarray
) -> None:
    tr = plan(start, goal, vmax, amax)
    d = goal - start
    moving = np.abs(d) > 1e-6
    if not np.any(moving):
        return
    qs = tr.positions(np.linspace(0.0, tr.duration, 400)[1:-1])
    progress = (qs[:, moving] - start[moving]) / d[moving]
    # Every moving joint is the same fraction of the way there at every instant...
    np.testing.assert_allclose(progress, progress[:, :1] * np.ones_like(progress), atol=1e-9)
    # ...and none has arrived before the end.
    assert np.all(progress < 1.0)


@settings(max_examples=200)
@given(angles, angles, limits, limits)
def test_profile_is_as_fast_as_the_limits_allow(
    start: np.ndarray, goal: np.ndarray, vmax: np.ndarray, amax: np.ndarray
) -> None:
    tr = plan(start, goal, vmax, amax)
    d = np.abs(goal - start)
    if np.all(d < 1e-6):
        return
    peak_v = np.max(np.abs(tr.sample(np.linspace(0, tr.duration, 501))[1]) / vmax)
    peak_a = np.max(d * tr.s_acc / amax)
    # The binding joint reaches its acceleration limit, and its velocity limit unless the
    # profile is a triangle.
    assert peak_a == pytest.approx(1.0)
    assert peak_v <= 1.0 + FP


def test_trapezoid_known_values() -> None:
    # One joint, 2 rad at 1 rad/s and 1 rad/s^2: 1 s ramp, 1 s cruise, 1 s ramp.
    tr = plan(np.zeros(3), np.array([2.0, 0.0, 0.0]), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert tr.duration == pytest.approx(3.0)
    assert tr.t_acc == pytest.approx(1.0)
    assert tr.position(1.0)[0] == pytest.approx(0.5)
    assert tr.position(1.5)[0] == pytest.approx(1.0)
    assert tr.velocity(1.5)[0] == pytest.approx(1.0)


def test_triangle_known_values() -> None:
    # 0.5 rad at 1 rad/s^2 never reaches 1 rad/s: triangle with peak sqrt(0.5).
    tr = plan(np.zeros(3), np.array([0.5, 0.0, 0.0]), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert tr.duration == pytest.approx(2 * math.sqrt(0.5))
    assert tr.velocity(tr.duration / 2)[0] == pytest.approx(math.sqrt(0.5))


def test_slow_joint_sets_the_pace() -> None:
    tr = plan(np.zeros(3), np.array([1.0, 1.0, 0.1]), (1.0, 0.25, 1.0), (10.0, 10.0, 10.0))
    # Joint 1 limited to 0.25 rad/s dominates; joint 0 is slowed to match it.
    assert np.max(np.abs(tr.velocity(tr.duration / 2))) == pytest.approx(0.25)
    assert tr.velocity(tr.duration / 2)[0] == pytest.approx(0.25)


def test_zero_move() -> None:
    q = np.array([0.1, 0.2, 0.3])
    tr = plan(q, q.copy(), (1, 1, 1), (1, 1, 1))
    assert tr.duration == 0.0
    assert np.array_equal(tr.position(0.0), q)


def test_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        plan(np.zeros(3), np.ones(3), (1, 0, 1), (1, 1, 1))


@settings(max_examples=200)
@given(angles, angles, limits, limits, st.floats(-0.5, 1.5))
def test_scalar_position_matches_vectorised(
    start: np.ndarray, goal: np.ndarray, vmax: np.ndarray, amax: np.ndarray, frac: float
) -> None:
    tr = plan(start, goal, vmax, amax)
    t = frac * tr.duration
    np.testing.assert_array_equal(tr.position(t), tr.positions(np.array([t]))[0])
