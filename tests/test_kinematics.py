import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocell.kinematics import (
    ArmGeometry,
    IKSolution,
    Unreachable,
    forward,
    forward_batch,
    inverse,
    tip_position,
)

D = math.radians

# Default-config style arm.
ARM = ArmGeometry(
    base=(0.5, -0.2),
    base_height=0.4,
    l1=0.6,
    l2=0.5,
    joint_min=(D(-170), D(-20), D(-150)),
    joint_max=(D(170), D(110), D(150)),
)
# Same arm with (almost) no joint limits: only reach can make a point unreachable.
FREE = ArmGeometry(
    base=(0.0, 0.0),
    base_height=0.3,
    l1=0.7,
    l2=0.4,
    joint_min=(-math.pi, -math.pi, -math.pi),
    joint_max=(math.pi, math.pi, math.pi),
)


def solve(geom: ArmGeometry, p: tuple[float, float, float], **kw: float) -> IKSolution:
    sol = inverse(geom, p, **kw)  # type: ignore[arg-type]
    assert isinstance(sol, IKSolution), sol
    return sol


def joint_strategy(geom: ArmGeometry) -> st.SearchStrategy[tuple[float, float, float]]:
    return st.tuples(
        *(
            st.floats(lo, hi, allow_nan=False)
            for lo, hi in zip(geom.joint_min, geom.joint_max, strict=True)
        )
    )


def test_fk_zero_pose_is_straight_along_x() -> None:
    pose = forward(ARM, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(pose.shoulder, [0.5, -0.2, 0.4])
    np.testing.assert_allclose(pose.elbow, [1.1, -0.2, 0.4])
    np.testing.assert_allclose(pose.tip, [1.6, -0.2, 0.4])


def test_fk_known_pose() -> None:
    # Yaw 90 deg (+y), shoulder straight up, elbow bent 90 deg forward.
    pose = forward(ARM, (D(90), D(90), D(-90)))
    np.testing.assert_allclose(pose.elbow, [0.5, -0.2, 1.0], atol=1e-12)
    np.testing.assert_allclose(pose.tip, [0.5, 0.3, 1.0], atol=1e-12)


@given(joint_strategy(ARM))
def test_forward_batch_matches_forward(q: tuple[float, float, float]) -> None:
    elbows, tips = forward_batch(ARM, np.array([q, q]))
    pose = forward(ARM, q)
    np.testing.assert_allclose(elbows[1], pose.elbow, atol=1e-12)
    np.testing.assert_allclose(tips[0], pose.tip, atol=1e-12)


@settings(max_examples=500)
@given(joint_strategy(ARM), st.floats(-math.pi, math.pi))
def test_fk_ik_roundtrip_for_reachable_points(q: tuple[float, float, float], hint: float) -> None:
    """Every point FK can produce inside the joint limits must be solved, and FK(IK(p)) == p."""
    p = tip_position(ARM, q)
    sol = inverse(ARM, p, yaw_hint=hint)
    assert isinstance(sol, IKSolution), f"reachable point {p} reported {sol}"
    assert ARM.within_limits(sol.q)
    np.testing.assert_allclose(tip_position(ARM, sol.q), p, atol=1e-7)


@settings(max_examples=500)
@given(
    st.floats(-1.5, 1.5),
    st.floats(-1.5, 1.5),
    st.floats(-1.2, 1.8),
)
def test_ik_classifies_arbitrary_points(x: float, y: float, z: float) -> None:
    """With no joint limits, a point is reachable iff its distance from the shoulder allows."""
    p = (x, y, z)
    d = float(np.linalg.norm(np.array(p) - FREE.shoulder))
    sol = inverse(FREE, p)
    if isinstance(sol, IKSolution):
        np.testing.assert_allclose(tip_position(FREE, sol.q), p, atol=1e-7)
        assert abs(FREE.l1 - FREE.l2) - 1e-9 <= d <= FREE.reach + 1e-9
    else:
        assert sol is Unreachable.OUT_OF_REACH
        assert d > FREE.reach - 1e-9 or d < abs(FREE.l1 - FREE.l2) + 1e-9


def test_full_extension_is_reachable() -> None:
    p = (0.5 + ARM.reach, -0.2, 0.4)
    sol = solve(ARM, p)
    assert sol.q[2] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(tip_position(ARM, sol.q), p, atol=1e-9)


def test_full_extension_diagonal_is_reachable() -> None:
    # Straight arm pointing up and out at 45 deg in a diagonal plane.
    s = ARM.reach / math.sqrt(2)
    p = (0.5 + s / math.sqrt(2), -0.2 + s / math.sqrt(2), 0.4 + s)
    sol = solve(ARM, p)
    np.testing.assert_allclose(tip_position(ARM, sol.q), p, atol=1e-7)


def test_just_beyond_reach_is_unreachable() -> None:
    assert inverse(ARM, (0.5 + ARM.reach + 1e-6, -0.2, 0.4)) is Unreachable.OUT_OF_REACH


def test_inside_inner_radius_is_unreachable() -> None:
    # |L1 - L2| = 0.3 for FREE; a point 0.1 m from the shoulder cannot be reached.
    assert inverse(FREE, (0.1, 0.0, 0.3)) is Unreachable.OUT_OF_REACH


def test_inner_boundary_is_reachable_folded() -> None:
    sol = solve(FREE, (0.3, 0.0, 0.3))
    assert abs(sol.q[2]) == pytest.approx(math.pi, abs=1e-6)


def test_point_on_base_axis_keeps_hint_yaw() -> None:
    p = (0.5, -0.2, 0.4 + 0.8)
    sol = solve(ARM, p, yaw_hint=0.3)
    assert sol.q[0] == pytest.approx(0.3)
    np.testing.assert_allclose(tip_position(ARM, sol.q), p, atol=1e-9)


def test_point_on_base_axis_hint_outside_limits_is_wrapped() -> None:
    # 200 deg is outside +/-170 but equivalent to -160 deg, which is inside.
    sol = solve(ARM, (0.5, -0.2, 1.2), yaw_hint=D(200))
    assert sol.q[0] == pytest.approx(D(-160))


def test_elbow_up_preferred() -> None:
    p = np.array([1.2, 0.2, 0.6])
    sol = solve(ARM, (1.2, 0.2, 0.6))
    assert sol.elbow_up
    assert sol.q[2] < 0  # for a forward reach, elbow-up bends link 2 down relative to link 1
    # The elbow sits above the straight line from shoulder to target.
    elbow, shoulder = forward(ARM, sol.q).elbow, ARM.shoulder
    t = np.dot(elbow - shoulder, p - shoulder) / np.dot(p - shoulder, p - shoulder)
    assert elbow[2] > (shoulder + t * (p - shoulder))[2]


def test_elbow_down_fallback_when_elbow_up_violates_limits() -> None:
    # Elbow restricted to [5, 150] deg: only the elbow-down branch is legal.
    geom = ArmGeometry(
        ARM.base, ARM.base_height, ARM.l1, ARM.l2, (D(-170), D(-60), D(5)), ARM.joint_max
    )
    p = (1.2, 0.2, 0.6)
    sol = solve(geom, p)
    assert not sol.elbow_up
    assert sol.q[2] > 0
    np.testing.assert_allclose(tip_position(geom, sol.q), p, atol=1e-9)
    assert inverse(geom, p, allow_elbow_down=False) is Unreachable.JOINT_LIMITS


def test_yaw_limit_violation_detected() -> None:
    # Yaw limited to +/-90 deg and shoulder to <= 80 deg: nothing behind the base is reachable.
    geom = ArmGeometry((0.0, 0.0), 0.4, 0.6, 0.5, (D(-90), D(-20), D(-150)), (D(90), D(80), D(150)))
    assert inverse(geom, (-0.8, 0.1, 0.4)) is Unreachable.JOINT_LIMITS
    assert isinstance(inverse(geom, (0.8, 0.1, 0.4)), IKSolution)


def test_over_the_top_used_when_front_yaw_out_of_limits() -> None:
    geom = ArmGeometry(
        (0.0, 0.0), 0.4, 0.6, 0.5, (D(-90), D(-20), D(-150)), (D(90), D(170), D(150))
    )
    p = (-0.3, 0.05, 1.0)
    sol = solve(geom, p)
    # The front yaw (~170 deg) is outside +/-90, so the arm faces away and reaches backwards.
    assert sol.q[0] == pytest.approx(math.atan2(0.05, -0.3) + math.pi - math.tau)
    np.testing.assert_allclose(tip_position(geom, sol.q), p, atol=1e-9)


def test_shoulder_limit_violation_detected() -> None:
    # Target far below the shoulder needs shoulder < -20 deg.
    assert inverse(ARM, (0.5 + 0.3, -0.2, 0.4 - 0.8)) is Unreachable.JOINT_LIMITS


def test_elbow_limit_violation_detected() -> None:
    # Very close to the shoulder needs a tightly folded elbow (> 150 deg).
    assert inverse(ARM, (0.5 + 0.12, -0.2, 0.4 + 0.05)) is Unreachable.JOINT_LIMITS


def test_yaw_wrapping_picks_nearest_representation() -> None:
    wide = ArmGeometry(
        (0.0, 0.0), 0.4, 0.6, 0.5, (D(-270), D(-20), D(-150)), (D(270), D(110), D(150))
    )
    p = (math.cos(D(170)), math.sin(D(170)), 0.4)
    near_neg = solve(wide, p, yaw_hint=D(-170))
    near_pos = solve(wide, p, yaw_hint=D(160))
    assert near_neg.q[0] == pytest.approx(D(-190))
    assert near_pos.q[0] == pytest.approx(D(170))
    for sol in (near_neg, near_pos):
        np.testing.assert_allclose(tip_position(wide, sol.q), p, atol=1e-9)


def test_yaw_at_atan2_branch_cut() -> None:
    # Target exactly behind (-x): atan2 returns +pi; yaw limits of +/-180 deg must accept it.
    geom = ArmGeometry(
        (0.0, 0.0), 0.4, 0.6, 0.5, (-math.pi, D(-20), D(-150)), (math.pi, D(110), D(150))
    )
    p = (-0.8, 0.0, 0.5)
    for hint in (-3.0, 3.0):
        sol = solve(geom, p, yaw_hint=hint)
        assert abs(sol.q[0]) == pytest.approx(math.pi)
        np.testing.assert_allclose(tip_position(geom, sol.q), p, atol=1e-9)
