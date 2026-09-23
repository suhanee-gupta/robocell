"""Forward and closed-form inverse kinematics for a 3-joint (yaw, shoulder, elbow) arm.

Conventions:
- q = (yaw, shoulder, elbow) in radians.
- yaw rotates the arm's vertical plane about the base z axis; yaw = 0 points along +x.
- shoulder is the elevation of link 1 above the horizontal.
- elbow is the angle of link 2 relative to link 1 (0 = straight arm).

In the arm's vertical plane, with radial distance r and height z above the shoulder:
    r = L1 cos(q1) + L2 cos(q1 + q2)
    z = L1 sin(q1) + L2 sin(q1 + q2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from robocell.config import ArmConfig

FloatArray = NDArray[np.float64]

# Slack for floating point noise at the workspace boundary and at joint limits.
REACH_EPS = 1e-9
LIMIT_EPS = 1e-9
# Below this radial distance the target is on the base axis and yaw is undetermined.
AXIS_EPS = 1e-9
# Maximum spacing (m) of the sample points along each link used for zone occupancy.
LINK_SPACING = 0.05


@dataclass(frozen=True)
class ArmGeometry:
    base: tuple[float, float]
    base_height: float
    l1: float
    l2: float
    joint_min: tuple[float, float, float]
    joint_max: tuple[float, float, float]

    @classmethod
    def from_config(cls, cfg: ArmConfig) -> ArmGeometry:
        return cls(
            base=cfg.base,
            base_height=cfg.base_height,
            l1=cfg.links[0],
            l2=cfg.links[1],
            joint_min=cfg.joint_min,
            joint_max=cfg.joint_max,
        )

    @property
    def shoulder(self) -> FloatArray:
        return np.array([self.base[0], self.base[1], self.base_height])

    @property
    def reach(self) -> float:
        return self.l1 + self.l2

    def within_limits(self, q: FloatArray | tuple[float, float, float]) -> bool:
        return all(
            lo - LIMIT_EPS <= float(v) <= hi + LIMIT_EPS
            for v, lo, hi in zip(q, self.joint_min, self.joint_max, strict=True)
        )


@dataclass(frozen=True)
class ArmPose:
    """Cartesian positions of the arm's key points for one joint configuration."""

    shoulder: FloatArray
    elbow: FloatArray
    tip: FloatArray


def forward(geom: ArmGeometry, q: FloatArray | tuple[float, float, float]) -> ArmPose:
    yaw, sh, el = (float(v) for v in q)
    cy, sy = math.cos(yaw), math.sin(yaw)
    r1 = geom.l1 * math.cos(sh)
    z1 = geom.l1 * math.sin(sh)
    r2 = r1 + geom.l2 * math.cos(sh + el)
    z2 = z1 + geom.l2 * math.sin(sh + el)
    bx, by = geom.base
    h = geom.base_height
    return ArmPose(
        shoulder=np.array([bx, by, h]),
        elbow=np.array([bx + r1 * cy, by + r1 * sy, h + z1]),
        tip=np.array([bx + r2 * cy, by + r2 * sy, h + z2]),
    )


def forward_batch(geom: ArmGeometry, qs: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Vectorised FK for an (N, 3) array of configurations -> (elbows, tips), each (N, 3)."""
    yaw, sh, el = qs[:, 0], qs[:, 1], qs[:, 2]
    cy, sy = np.cos(yaw), np.sin(yaw)
    r1 = geom.l1 * np.cos(sh)
    z1 = geom.l1 * np.sin(sh)
    r2 = r1 + geom.l2 * np.cos(sh + el)
    z2 = z1 + geom.l2 * np.sin(sh + el)
    bx, by = geom.base
    h = geom.base_height
    elbows = np.column_stack([bx + r1 * cy, by + r1 * sy, h + z1])
    tips = np.column_stack([bx + r2 * cy, by + r2 * sy, h + z2])
    return elbows, tips


def _link_fractions(length: float) -> FloatArray:
    n = max(1, math.ceil(length / LINK_SPACING))
    return np.arange(1, n + 1) / n


def body_gap(geom: ArmGeometry) -> float:
    """Largest distance between neighbouring body sample points along a link."""
    return max(geom.l1 / len(_link_fractions(geom.l1)), geom.l2 / len(_link_fractions(geom.l2)))


def body_points_batch(geom: ArmGeometry, qs: FloatArray) -> FloatArray:
    """Sample points on the arm's body for (N, 3) configurations -> (N, P, 3).

    Shoulder, then points along link 1 ending at the elbow, then along link 2 ending at
    the tip, spaced at most LINK_SPACING apart. A zone is "occupied" by an arm when any of
    these points is inside it, so a link passing through a zone is caught, not only the
    elbow and tip.
    """
    elbows, tips = forward_batch(geom, np.atleast_2d(qs))
    shoulder = geom.shoulder
    f1 = _link_fractions(geom.l1)[None, :, None]
    f2 = _link_fractions(geom.l2)[None, :, None]
    link1 = shoulder + f1 * (elbows - shoulder)[:, None, :]
    link2 = elbows[:, None, :] + f2 * (tips - elbows)[:, None, :]
    shoulders = np.broadcast_to(shoulder, (len(elbows), 1, 3))
    return np.concatenate([shoulders, link1, link2], axis=1)


def body_points(geom: ArmGeometry, q: FloatArray | tuple[float, float, float]) -> FloatArray:
    pts: FloatArray = body_points_batch(geom, np.asarray(q, dtype=np.float64))[0]
    return pts


def tip_position(geom: ArmGeometry, q: FloatArray | tuple[float, float, float]) -> FloatArray:
    return forward(geom, q).tip


class Unreachable(Enum):
    OUT_OF_REACH = "out_of_reach"
    JOINT_LIMITS = "joint_limits"


@dataclass(frozen=True)
class IKSolution:
    q: FloatArray
    elbow_up: bool


def _representations(angle: float, lo: float, hi: float) -> list[float]:
    """Every angle + 2*pi*k that lies inside [lo, hi] (with LIMIT_EPS slack)."""
    k_min = math.ceil((lo - LIMIT_EPS - angle) / math.tau)
    k_max = math.floor((hi + LIMIT_EPS - angle) / math.tau)
    # Clamping only absorbs the LIMIT_EPS slack; it never moves an angle by more than that.
    return [min(max(angle + k * math.tau, lo), hi) for k in range(k_min, k_max + 1)]


def _planar(geom: ArmGeometry, r: float, z: float) -> list[tuple[float, float]] | None:
    """Two-link planar IK in the arm's vertical plane (r may be negative: reaching backwards).

    Law of cosines on the triangle (shoulder, elbow, target):
        d^2 = r^2 + z^2 = L1^2 + L2^2 + 2 L1 L2 cos(q2)
    so cos(q2) = (d^2 - L1^2 - L2^2) / (2 L1 L2), with two mirror solutions +/-q2.
    Then q1 = atan2(z, r) - atan2(L2 sin q2, L1 + L2 cos q2).
    Returns both (q1, q2) solutions, elbow-up (higher elbow) first, or None if out of reach.
    """
    l1, l2 = geom.l1, geom.l2
    c2 = (r * r + z * z - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if c2 > 1.0 + REACH_EPS or c2 < -1.0 - REACH_EPS:
        return None
    c2 = min(1.0, max(-1.0, c2))
    sols = []
    for q2 in (-math.acos(c2), math.acos(c2)):
        q1 = math.atan2(z, r) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        sols.append((q1, q2))
    # Elbow height above the shoulder is L1 sin(q1); "elbow-up" is the higher of the two.
    # (For r >= 0 that is the q2 <= 0 branch; for r < 0 the mirror image.)
    sols.sort(key=lambda s: -math.sin(s[0]))
    return sols


def inverse(
    geom: ArmGeometry,
    target: FloatArray | tuple[float, float, float],
    *,
    yaw_hint: float = 0.0,
    allow_elbow_down: bool = True,
) -> IKSolution | Unreachable:
    """Closed-form IK: yaw from atan2, then the 2-link planar solution in the yaw plane.

    Candidates, in order of preference:
      1. yaw = atan2(dy, dx), elbow-up      3. yaw + pi (reach over the top), elbow-up
      2. yaw = atan2(dy, dx), elbow-down    4. yaw + pi, elbow-down
    Elbow-down and over-the-top are only used when the preferred pose violates joint limits.
    Among equivalent yaw angles (+/- 2 pi) inside the limits, picks the one nearest
    `yaw_hint` (normally the current yaw) so the base takes the short way round.
    """
    x, y, z = (float(v) for v in target)
    dx, dy = x - geom.base[0], y - geom.base[1]
    dz = z - geom.base_height
    r = math.hypot(dx, dy)

    planar_front = _planar(geom, r, dz)
    if planar_front is None:
        return Unreachable.OUT_OF_REACH
    planar_back = _planar(geom, -r, dz)
    assert planar_back is not None  # same distance, so equally reachable

    if r < AXIS_EPS:
        # On the base axis every yaw reaches the point: keep the current one if it is legal,
        # else the nearest legal yaw (the hint may be outside limits, e.g. the default 0).
        lo, hi = geom.joint_min[0], geom.joint_max[0]
        yaw = yaw_hint if _representations(yaw_hint, lo, hi) else min(max(yaw_hint, lo), hi)
    else:
        yaw = math.atan2(dy, dx)

    for yaw_raw, planar in ((yaw, planar_front), (yaw + math.pi, planar_back)):
        yaws = _representations(yaw_raw, geom.joint_min[0], geom.joint_max[0])
        if not yaws:
            continue
        yaw_q = min(yaws, key=lambda c: (abs(c - yaw_hint), c))
        branches = planar if allow_elbow_down else planar[:1]
        for i, (q1, q2) in enumerate(branches):
            # atan2/acos return one representative of each angle; the limits may use another.
            q1s = _representations(q1, geom.joint_min[1], geom.joint_max[1])
            q2s = _representations(q2, geom.joint_min[2], geom.joint_max[2])
            if q1s and q2s:
                return IKSolution(q=np.array([yaw_q, q1s[0], q2s[0]]), elbow_up=i == 0)
    return Unreachable.JOINT_LIMITS
