"""Synchronised joint-space trapezoidal velocity profiles.

All joints follow one shared time scaling s(t) in [0, 1]:
    q(t) = start + (goal - start) * s(t)
so every joint starts and finishes at the same instant and the arm moves along a straight
line in joint space. s(t) is a trapezoid (accelerate, cruise, decelerate) whose peak
rate vs and acceleration as are chosen so that no joint exceeds its limits:
    |d_i| * vs <= vmax_i   and   |d_i| * as <= amax_i   for every joint i.
The tightest joint therefore sets the pace and the others are slowed to match it.
If the cruise speed cannot be reached within half the distance the profile is a triangle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robocell.kinematics import FloatArray

# Joint moves smaller than this (radians) are treated as "already there".
MOVE_EPS = 1e-12


@dataclass(frozen=True)
class Trajectory:
    start: FloatArray
    goal: FloatArray
    duration: float
    t_acc: float  # length of the acceleration (and deceleration) phase
    s_vel: float  # peak rate of the time scaling (1/s)
    s_acc: float  # acceleration of the time scaling (1/s^2)

    @property
    def delta(self) -> FloatArray:
        return self.goal - self.start

    def _scaling(self, t: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Time scaling s(t) and its first two derivatives, for an array of times."""
        total, ta, v, a = self.duration, self.t_acc, self.s_vel, self.s_acc
        t = np.clip(t, 0.0, total)
        rem = total - t
        ramp_up = t < ta
        ramp_down = t > total - ta
        s = np.where(
            ramp_up,
            0.5 * a * t * t,
            np.where(ramp_down, 1.0 - 0.5 * a * rem * rem, 0.5 * a * ta * ta + v * (t - ta)),
        )
        sd = np.where(ramp_up, a * t, np.where(ramp_down, a * rem, v))
        sdd = np.where(ramp_up, a, np.where(ramp_down, -a, 0.0))
        # Before the start and after the end the arm is at rest.
        at_rest = (t <= 0.0) | (t >= total)
        return s, np.where(at_rest, 0.0, sd), np.where(at_rest, 0.0, sdd)

    def sample(self, times: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Positions, velocities and accelerations at the given times, each (N, 3)."""
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        n = len(times)
        if self.duration == 0.0:
            zeros = np.zeros((n, 3))
            return np.tile(self.goal, (n, 1)), zeros, zeros.copy()
        s, sd, sdd = self._scaling(times)
        d = self.delta
        pos = self.start + np.outer(s, d)
        # Land exactly on the goal at (and after) the end: no floating point drift.
        pos[times >= self.duration] = self.goal
        return pos, np.outer(sd, d), np.outer(sdd, d)

    def position(self, t: float) -> FloatArray:
        """Scalar fast path of sample() (called every tick for every moving arm)."""
        total, ta, a = self.duration, self.t_acc, self.s_acc
        if t >= total:
            return self.goal.copy()
        if t <= 0.0:
            return self.start.copy()
        if t < ta:
            s = 0.5 * a * t * t
        elif t > total - ta:
            rem = total - t
            s = 1.0 - 0.5 * a * rem * rem
        else:
            s = 0.5 * a * ta * ta + self.s_vel * (t - ta)
        pos: FloatArray = self.start + self.delta * s
        return pos

    def velocity(self, t: float) -> FloatArray:
        vel: FloatArray = self.sample(np.array([t]))[1][0]
        return vel

    def acceleration(self, t: float) -> FloatArray:
        acc: FloatArray = self.sample(np.array([t]))[2][0]
        return acc

    def positions(self, times: FloatArray) -> FloatArray:
        return self.sample(times)[0]


def plan(
    start: FloatArray,
    goal: FloatArray,
    max_vel: FloatArray | tuple[float, ...],
    max_acc: FloatArray | tuple[float, ...],
) -> Trajectory:
    start = np.asarray(start, dtype=np.float64).copy()
    goal = np.asarray(goal, dtype=np.float64).copy()
    vmax = np.asarray(max_vel, dtype=np.float64)
    amax = np.asarray(max_acc, dtype=np.float64)
    if np.any(vmax <= 0) or np.any(amax <= 0):
        raise ValueError("max velocity and acceleration must be positive")
    dist = np.abs(goal - start)
    moving = dist > MOVE_EPS
    if not np.any(moving):
        return Trajectory(start, goal, 0.0, 0.0, 0.0, 0.0)

    # Largest time-scaling rate/acceleration that keeps every joint within its limits.
    s_vel = float(np.min(vmax[moving] / dist[moving]))
    s_acc = float(np.min(amax[moving] / dist[moving]))

    if s_vel * s_vel / s_acc >= 1.0:
        # Triangle: accelerate for half the distance, decelerate for the other half.
        t_acc = math.sqrt(1.0 / s_acc)
        duration = 2.0 * t_acc
        s_vel = s_acc * t_acc
    else:
        t_acc = s_vel / s_acc
        duration = t_acc + 1.0 / s_vel
    return Trajectory(start, goal, duration, t_acc, s_vel, s_acc)
