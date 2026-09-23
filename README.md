# robocell

A deterministic 3D simulator of a factory cell in which several robot arms share a workspace.
Tasks of the form "reach point (x, y, z)" arrive over time. A scheduler gives each task to a
suitable arm. The arm solves inverse kinematics in closed form and moves along a
synchronised trapezoidal joint trajectory. Shared zones are locked in a fixed global order, so
two arms are never inside the same zone at once. E-stops and injected faults are part of the
model, not an afterthought.

```
uv sync
uv run python -m robocell run --config configs/default.toml --tasks 200 --seed 1          # JSON summary
uv run python -m robocell run --tasks 200 --seed 1 --viz                                   # 3D replay window
MPLBACKEND=Agg uv run python -m robocell run --tasks 200 --seed 1 --frames out/            # PNG frames, headless
scripts/verify.sh                                                                         # end-to-end checks
```

Stack: Python 3.12+, NumPy, matplotlib, asyncio, with pytest + hypothesis, ruff, mypy --strict.
There are no servers or databases, and the whole run is a pure function of config + seed.

## Layout

| Module | Role |
|---|---|
| `kinematics.py` | Forward kinematics, closed-form IK, and body sample points for zone checks |
| `trajectory.py` | Synchronised joint-space trapezoidal velocity profiles |
| `clock.py` | `SimClock`: lock-step barrier that drives the asyncio actors on 10 ms ticks |
| `arm.py` | Arm state machine (Idle / Moving / Fault / EStopped), motion execution, zone locks |
| `zones.py` | Zone boxes, FIFO locks acquired in global order, conservative path sweep |
| `scheduler.py` | Priority + FIFO queue and arm selection by lowest estimated move time |
| `cell.py` | Wires everything together: task generation, faults, E-stops, safety monitor |
| `metrics.py` | Throughput, wait-time mean/p95, utilisation, zone waits, violation counters |
| `viz.py` | Snapshot recorder, matplotlib 3D renderer, animation |
| `cli.py` | `python -m robocell run ...` |

## The math, briefly

**Arm model.** Joint 1 is base yaw θ₀ about the vertical axis. Joint 2 is the shoulder
elevation θ₁. Joint 3 is the elbow θ₂, measured relative to link 1. The links have lengths L₁
and L₂, and the shoulder sits at height h above the base (bx, by).

**Forward kinematics.** In the arm's vertical plane:

    r = L₁ cos θ₁ + L₂ cos(θ₁ + θ₂)          z = h + L₁ sin θ₁ + L₂ sin(θ₁ + θ₂)
    tip = (bx + r cos θ₀,  by + r sin θ₀,  z)

**Inverse kinematics (closed form).**
1. θ₀ = atan2(dy, dx). This projects the problem into a plane, with r = √(dx² + dy²) and
   z' = z − h.
2. Law of cosines: cos θ₂ = (r² + z'² − L₁² − L₂²) / (2 L₁ L₂). If |cos θ₂| > 1 the point is
   out of reach.
3. θ₂ = ±acos(·), then θ₁ = atan2(z', r) − atan2(L₂ sin θ₂, L₁ + L₂ cos θ₂).
4. "Elbow-up" is whichever solution puts the elbow higher. The elbow-down solution, and the
   reach-over-the-top branch (θ₀ + π with r negated), are used only if the preferred solution
   violates joint limits.
5. Each angle is wrapped into its joint limits (±2π representations). The yaw nearest the
   current yaw is chosen, and on the base axis (r = 0) the current yaw is kept.
6. The result is either a solution or a clear `Unreachable.OUT_OF_REACH` /
   `Unreachable.JOINT_LIMITS`.

**Trajectory.** All joints share one time scaling s(t) ∈ [0, 1], so q(t) = q₀ + Δq·s(t). The
arm therefore travels a straight line in joint space and every joint finishes at the same
instant. s(t) is a trapezoid (or a triangle, for short moves). Its peak rate and acceleration
are the largest values for which every joint stays within its limits:
`ṡ ≤ min vmaxᵢ/|Δqᵢ|` and `s̈ ≤ min amaxᵢ/|Δqᵢ|`.

## Design

**Deterministic concurrency.** Every arm, plus a `control` actor and a `monitor` actor, is an
asyncio task. Each loops on `await clock.wait_tick()`. `SimClock` advances only when every
actor is parked, then wakes them in a fixed order: control → arms → monitor. Nothing awaits
wall time, so the same seed produces byte-identical output. `pace` (live viewer only) adds
wall-clock sleeps without changing any result.

**Scheduling.** Queue order is priority first, then FIFO. A task goes to the idle arm with the
lowest estimated move time, which is the planned trajectory duration. A task no arm can reach
is rejected on arrival. A task whose capable arms are all busy waits, while lower-priority
tasks may still use the other arms. When an arm faults, its task is requeued at its original
position.

**Zone locking (why it cannot deadlock).** Before its first motion tick, an arm computes every
zone its whole path might touch and acquires those locks in one global order (sorted by name),
waiting in a FIFO queue at the first one that is busy. It never moves while holding only part
of the set. A moving arm never waits for a lock, and an arm at rest never holds one: a target
inside a zone gets an automatic retreat to home. So every waiter holds only lower-numbered
locks than the one it waits for, and a wait-for cycle is impossible. Locks are released as
soon as the remaining path can no longer touch the zone.

**Not missing a zone.** A zone counts as occupied when any body sample point is inside it:
the shoulder, points at most 5 cm apart along both links, the elbow and the tip. Paths are
sampled every ¼ tick against boxes grown by a proven bound:
`(L₁+L₂)·Σvmaxᵢ·Δt/2 + point spacing/2`. That makes the sampled check conservative for the
continuous motion of the whole body. Rest poses and the monitor use boxes grown by half the
point spacing, so a link cannot clip a zone between two sample points either.

**E-stop and faults.** An E-stop (per arm or global) freezes the arm before its next tick. It
needs `reset()`, after which the arm resumes, re-timed from rest along the same joint-space
line, so it never needs a new lock. A fault hands the task back for requeueing. If the arm
stopped inside a zone, it keeps its locks and, once the fault is cleared, moves only as far as
the first pose clear of all zones. A fault stays latched through an E-stop and reset.

**Safety monitor.** Every tick the monitor checks that no zone holds two arms, that any arm
inside a zone holds its lock, and that no idle arm holds a lock. It also checks joint limits
and per-tick velocity and acceleration (except across deliberate emergency stops). The
counters appear in the JSON output, and `verify.sh` requires them to be zero.

More detail, including an interview-style walkthrough, is in `docs/WALKTHROUGH.md`. Review
findings and their fixes are in `docs/REVIEW.md`.
