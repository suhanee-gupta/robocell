# Independent reviews

Each review was done by a fresh reviewer agent that had not seen the implementation
reasoning. The brief covered kinematics (math, angle wrapping, full extension, base axis),
trajectory limits, the scheduler, zone locking (deadlocks, missed zone entries, lock leaks),
E-stop handling and the asyncio task lifecycle. Every real finding got a fix and a
regression test, and the full verification gate was rerun afterwards.

## Round 1: after milestone 5 (shared zones)

| # | Severity | Finding | Fix | Regression test |
|---|----------|---------|-----|-----------------|
| 1 | high | `[faults]` and `[estop]` config was parsed but never used; `faults`/`estops` stayed 0 | Milestone 6: `Cell._plan_faults` / `_fault` / `_recover` inject seeded faults and clear them after `recovery_s`; `_global_estop_press` presses and resets the global E-stop at `global_at_s`. `verify.sh` asserts faults > 0, requeued == faults, estops == 2 | `tests/test_faults.py::test_random_faults_and_frequent_estops`, `scripts/verify.sh` |
| 2 | medium | E-stop + reset on a faulted arm silently cleared the fault (FAULT -> ESTOPPED -> IDLE) | `Arm.fault_latched`; `reset()` returns to FAULT while a fault is latched, and only `clear_fault()` clears it | `tests/test_arm.py::test_estop_reset_does_not_clear_a_fault` |
| 3 | medium | After a fault inside a zone, the arm kept its whole plan and drove on to the abandoned (already requeued) target, possibly converging with the arm now serving it | `Arm._abandon_plan` truncates the remaining path at the first pose clear of every zone (`_clearing_waypoints`); still needs only locks it already holds | `tests/test_zones.py::test_fault_inside_zone_clears_out_and_releases` |
| 4 | low | Target on the base axis reported unreachable when yaw limits exclude the default hint 0 | On the axis, IK uses the hint if legal, else the nearest legal yaw | `tests/test_kinematics.py::test_base_axis_reachable_when_yaw_limits_exclude_zero` |
| 5 | low | Zone occupancy only checked elbow and tip, so a link could pass through a zone unnoticed | Occupancy now uses body points: shoulder plus points <= 5 cm apart along both links (elbow and tip included). The sweep margin adds half that spacing, so planning is conservative for the continuous links | `tests/test_zones.py::test_link_through_zone_counts_as_occupied`; `test_sweep_never_misses_contact` now compares against densely sampled links |

Nits reviewed and left as is (no behavioural effect in the simulator):
- Joints moving less than 1e-12 rad are ignored when timing a move.
- `AXIS_EPS` is absolute (1e-9 m).
- A `Cell` can run only once (build a new one per run).
- An arm E-stopped while queued for a zone rejoins the back of the queue after reset (fairness only).
- The scheduler breaks ties by position in the capable-arm list, which is the same order as the arm index.

## Round 2: after milestone 7 (viewer, README)

The reviewer also stress-tested 12 seeds with fault probability 0.2-0.9, 8 random global
E-stops each, and short and long recovery/reset times. Every run finished with 0 zone and
0 limit violations, requeued == faults and completed + rejected == generated. Further
checks: 20,000 random geometries for FK -> IK -> FK and 20,000 chained trajectory segments
(peak velocity and acceleration at exactly 1.0x the limits).

| # | Severity | Finding | Fix | Regression test |
|---|----------|---------|-----|-----------------|
| 1 | medium | Rest-pose decisions (retreat needed? home valid? fault clearing pose) and the monitor tested discrete body points against exact boxes, so a link could clip a zone between two samples while the arm idled there without the lock | Occupancy is now conservative everywhere: body points are tested against boxes grown by half the point spacing (`Arm.body_margin`). Every point on a link is within that distance of a sample. The motion sweep margin already included it | `tests/test_zones.py::test_rest_pose_with_link_clipping_zone_between_samples_retreats` |
| 2 | low/medium | Overlapping global E-stops: the first press's scheduled reset also released arms held by a later press | `Cell._estop_reset_due` keeps the latest deadline; a reset only fires when it is due | `tests/test_faults.py::test_overlapping_global_estops_hold_until_last_reset` |
| 3 | low | A faulted arm kept locks for the abandoned rest of its path during recovery, blocking others | `_abandon_plan` releases every held zone that the clearing path does not need and that the arm is not inside | assertion added to `tests/test_zones.py::test_fault_inside_zone_clears_out_and_releases` |
| 4 | low | A `ConfigError` raised while building the cell (e.g. home pose in a zone) crashed with a traceback and exit code 1; `--max-frames <= 0` and `--speed 0` crashed | CLI validates viewer flags and catches cell-construction config errors (exit 2) | `tests/test_cli.py::test_home_in_zone_config_is_a_usage_error`, `test_bad_viewer_flags_rejected` |
| 5 | low (doc) | `_abandon_plan` docstring claimed the arm never drives to the abandoned target, but when the target is inside a zone the first clear pose lies on the retreat leg | Docstring corrected: the arm may pass the target, still under lock (safe) | n/a (documentation) |

Nits reviewed and left as is:
- A target just outside `AXIS_EPS` takes its yaw from numerical noise.
- With joint limits wider than 2π, the shoulder/elbow representation is not chosen nearest the current pose.
- A fault's random time is drawn over the move duration counted from assignment, so it can fire while the arm still waits for zones. It is still handled correctly.
- The pedestal (base to shoulder) is not part of the body points (listed as a simplification in the walkthrough).
