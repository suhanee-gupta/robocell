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
