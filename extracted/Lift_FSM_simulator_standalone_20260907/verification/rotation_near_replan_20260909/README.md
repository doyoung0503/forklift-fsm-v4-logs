# CAN-timed rotation and near-range replanning — 2026-09-09

Normal fitted rotations now arm a deadline owned by `calib/control.py`'s CAN
sender. The first successful movement write anchors the hold; the sender wakes
at the deadline and sends STOP without waiting for vision/FSM updates. Movement
resends remain 100 Hz. Expired rotations cannot restart from repeated stale FSM
commands. Visual early STOP remains available. Adaptive slope uses the recorded
movement-write-to-STOP duration, including delayed first writes and early stops,
and consumes each settled observation once.

These timestamps describe host CANlib write calls, not measured hardware bus
acknowledgements. Real OS scheduling, CAN buffering, and vehicle response were
not tested. The virtual transport schedules the same deadline as its own event.

Below camera Z 2.2 m, an unavailable insertion-only turn now returns to normal
waypoint planning when staging approach room and action budget remain. The
approved corrective drive can proceed inside 2.2 m. At each settled pose the FSM
checks insertion again. Visibility, nine-block insertion sweep, staging distance,
total forward distance, no-progress, timeout and cycle limits still apply.
Nominal aligned approach room is up to 0.62 m (camera 2.2 to 1.58 m); actual
available distance depends on pose and the planner. This is not unlimited retry.

## Verification

Run from the simulator directory: `python verify_rotation_near_replan.py`.
The cases use the real FSM through `process_runtime.Session`, 30 Hz perception,
zero added motion/perception error, coarse alignment disabled, and predictive
forward advance capped at zero. D below is pallet-normal pivot distance, not
camera Z. Initial heading faces the pallet center. Exact options, source hashes,
events, final poses and result reasons are in `case_01.json` through
`case_08.json` and `summary.json`. The workspace also contains the separately
requested adaptive forward cap; these runs use that current policy.

| D (m) | Lateral (m) | Result | Near replans |
|---|---|---|---|
| 4.5 | +0.64125 | DONE | 0 |
| 4.5 | +0.70 | DONE | 0 |
| 3.0 | +0.40 | Visibility planner rejected remaining path | 2 |
| 3.0 | -0.40 | Visibility planner rejected remaining path | 2 |
| 3.5 | +0.50 | Visibility planner rejected remaining path | 2 |
| 4.0 | +0.60 | DONE | 1 |
| 4.0 | -0.60 | DONE | 1 |
| 3.0 | 0 | DONE | 0 |

The previously failing 4.5 m / 0.64125 m case now succeeds alongside 0.70 m.
This is a regression check, not proof of monotonic reachability or a new
threshold calibration. Prior lateral boundaries and fitted thresholds predate
these motion/planning changes and require recalibration before reuse.

Transport tests cover no-vision deadline expiry, delayed first write, early
cancellation, stale repeats, sequential turns, and distinct 0.3/0.5 degree holds.
FSM/adaptive tests cover actual TX duration and once-only learning. The insertion
fine-turn minimum remains 0.5 degrees. Existing adaptive slope estimates physical
gain from actual active time; it does not remove a vision-sampled STOP delay or
add candidates below the configured minimum. Lowering that minimum is a separate
planner setting whose hardware repeatability has not been established here.
