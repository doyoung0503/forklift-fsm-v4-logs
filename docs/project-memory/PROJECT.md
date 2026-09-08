# Project Memory

## IMU initial correction (2026-09-08)

- Implemented through `calib/fsm_v4/coarse.py` and `calib/imu_stream.py`; `main_rec_v4.py` real execution binds a RealSense IMU callback via main_rec. Default enabled, absolute model yaw >25 degrees, one-shot after initial facing.
- Sequence: stopped gyro bias estimate → tangent IMU rotation → existing forward_seconds(abs(rot_x_pallet_m)) timed FWD → opposite relative 90-degree IMU rotation → stopped PnP reacquisition → existing FSM. Final turn is relative to its own actual start, per user request.
- UI diagram has six phases with Initial Correction as phase 2 and live angle/time progress. Debug mode runs the coarse sequence from COARSE_PREPARE to COARSE_REACQUIRE as one macro step.
- Limits include freshness, optional TX-thread command lease, heading/overshoot bounds, distance/time bounds and front-plane clearance; this is not full swept-volume obstacle checking. Hardware validation remains outstanding.
- Validation: 59 selected offline tests passed (coarse, UI, initial visibility, settle, forward model, distance routing, insertion, launcher); changed Python files compile; six-phase render inspected. Guide: `extracted/depth_cam/IMU_INITIAL_CORRECTION.md`.

- Last updated: 2026-09-08
- Project root: `/Users/doyoung/Documents/ChatGPT/auto lift/forklift-fsm-v4-logs`
- Purpose: Forklift FSM v4 source and recording logs.
- Current status: Cloned from GitHub `main`; added IMU archive extracted and inspected on 2026-09-08.
- IMU package: `imu_rotation_code/depth_cam/rotation_return_test.py` integrates RealSense gyro Y for relative yaw, stops at 88 degrees by default, and compares stopped pallet PnP measurements before/after left-right and right-left round trips. Post-stop correction is fixed off in CLI defaults. CAN uses `calib/control.py`; the bundled `CAN/control_forklift_v2.py` is reference code.
- IMU validation: All 19 extracted files match ZIP contents; 16 Python files pass AST syntax parsing. Hardware execution was not performed. No gyro bias compensation or command-expiry watchdog was found in the inspected control path; camera-frame waits can delay STOP.
- Important files: `extracted/depth_cam/main_rec_v4.py` launches v4; `calib/fsm_v4/planner.py` computes fork-ray opening alignment; `top.py` authorizes insertion and executes timed forward; `config.py` contains geometry/tolerances. Paths under extracted/depth_cam unless stated.
- Run, test, and validation commands: To be documented when verified.
- Recent changes: Project memory initialized after clone.
- Insertion: user-measured 1.10 m pallet, nine 0.20 m blocks with 0.25 m gaps, single fork width 0.115 m; retained outer fork span 0.60 m. `insertion_geometry.py` tests continuous straight swept strips against all nine blocks enlarged by 0.01 m; planner and final approval share the gate. Target remains accepted camera Z minus 0.30 m with fitted-time/no-vision execution. See `extracted/depth_cam/docs/INSERTION_GEOMETRY.md`.
- Insertion validation: 50 focused offline tests passed; full depth_cam suite 153 passed / 3 torch import errors (156 total). No hardware run.
- Insertion limits: planar nominal-pose geometry only; 1 cm margin is not a bound on perception/drift/braking error. Current simulator collision display still uses its previous opening model. CAN and automatic insertion remain enabled in config; actual vehicle behavior unverified.
