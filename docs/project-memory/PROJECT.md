# Project Memory

## Lateral threshold calibration (2026-09-09)

- Current applied results: `extracted/Lift_FSM_simulator_standalone_20260907/verification/lateral_recalibrated_20260909/README.md`. Recalibrated after TX-timed rotation, near-range replanning and adaptive forward cap; scripts `calibrate_lateral.py`, `summarize_lateral.py`, `verify_adaptive_coarse.py`.
- Audit: zero added drive noise still had predictive STOP up to12cm ahead with zero plant coast. D4/L.55 first .8m drive moved .684696m; disabling prediction moved .807881m. Rotation selected endpoint fit already includes inertia; extra coast subtraction0. CAN-duration model and observed endpoints match.
- New D3/3.5/4/4.5/5m inner sampled success bounds: .276171875/.41875/.65/.85625/.848671875m (both sides). D is pallet-normal pivot distance, initial heading faces pallet centre. D3 and D5 have disconnected success regions; first observed inner failures are guarded, not largest successes.
- APPLIED: abs(rot_x_pallet_m) > .10 + .190*max(0,min(D,5)-2.26), D=-rot_z_pallet_m; replaces yaw trigger. Domain measured3..5m; farther clamp5, closer approach term shrinks to0 at staging. Lower envelope .2380701 x80%, round down.201 search/audit trials,44/44 candidate checks,20/20 applied-FSM/IMU routing+completion cases,126 focused tests pass. No hardware/noisy reliability claim. Prior .169 proposal is historical only.
- Subsequent diagnosis: all26 collision-labelled search failures clear the FSM nine-block sweep even through actual stop; old simulator continuous walls intersect empty block gaps(depth.804..847m). User requested removal: simulator collision computation/classification and FORK CLEAR badge removed; FSM geometry stays. Legacy collision/clearance fields are null. D3/L.28125 andD5/L.855625 now pass; formula unchanged. See verification/recalibrated_failure_diagnosis_20260909 and simulator_collision_removed_20260909.
- Insertion candidate refinement now APPLIED in shared planner: try old candidates first, then subdivide failed intervals (including both-endpoints-fail) until success or width<=INSERT_TURN_REFINE_RESOLUTION_DEG=.05. Full geometry/yaw/visibility checks and minimum executed.5deg retained; normal waypoint candidate generation unchanged.72 focused tests pass;6 actual-FSM virtual-CAN cases pass, including D5/L+-.91125 previously failing (now selects+/-5.75deg earlier). See verification/insertion_turn_refinement_20260909. Threshold coefficients unchanged; no hardware run.

## IMU initial correction (2026-09-08)

- Implemented through `calib/fsm_v4/coarse.py` and `calib/imu_stream.py`; `main_rec_v4.py` real execution binds a RealSense IMU callback via main_rec. Default enabled, one-shot after initial facing; current trigger is calibrated distance-adaptive lateral (above), yaw only controls turn geometry.
- Sequence: stopped gyro bias estimate → tangent IMU rotation → existing forward_seconds(abs(rot_x_pallet_m)) timed FWD → opposite relative 90-degree IMU rotation → stopped PnP reacquisition → existing FSM. Final turn is relative to its own actual start, per user request.
- UI diagram has six phases with Initial Correction as phase 2 and live angle/time progress. Debug mode runs the coarse sequence from COARSE_PREPARE to COARSE_REACQUIRE as one macro step.
- Limits include freshness, optional TX-thread command lease, heading/overshoot bounds, distance/time bounds and front-plane clearance; this is not full swept-volume obstacle checking. Hardware validation remains outstanding.
- Validation: 59 selected offline tests passed (coarse, UI, initial visibility, settle, forward model, distance routing, insertion, launcher); changed Python files compile; six-phase render inspected. Guide: `extracted/depth_cam/IMU_INITIAL_CORRECTION.md`.

- Last updated: 2026-09-08
- Simulator virtual IMU supports gyro scale, fixed bias and seeded Gaussian rate noise integrated into yaw, with UI inputs and URL presets. These settings apply only to simulation; default IMU remains ideal. Manual noisy yaw40/Z4 scenario prepared on 2026-09-08; results pending user Run.
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

- Coarse travel caps disabled via COARSE_TRAVEL_LIMITS_ENABLED=False: no5m upper distance or15s FWD cap for coarse planning; normal drive limits and IMU/total100s watchdog remain.

- Coarse IMU: initial STOP waits2s before1.5s bias sampling; post-motion reference/transition uses fresh samples after2s continuous quiet.

- Live browser clock now retains busy time and unprocessed capped batches by charging only actual displayed sim progress; Pause/Run resets baseline. Background RAF throttling and compute limits are unchanged.

- Live simulation Run/Pause now uses server-owned background clock; browser only polls recorded frames and presents synchronized views. Hidden tabs no longer own physics/FSM advancement. Background timing file: run_server_performance.jsonl. Browser/native presentation updates when browser polls.

- Live sessions now use a world+FSM batch subprocess; <=.1sim second per IPC, unchanged inner Session.step feedback order. Per-tick reference retained;5029-tick parity test passes. See BATCHING.md.

- Normal waypoint forward cap is now adaptive: abs(pallet-frame pivot lateral)<=.10m ->1.50m, .10..20m linear to.80m, >=.20m ->.80m. Shared planner helper applies to predictive/legacy planning and settled continuation. Extensions stay within predicted lateral +/- .20m; existing visibility/staging/goal/budget constraints remain. Config snapshot includes thresholds. Previous lateral calibration predates this policy and needs rerun before reuse.

- Native simulator window routine tick/idle event polling is throttled to30Hz wall time; synchronized display forces a flush before ACK. Pump attempt/executed counters distinguish throttling.

- Normal fitted rotation STOP now belongs to CAN TX deadline (ROT_TX_TIMED_STOP_ENABLED=True), anchored to first successful host movement write; independent of vision30Hz. Adaptive slope learns from actual write-to-STOP hold. Simulator mirrors deadline events;117 targeted tests pass; hardware timing unverified.
- Below cameraZ2.2, unavailable insertion-only alignment can replan/drive within remaining staging room and existing safety/action limits, then recheck insertion. Regression results: D4.5/L.64125 and.7 pass; D4/L+-.6 pass using near replans; some closer/high-lateral paths still fail visibility. See simulator verification/rotation_near_replan_20260909/README.md. Previous lateral thresholds require recalibration.

- Headless multi-core placement/seed batch UI + /api/batch API implemented via parallel_batch.py; fresh spawned child per experiment, no rendering/detailed tick logs. Summaries persisted in batch_results. See PARALLEL_BATCH.md; server restart required to load new routes.

- Simulator errors now zero-mean additive: drive/rotation per-segment relative sigma.06079568 defaults (90% within+/-10%), IMU run-constant gain bias, endpoint sigma5deg at true90deg (no per-sample jitter). Retired fixed gain/bias inputs normalized to1/0. See ZERO_MEAN_ERRORS.md; old-run replay semantics changed.
