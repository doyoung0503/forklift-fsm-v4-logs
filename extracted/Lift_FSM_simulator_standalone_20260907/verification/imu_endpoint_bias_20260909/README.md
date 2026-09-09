# IMU endpoint bias verification ? 2026-09-09

Model: run_constant_gain_v1. Sample epsilon ~ N(0,(sigma_deg/90)^2) once per session. Unwrapped relative yaw and angular rate share gain 1+epsilon. No per-sample jitter; no stationary drift. Default sigma5deg refers to the final reading error for a true90deg rotation, not exactly5deg per turn. Across seeds the bias is zero-mean; within one run it stays fixed.

Copied cases53 and140 from zero_mean_matrix_20260909/manifest.json with all options unchanged, using fresh actual FSM subprocesses via process_runtime.Session. Results saved separately; historical reports retained.

| Case | Sampled IMU gain | Result | Simulation seconds | Stationary samples | Maximum stationary yaw change |
|---|---:|---|---:|---:|---:|
| 53 | 1.0349316309 | DONE / success | 65.98 | 3536 | 0 degrees |
| 140 | 1.0584820063 | DONE / success | 60.9133 | 3214 | 0 degrees |

Stationary samples are trace IMU packets with zero rate and a preceding packet. Full reports include source/simulator provenance and sampled gain. This is targeted regression verification, not a rerun of the228-case matrix or a new lateral calibration.

Checks: unittest test_zero_mean_errors.py4passed; test_parallel_batch.py2passed (deterministic sequential/parallel parity, not all scenarios succeed). Endpoint tests cover synthetic95/90 gain, angle wrap, smooth rotation, stationary holds, reverse rotation, ideal sigma0 and30/120Hz invariance.

Server restarted and /api/session verified run_constant_gain_v1. Reload the browser and create a new session. Previously saved results retain the old per-sample noise interpretation.
