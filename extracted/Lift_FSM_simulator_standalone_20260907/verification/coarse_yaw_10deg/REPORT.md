# IMU coarse alignment: seven yaw scenarios

Relative camera X=0 m, Z=4 m; positive yaw. Per-run COARSE_YAW_TRIGGER_DEG=10; physical default remains 25. Trigger uses strict abs(yaw)>10. No model noise, latency, dropout, or additional drive error. Ideal IMU from CAN-driven world heading. 30 Hz model, camera height 0.65 m, fixed front face, FOV checks enabled.

|Initial yaw|Coarse completed|Post-coarse yaw|Coarse duration s|Final state|Total s|
|---|---|---|---|---|---|
|10|False|-|-|FAILED|33.567|
|15|True|-0.0071|28.363|DONE|59.083|
|20|True|0.0309|29.323|DONE|59.510|
|25|True|-0.0521|30.243|DONE|56.087|
|30|True|-0.0141|31.103|DONE|56.323|
|35|True|0.0239|31.903|DONE|56.290|
|40|False|-|-|FAILED|2.333|

10 degrees: skipped coarse; failed later at FINAL_POSE_LOCK (front pocket clearance).
40 degrees: required lateral travel (4+0.68)*sin(40)=3.008246 m exceeds COARSE_MAX_LATERAL_M=3 m.

All seven model+IMU trace replays match FSM/CAN outputs. Related tests: 3 requested-inference + 11 process-runtime passed.
Limitations: ideal planar IMU; no bias/noise/dropout or extra physical drive error. Insertion approval uses the new nine-block gate; simulator collision display remains its previous model. No hardware execution.
Reproduce: python test_coarse_scenarios.py. Per-case results and run logs are in this directory.