# Current FSM simulation audit (2026-09-09)

92 headless experiments, 4 processes, 43.1s wall. 74 passed / 18 did not pass. All results have one identical FSM source ID (see audit_summary.json). Controller/config not modified for this audit. Current source uses lateral-dependent coarse trigger, adaptive forward distance, TX-timed rotation stop and insertion candidate refinement.

| Group | Passed/total |
|---|---|
| Nominal: Z2/3/4m, X-.3/0/.3m, yaw -10/-5/0/5/10 |45/45|
| Large yaw at X0/Z4: +/-10,15,20,25,30,35,40 |14/14|
| Drive/IMU error: yaw -25/-15/0/15/25, seeds11/12/13 |2/15|
| Perception noise: yaw -15/0/15, seeds11/12/13 |5/9|
| Lateral stress: X+/-.6m, Z2/3/4, yaw0 |6/6|
| Observation loss: t8..11s, yaw -15/0/15 |2/3|

Baseline: relative camera coordinates, ideal model and actuation, fixed front face, 30Hz, camera height.5m, seed20260908. Drive/IMU group: Gaussian drive gain sigma=.187997 (expected absolute15%), rotation gain sigma=.05, fixed IMU gain95/90. This IMU condition is NOT independent random 5-degree MAE. Perception: XYZ sigma .01/.01/.02m, yaw sigma1deg, delay sigma.05s and dropout.05. All experiments retain existing predictive STOP etc; zero added error does not disable controller compensation. Results are coverage samples, not operational success probabilities.

## Findings
1. Nine runs report FSM DONE but fail simulator remaining-distance check. Residual insertion .09765..43160m vs tolerance.05m. Case59 (yaw-25,seed11): accepted camera Z2.195, requested1.895m insertion; timed FWD61.1167..69.3167s; DONE71.3167s with FSM remaining0 and true remaining.431595m. top.py INSERT_DRIVE/INSERT_SETTLE uses fitted time and sets remaining0 without final movement feedback. This is a completion verification limitation, not a crash.
2. Four drive-noise runs hit forward timeout. Case65 X0/Z4/yaw0/seed11 requests1.5m at2.6667s and aborts10.5s (7.8333s later). Actual advance about1.38067m at abort. Timeout margin is1s beyond fitted duration; lower motion gain can exhaust this budget before the predictive arrival gate. Keep safeguards; review duration adaptation rather than simply removing timeout.
3. Four perception-noise runs fail after two waypoint cycles without progress. Case78 yaw0/seed12 repeatedly starts forward then stops on insertion_range_entered at~2.2m, ending25.7452s. Cycles3/4 report only.0336/.0941m visual progress before STOP. The source uses instantaneous pose Z<=2.2 unless near-approach is active. Boundary handling under noisy observations needs review (hysteresis/stable gating and route consistency); this audit does not prove a particular fix.
4. One3s observation loss run fails PnP recovery: loss starts8s, STOP9.0333s, timeout10.0333s before observations return11s. This is the configured loss policy, not proof of an algorithm bug.

No worker exceptions or simulated collisions observed. Collision result only covers the simulator geometry; physical blockage/full pallet insertion is not proven. Four detailed reruns (59/65/78/90) exactly match batch trace hashes, with case_N.json and detail_N/ logs. manifest.json captures all inputs; results.jsonl contains source/config provenance. Reproduce with verify_current_matrix.py; inspect_matrix_failures.py records selected failures.
