# Zero-mean error audit (2026-09-09)

228 experiments in52.72s on4 processes:213 requested-noise runs and15 diagnostic controls. Main result58/213 passed (27.23%),155 did not pass. One identical FSM source ID across all results. No FSM/config change made for this audit.

All main runs use additive zero-mean drive/rotation relative sigma.0607956831911769 (central90% within+/-10%) and independent additive IMU angle sigma5deg per sample. No biased IMU gain. Baseline perception ideal, fixed front face,30Hz, camera height.5m. Every placement uses seeds11,12,13.

| Group | Passed/total |
|---|---|
| Basic Z2/3/4m, X-.3/0/.3m, yaw-10/-5/0/5/10 |56/135|
| Large yaw X0/Z4, +/-10/15/20/25/30/35/40 |0/42|
| Lateral X+/-.6m, Z2/3/4, yaw0 |0/18|
| Additional perception noise (XYZ .01/.01/.02m, yaw1deg, delay sigma.05s, dropout.05), yaw-15/0/15 |2/9|
| Additional3s observation loss at t8..11s, yaw-15/0/15 |0/9|

Main failure reasons:136 coarse turn timeout or reversed IMU direction;14 FSM DONE with insertion distance remaining;3 PnP recovery timeout;2 first-turn overshoot beyond tolerance. These are controller/validation failures, not Python exceptions. Residual insertion in DONE-but-failed cases .05354..20484m (tolerance.05m).

## IMU control comparison
For the SAME X0/Z4/yaw-25/-15/0/15/25 and seeds11/12/13, keeping drive/rotation errors unchanged: IMU sigma5 gives2/15 success; IMU sigma0 gives14/15. This isolates IMU noise as a major issue in this sample, not a general success probability.

Case53 (yaw15/seed11) detailed rerun exactly matches batch trace hash. COARSE_ROTATE begins at5.843333s and fails6.033333s,0.19s later. True heading and gyro rate are still0 at both times. Reported angle changes6.87135->14.53057deg solely due synthetic angle noise. The current coarse controller subtracts the start reference and aborts when signed progress<-5deg. Independent5deg noise in both observations has difference sigma about7.07deg; the model can trigger the reverse guard before actual rotation starts. The combined failure label also includes timeout, so the136 group is not individually classified by timing here.

Timed final insertion remains another limitation: DONE does not verify actual insertion distance. Current collision and minimum-clearance fields are null for these runs (geometry acceptance belongs to FSM); do NOT interpret this audit as independent collision verification or proof of physical insertion.

Full inputs:manifest.json. Results with provenance:results.jsonl. Counts:audit_summary.json. Representative detailed report:case_53.json and detail_53/. Reproduce:verify_zero_mean_matrix.py; representative replay:inspect_zero_mean_failure.py. Existing earlier audit files remain unchanged. Counts are finite-condition test results, not real-world reliability estimates.
