# Headless parallel experiments

Restart serve_v4.py and reload the browser after installing this change.
In the multi-core batch section, enter comma-separated camera-relative X/Z/yaw values. A single value in each field tests a single placement. These form a Cartesian product.

Enter explicit comma-separated integer seeds, or leave that field blank to use the main seed + repeat index. Choose concurrency (1..min(32, CPU count)). Maximum 1,000 experiments per job; one active job per server.

Start closes the current interactive session and runs experiments without native windows, camera rendering, inference, or physical CAN. Virtual timing, noise and controller semantics are retained. Stop finishes already dispatched experiments and skips undispatched experiments. Browser visibility does not control server execution. Closing the server stops this service; jobs are not automatically resumed after restart.

Results stream into the table with placement/seed, FSM result/reason, virtual and compute time, collision and remaining distance. Per-placement success counts/rates are shown. CSV and JSON export buttons remain available. Replay recalculates the selected case with detailed capture and checks its trace hash; changed source/config can cause a mismatch.

Durable output: batch_results/<job-id>/manifest.json (accepted inputs), results.jsonl (one finished report per line), summary.json (final status and groups). Reports include source/config provenance and trace hash. No per-tick run logs are written by default. They are summary outcomes of the simulator, not physical insertion verification. Worker errors are individual ERROR rows.

The server uses a separate job lock, outside the live FSM lock. A spawned ProcessPoolExecutor limits concurrency; max_tasks_per_child=1 isolates each experiment's mutable FSM globals. Each child uses Session with LocalController, keeping CAN/world/IMU/model feedback inside the process without per-tick IPC. Full trace/frame capture and rendering are disabled, but the existing per-run world CAN history remains in memory until that case exits. Source files should remain unchanged during a batch; provenance in each result permits checking mixed source versions.

POST /api/batch/start: {cases:[{options:{x:0,z:4,yaw:15,seed:1},overrides:{}}],workers:2}
POST /api/batch/status: {id:"...",cursor:0} returns newly completed results, cursor, groups, status and output directory.
POST /api/batch/cancel: {id:"..."} stops further dispatch.

Validation: serial-vs-two-process two-seed experiment trace hashes and final states match; fresh process IDs differ; cancelled/invalid jobs and durable output tested. HTTP start/status tested without browser polling driving time; UI seed Cartesian product test passes. Measured example 3.53s serial vs2.05s parallel, not a hardware-independent speed guarantee.
