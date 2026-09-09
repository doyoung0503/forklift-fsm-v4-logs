# Simulator performance log

Restart `serve_v4.py` and refresh the browser after installing this change.
Each live browser step appends `run_performance.jsonl` to its usual run_logs folder.
Replay and initial placement do not create performance samples.

Fields (durations in milliseconds):
- `wall_utc`: actual server receipt time, unlike synthetic timestamps in existing logs.
- `browser_mono_ms`: browser monotonic completion timestamp.
- `sim_time_s`, `sim_advance_s`, `state`, `command`: displayed simulation progress.
- `interval_ms`: time between display completions; null for first sample after Run/resume/reset.
- `cycle_ms`: this step request through completed browser drawing; excludes performance-log upload.
- `step_request_ms`: HTTP step round trip and response decoding.
- `server_step_ms`: server simulation batch, including model generation, physics, controller IPC and normal logs.
- `controller_tick_ms`: subset of server step spent waiting for FSM worker ticks, including IPC.
- `camera_request_ms`: camera HTTP round trip; `server_camera_ms` includes renderer worker IPC; `renderer_ms` is the renderer's own timer.
- `present_request_ms`: native-window presentation round trip; `server_present_ms` is its server-side subset.
- `browser_paint_ms`: browser DOM/canvas commands including reconstruction of historical event text. This does not measure GPU completion/compositing.
- `display_total_ms`: camera/presentation/browser drawing total.
- `frame_count`, `time_scale`, `document_hidden`: accumulated history and UI context.

Compute observed speed as `1000 * sim_advance_s / interval_ms`, excluding null intervals, and compare windows at the same time_scale. Background tab throttling is indicated by document_hidden. Times are nested: do not add server timing to its HTTP round trip. Performance-log upload and frame scheduling are included in the following interval, not cycle_ms. The logger itself adds a small HTTP/file-write cost. Timing collection does not fix the existing busy-clock behavior, so this run can diagnose it without changing scheduling.


## Live batch profiling (2026-09-09)
Restart the simulator server and reload the browser before the next run.
`run_server_performance.jsonl` records cumulative `phase_totals` and `transport_totals` approximately once per wall second and at completion. Subtract consecutive samples for interval costs; counters persist across Pause/Run, while server_elapsed_s resets on Play.

- sensors_ms: observation/IMU/input preparation (includes first logging startup).
- controller_ms: local controller request inclusive of window_pump_ms and fsm_ms; do not sum parent and children.
- window_pump_ms/calls: native event pump inside ticks; fsm_ms/calls: driver.tick only.
- outer_pump_ms/calls: native event pump outside requests (also includes idle time event processing); separate from tick totals.
- world_snapshot_ms: immediate CAN, state/geometry snapshot and transitions.
- world_advance_ms: timed world advance with scheduled CAN.
- trace_ms: trace construction, JSON hash and capture bookkeeping.
- logging_ms: per-tick logs/flush, final summary on last tick.
- tick_total_ms/ticks: complete successful tick cost.
- transport roundtrip_ms: batch request through returned result. worker_operation_ms: worker dispatch/step/result construction. transport_residual_ms is their difference, including worker scheduling, outer event processing, JSON encoding/decoding and pipe waits; it is NOT pure CAN or wire time. merge_ms: parent frame merge; requests: batch count.

`run_performance.jsonl` is restored for server-clock live playback. interval_ms measures successive completed native ACK/browser draw callbacks, not monitor scanout/FPS. display_wall_utc is browser completion time; wall_utc is delayed server receipt. poll_request_ms and poll_lock_wait_ms distinguish response wait from measured server lock wait. Existing camera, native presentation, browser paint timings are retained; present_lock_wait_ms is separate from server_present_ms. Samples piggyback on the following poll; final sample is flushed separately. Abrupt tab close may lose the last queued sample. Pausing resets the display interval baseline on resume.
Timing probes do not change virtual time, tick order or CAN commands. They add small measurement overhead. No speed optimization is included in this change.

Native event pump tick/idle calls now share a 30Hz wall-time gate. window_pump_calls/outer_pump_calls count attempts; *_pump_executed count actual calls passing the gate. Explicit synchronized presentation still forces event processing before its ACK and resets the same gate. Thus the 30Hz limit applies to routine event checks, not forced presentation or window shutdown.
