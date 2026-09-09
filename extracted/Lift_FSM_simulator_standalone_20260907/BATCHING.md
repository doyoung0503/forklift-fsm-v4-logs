# Live tick batching

Live /api/session uses BatchedSession. Each run owns one subprocess with the
independent world/sensors and CurrentFSMDriver. The web server does not import
the real FSM/CAN implementation. Real hardware execution is unchanged.

The server requests up to0.1 simulation seconds (maximum120 inner ticks).
The worker calls the original process_runtime.Session.step for every tick:
observe/request model, sample IMU, call FSM, apply current CAN, snapshot/log,
apply future CAN while advancing plant, then repeat using the updated world.
No held-input tick array, dropped CAN messages, or enlarged internal timestep
is used. Every frame and trace row is retained. Native window events are pumped
within the inner loop. Pause takes effect after the current batch completes.

The old per-tick subprocess Session remains available for reference tests and
offline tools. Live report transfers use a temporary file to avoid overflowing
the bounded JSON IPC response with an entire run. Worker cleanup removes it.

Validation: full noisy yaw15/Z4 sequence (IMU95/90,drive sigma.188,turn sigma.05,
seed20260908) matched all5029 trace rows, CAN frames, events, and final pose.
638 batch calls instead of5029 tick calls. One headless local measurement with
logging enabled in both modes:12.22s reference vs11.27s batch. This is not a
guaranteed live-window speedup or real-hardware validation.
