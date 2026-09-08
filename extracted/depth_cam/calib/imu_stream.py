"""All-sample gyro integration and a latest-video queue for RealSense callbacks."""
import math
import queue
import threading
import time


class ImuVideoStream:
    def __init__(self, rs, sign=1.0, max_gap=0.25):
        self.rs, self.sign, self.max_gap = rs, sign, max_gap
        self.video = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.yaw = 0.0
        self.timestamp = None
        self.received = None
        self.rate = 0.0
        self.error = None

    def feed_gyro(self, rate_rad_s, timestamp_ms, received=None):
        now = time.monotonic() if received is None else received
        with self.lock:
            if not math.isfinite(rate_rad_s) or not math.isfinite(timestamp_ms):
                self.error = "non-finite IMU sample"
                return
            if self.timestamp is not None:
                dt = (timestamp_ms - self.timestamp) / 1000.0
                if dt <= 0.0 or dt > self.max_gap:
                    self.error = "IMU timestamp discontinuity"
                    return
                self.yaw += self.sign * math.degrees(rate_rad_s) * dt
            self.timestamp = timestamp_ms
            self.received = now
            self.rate = self.sign * math.degrees(rate_rad_s)

    def snapshot(self):
        with self.lock:
            return self.yaw, self.received, self.rate, self.error

    def __call__(self, frame):
        try:
            frames = frame.as_frameset() if frame.is_frameset() else None
            for item in frames if frames else [frame]:
                if item.is_motion_frame() and item.get_profile().stream_type() == self.rs.stream.gyro:
                    self.feed_gyro(item.as_motion_frame().get_motion_data().y, item.get_timestamp())
            if frames and frames.get_color_frame() and frames.get_depth_frame():
                try:
                    self.video.put_nowait(frames)
                except queue.Full:
                    try:
                        self.video.get_nowait()
                    except queue.Empty:
                        pass
                    self.video.put_nowait(frames)
        except Exception as exc:
            with self.lock:
                self.error = f"IMU callback failed: {exc}"

    def poll_for_frames(self):
        try:
            return self.video.get_nowait()
        except queue.Empty:
            return None

    def wait_for_frames(self):
        try:
            return self.video.get(timeout=5.0)
        except queue.Empty as exc:
            raise RuntimeError("RealSense video timeout") from exc
