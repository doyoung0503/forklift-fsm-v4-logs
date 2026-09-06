from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


DEPTH_CAM_DIR = Path(__file__).resolve().parents[1]
if str(DEPTH_CAM_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_CAM_DIR))

from calib.tracelog import TraceLogger  # noqa: E402


class RawVideoMappingTraceTests(unittest.TestCase):
    def test_timing_rows_record_exact_raw_video_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = str(Path(temporary) / "recording")
            tracer = TraceLogger(base, enabled=True)
            try:
                for frame_i, raw_index in ((35, None), (36, 0), (37, 1), (42, 2)):
                    tracer.log_inference_timing(
                        frame_i=frame_i,
                        raw_video_frame_index=raw_index,
                        fsm_state="TEST",
                        align_sub="TEST",
                        camera_frame_number=frame_i,
                        camera_sensor_timestamp_ms=frame_i * 10.0,
                        camera_timestamp_domain="hardware_clock",
                        camera_input_host_mono_ms=frame_i * 10.0,
                        inference_start_host_mono_ms=None,
                        inference_end_host_mono_ms=None,
                        pose_result_host_mono_ms=None,
                        inference_ran=False,
                        model_det_ok=False,
                        pnp_ok=False,
                        yaw_deg=None,
                        pos_x_m=None,
                        pos_z_m=None,
                        center_bearing_deg=None,
                    )
            finally:
                tracer.close()

            with Path(base + "_inference_timing.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                [row["raw_video_frame_index"] for row in rows],
                ["", "0", "1", "2"],
            )
            self.assertEqual(
                [row["frame_i"] for row in rows],
                ["35", "36", "37", "42"],
            )


if __name__ == "__main__":
    unittest.main()
