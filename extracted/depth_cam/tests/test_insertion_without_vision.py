import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4.top import CalibrationFSMV4
from calib.fsm_v4 import config as cfg
from calib.fsm_v4.motion import forward_seconds


class InsertionWithoutVisionTests(unittest.TestCase):
    def setUp(self):
        with patch("calib.fsm_v4.top.CommandExecutor"), patch("builtins.print"):
            self.fsm = CalibrationFSMV4()
        self.fsm._pipeline_started_mono = 100.0
        self.fsm._accept_insertion(SimpleNamespace(pallet_z_m=1.8), [])
        self.fsm._observe_pose = Mock(side_effect=AssertionError("PnP must be bypassed"))
        self.fsm._stable_observation = Mock(side_effect=AssertionError("No visual settle"))

    def step_at(self, now):
        with patch("calib.fsm_v4.top.time.monotonic", return_value=now):
            return self.fsm.step(False, None, None, None, None)

    def test_missing_pnp_through_entire_insertion_stops_and_completes_once(self):
        self.assertTrue(self.fsm.skip_detection)
        self.step_at(100.0)
        hold = forward_seconds(1.8 - cfg.INSERT_CAMERA_Z_REMAINDER_M)
        self.assertEqual(self.fsm.state, "INSERT_DRIVE")
        self.assertAlmostEqual(self.fsm._translation_deadline_mono, 100.0 + hold)
        self.step_at(100.0 + hold - .01)
        self.assertEqual(self.fsm.state, "INSERT_DRIVE")
        self.step_at(100.0 + hold)
        self.assertEqual(self.fsm.state, "INSERT_SETTLE")
        self.fsm.execu.exec.assert_called_with("STOP")
        self.step_at(100.0 + hold + cfg.STOP_MIN_SETTLE_SEC + .01)
        self.assertEqual(self.fsm.state, "DONE")
        self.step_at(120.0)
        self.assertEqual(self.fsm.state, "DONE")
        self.fsm.execu.exec.assert_called_with("STOP")
        self.fsm._observe_pose.assert_not_called()
        self.fsm._stable_observation.assert_not_called()

    def test_ready_can_wait_without_pnp_when_auto_insert_disabled(self):
        with patch.object(cfg, "AUTO_INSERT_ENABLED", False):
            self.step_at(100.0)
            self.step_at(110.0)
        self.assertEqual(self.fsm.state, "READY_TO_INSERT")
        self.fsm.execu.exec.assert_called_with("STOP")

    def test_timeout_still_stops_the_vehicle(self):
        self.step_at(100.0)
        self.step_at(100.0 + cfg.INSERT_MAX_TOTAL_SEC)
        self.assertEqual(self.fsm.state, "FAILED")
        self.assertEqual(self.fsm.failure_reason, "insertion total timeout")
        self.fsm.execu.exec.assert_called_with("STOP")
        self.step_at(125.0)
        self.fsm._observe_pose.assert_not_called()

    def test_invalid_or_clamped_targets_do_not_start_motion(self):
        for target in (float("nan"), -1.0, 100.0):
            with self.subTest(target=target):
                self.fsm.state = "READY_TO_INSERT"
                self.fsm._insert_remaining_m = target
                self.fsm.execu.exec.reset_mock()
                self.step_at(100.0)
                self.assertEqual(self.fsm.state, "FAILED")
                self.assertNotIn(("FWD",), [c.args for c in self.fsm.execu.exec.call_args_list])

    def test_telemetry_is_time_not_stale_pose_distance(self):
        self.step_at(100.0)
        with patch("calib.fsm_v4.top.time.monotonic", return_value=101.0):
            status = self.fsm.motion_target_status
        self.assertEqual(status["unit"], "s")
        self.assertEqual(status["current_value"], 1.0)
        self.assertEqual(status["kind"], "timed_insertion")
        self.fsm.observe_debug_telemetry(False, None, None, None, None)
        self.fsm._observe_pose.assert_not_called()

    def test_reset_reenables_detection(self):
        self.fsm.reset()
        self.assertFalse(self.fsm.skip_detection)
        self.assertIsNone(self.fsm._insert_accepted_z_m)

    def test_runtime_does_not_wait_for_frames_or_run_inference_after_acceptance(self):
        import main_rec as runtime
        with (patch.object(runtime, "realsense_check_or_exit"),
              patch.object(runtime, "rs") as rs,
              patch.object(runtime, "run_initialisation"),
              patch.object(runtime, "configure_can_enabled"),
              patch.object(runtime, "can_init"),
              patch.object(runtime, "configure_can_tx_observer"),
              patch.object(runtime, "TraceLogger"),
              patch.object(runtime, "setup_video_writer_filename", return_value=("preview.mp4", "raw.mp4")),
              patch.object(runtime, "Perception") as perception,
              patch.object(runtime, "pose_from_visible_kpts_pnp") as pnp,
              patch.object(runtime.cv2, "imshow"),
              patch.object(runtime.cv2, "waitKey", return_value=27),
              patch.object(runtime.cv2, "destroyAllWindows"),
              patch("builtins.print")):
            rs.pipeline.return_value.poll_for_frames.return_value = None
            from ui.diagram_v4 import draw_fsm_v4_diagram_panel
            self.fsm._pipeline_started_mono = __import__("time").monotonic()
            runtime.main(fsm_factory=lambda **kwargs: self.fsm, camera_enabled=True,
                         can_enabled=False, diagram_drawer=draw_fsm_v4_diagram_panel)
        rs.pipeline.return_value.wait_for_frames.assert_not_called()
        perception.return_value.infer_front.assert_not_called()
        pnp.assert_not_called()
        self.assertEqual(self.fsm.state, "INSERT_DRIVE")


if __name__ == "__main__":
    unittest.main()
