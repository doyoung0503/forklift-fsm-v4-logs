from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


DEPTH_CAM_DIR = Path(__file__).resolve().parents[1]
if str(DEPTH_CAM_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_CAM_DIR))

from calib.fsm_v4.top import CalibrationFSMV4  # noqa: E402
from ui.diagram_v4 import _phase_index, draw_fsm_v4_diagram_panel  # noqa: E402


class MotionTargetStatusTests(unittest.TestCase):
    def _bare_fsm(self):
        return object.__new__(CalibrationFSMV4)

    def test_rotation_reports_requested_completed_and_remaining_angle(self):
        fsm = self._bare_fsm()
        fsm.state = "WAYPOINT_TURN"
        fsm._rotation = SimpleNamespace(
            active=True,
            plan=SimpleNamespace(target_deg=15.0),
            start_error_deg=15.0,
            last_error_deg=9.0,
            expected_delta_sign=-1.0,
            command="ROT_RIGHT",
        )

        status = fsm.motion_target_status

        self.assertEqual(status["kind"], "rotation")
        self.assertEqual(status["direction"], "RIGHT")
        self.assertAlmostEqual(status["target_value"], 15.0)
        self.assertAlmostEqual(status["current_value"], 6.0)
        self.assertAlmostEqual(status["remaining_value"], 9.0)
        self.assertAlmostEqual(status["progress_ratio"], 0.4)

    def test_translation_reports_live_distance_progress(self):
        fsm = self._bare_fsm()
        fsm.state = "WAYPOINT_DRIVE"
        fsm._translation_target_m = 0.8
        fsm._translation_command = "FWD"
        fsm._translation_start_rot = (1.0, 2.0)
        fsm._last_valid_pose = SimpleNamespace(
            rot_x_pallet_m=1.3,
            rot_z_pallet_m=2.4,
        )

        status = fsm.motion_target_status

        self.assertEqual(status["kind"], "translation")
        self.assertEqual(status["direction"], "FORWARD")
        self.assertAlmostEqual(status["target_value"], 0.8)
        self.assertAlmostEqual(status["current_value"], 0.5)
        self.assertAlmostEqual(status["remaining_value"], 0.3)
        self.assertAlmostEqual(status["progress_ratio"], 0.625)

    def test_search_explains_that_its_rotation_has_no_fixed_angle(self):
        fsm = self._bare_fsm()
        fsm.state = "SEARCH_SWEEP"

        status = fsm.motion_target_status

        self.assertEqual(status["kind"], "rotation")
        self.assertFalse(status["fixed_target"])
        self.assertEqual(status["direction"], "RIGHT")

    def test_diagram_renders_with_live_target_card(self):
        fsm = SimpleNamespace(
            state="WAYPOINT_DRIVE",
            cmd_status=SimpleNamespace(code="FWD"),
            motion_target_status={
                "kind": "translation",
                "direction": "FORWARD",
                "fixed_target": True,
                "target_value": 0.8,
                "current_value": 0.25,
                "remaining_value": 0.55,
                "progress_ratio": 0.3125,
                "unit": "m",
            },
        )

        image = draw_fsm_v4_diagram_panel(fsm, panel_size=(480, 900))

        self.assertEqual(image.shape, (480, 900, 3))
        self.assertGreater(int(image.sum()), 0)

    def test_initial_correction_and_final_alignment_have_visible_phases(self):
        for state, expected in [
            ("INITIAL_VISIBILITY_SWEEP", 0), ("INITIAL_POSE_ROTATE", 0),
            ("INITIAL_POSE_SETTLE", 0), ("STANDOFF_MOVE", 1),
            ("STAGING_PLAN", 2), ("FINAL_POSE_LOCK", 3),
            ("FINAL_ROTATE", 3), ("INSERT_DRIVE", 4), ("DONE", 4),
        ]:
            with self.subTest(state=state):
                self.assertEqual(_phase_index(SimpleNamespace(state=state)), expected)
        self.assertIsNone(_phase_index(SimpleNamespace(state="PRECHECK")))

    def test_recovery_and_failure_keep_interrupted_phase_until_replanning(self):
        fsm = self._bare_fsm()
        fsm.state = "WAYPOINT_DRIVE"
        fsm._samples = []
        fsm._failure_state = None
        fsm._set_state("RECOVER_VISUAL")
        self.assertEqual(_phase_index(fsm), 3)
        fsm._set_state("ACQUIRE_VERIFY")
        self.assertEqual(_phase_index(fsm), 3)
        fsm._failure_state = "ACQUIRE_VERIFY"
        fsm._set_state("FAILED")
        self.assertEqual(_phase_index(fsm), 3)
        fsm._set_state("STAGING_PLAN")
        self.assertEqual(_phase_index(fsm), 2)
        self.assertIsNone(fsm._visual_recovery_origin_state)


if __name__ == "__main__":
    unittest.main()
