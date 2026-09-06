from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


DEPTH_CAM_DIR = Path(__file__).resolve().parents[1]
if str(DEPTH_CAM_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_CAM_DIR))

from calib.fsm_v4.top import CalibrationFSMV4  # noqa: E402
from ui.diagram_v4 import draw_fsm_v4_diagram_panel  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
