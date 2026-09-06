"""Post-command guard timing without constructing a live command executor."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

DEPTH_CAM_DIR = Path(__file__).resolve().parents[1]
if str(DEPTH_CAM_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_CAM_DIR))

from calib.fsm_v4 import config as cfg
from calib.fsm_v4.top import CalibrationFSMV4


class SettleGuardTests(unittest.TestCase):
    def setUp(self):
        self.fsm = object.__new__(CalibrationFSMV4)
        self.fsm.state = "WAYPOINT_SETTLE"
        self.fsm._exec = Mock()
        self.fsm._update_rotation_coast = Mock()
        self.fsm._stable_observation = Mock(return_value=None)
        self.fsm._fail = Mock()
        self.fsm._settle_started_mono = 100.0
        self.fsm._settle_evaluation_started_mono = None
        self.fsm._samples = ["pre-guard observation"]
        self.pose = SimpleNamespace(yaw_deg=0.0, pallet_x_m=0.0, pallet_z_m=2.0,
                                    measurement_mono=102.0)
        self.fsm._pose_filter = SimpleNamespace(seed_vehicle=Mock(return_value=self.pose))

    def observe_at(self, now):
        with patch("calib.fsm_v4.top.time.monotonic", return_value=now):
            return self.fsm._settle_observation(self.pose, 0.0, 0.2, [])

    def test_guard_is_two_seconds_and_stability_keeps_four_seconds(self):
        self.assertEqual(cfg.STOP_MIN_SETTLE_SEC, 2.0)
        self.assertEqual(cfg.STOP_MAX_SETTLE_SEC, 4.0)
        self.fsm._set_state = Mock()
        self.fsm._begin_settle("WAYPOINT_SETTLE", now=100.0)
        self.fsm._set_state.assert_called_once_with("WAYPOINT_SETTLE", 6.0)

    def test_no_stability_decision_before_two_seconds(self):
        self.assertIsNone(self.observe_at(101.999))
        self.fsm._exec.assert_called_once_with("STOP")
        self.fsm._stable_observation.assert_not_called()
        self.fsm._pose_filter.seed_vehicle.assert_not_called()

    def test_two_seconds_starts_fresh_stability_window_not_motion(self):
        self.assertIsNone(self.observe_at(102.0))
        self.assertEqual(self.fsm._settle_evaluation_started_mono, 102.0)
        self.assertEqual(self.fsm._samples, [])
        self.fsm._pose_filter.seed_vehicle.assert_called_once()
        self.fsm._stable_observation.assert_not_called()
        stable = (self.pose, 0.0, 0.2)
        self.fsm._stable_observation.return_value = stable
        self.assertEqual(self.observe_at(102.1), stable)
        self.assertTrue(all(call.args == ("STOP",) for call in self.fsm._exec.call_args_list))

    def test_instability_timeout_is_four_seconds_after_guard(self):
        self.observe_at(102.0)
        self.observe_at(105.999)
        self.fsm._fail.assert_not_called()
        self.observe_at(106.0)
        self.fsm._fail.assert_called_once()


if __name__ == "__main__":
    unittest.main()
