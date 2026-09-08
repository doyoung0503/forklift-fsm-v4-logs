import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4.top import CalibrationFSMV4
from calib.fsm_v4 import config as cfg


class StoppedPoseReuseTests(unittest.TestCase):
    def setUp(self):
        with patch("calib.fsm_v4.top.CommandExecutor"), patch("builtins.print"):
            self.fsm = CalibrationFSMV4()
        self.pose = self.fsm._pose_filter.seed_vehicle(0., 0., 2., 100.)
        self.approved = (self.pose, 0., .2)
        self.fsm._approved_stopped_observation = self.approved
        self.fsm._stable_observation = Mock(return_value=None)

    def observe(self, pose=None, now=100.1, margin=.2):
        with patch("calib.fsm_v4.top.time.monotonic", return_value=now):
            return self.fsm._stopped_decision_observation(pose or self.pose, 0., margin, [])

    def test_adjacent_decisions_share_window_without_refreshing_timestamp(self):
        for state in ("STANDOFF_VERIFY", "STAGING_PLAN", "FINAL_POSE_LOCK"):
            self.fsm._set_state(state)
            self.assertIs(self.observe(), self.approved)
        self.fsm._stable_observation.assert_not_called()
        self.assertEqual(self.fsm._approved_stopped_observation[0].measurement_mono, 100.)

    def test_expired_window_requires_new_samples_even_with_fresh_current_pose(self):
        current = replace(self.pose, measurement_mono=100.4)
        self.assertIsNone(self.observe(current, now=100.4))
        self.fsm._stable_observation.assert_called_once()
        self.assertIsNone(self.fsm._approved_stopped_observation)

    def test_changed_pose_or_reduced_visibility_requires_reobservation(self):
        for changes, margin in [
            ({"yaw_deg":cfg.STABLE_YAW_MEDIAN_TOL_DEG+1}, .2),
            ({"pallet_x_m":cfg.STABLE_X_MEDIAN_TOL_M+.1}, .2),
            ({"pallet_z_m":2.+cfg.STABLE_Z_MEDIAN_TOL_M+.1}, .2),
            ({}, .1), ({"measurement_mono":99.9}, .2),
        ]:
            with self.subTest(changes=changes, margin=margin):
                self.fsm._approved_stopped_observation = self.approved
                self.assertIsNone(self.observe(replace(self.pose, **changes), margin=margin))
                self.assertIsNone(self.fsm._approved_stopped_observation)

    def test_motion_recovery_and_reset_discard_the_window(self):
        for state in ("WAYPOINT_TURN", "WAYPOINT_DRIVE", "RECOVER_VISUAL", "FAILED"):
            self.fsm._approved_stopped_observation = self.approved
            self.fsm._set_state(state)
            self.assertIsNone(self.fsm._approved_stopped_observation)
        self.fsm._approved_stopped_observation = self.approved
        self.fsm._exec("FWD")
        self.assertIsNone(self.fsm._approved_stopped_observation)
        self.fsm._approved_stopped_observation = self.approved
        self.fsm.reset()
        self.assertIsNone(self.fsm._approved_stopped_observation)

    def test_settle_approval_reaches_final_check_on_the_next_frame(self):
        self.fsm._set_state("WAYPOINT_DRIVE_SETTLE")
        self.fsm._settle_started_mono = 95.
        self.fsm._settle_evaluation_started_mono = 99.
        self.fsm._stable_observation.return_value = self.approved
        with patch("calib.fsm_v4.top.time.monotonic", return_value=100.):
            self.assertIs(self.fsm._settle_observation(self.pose, 0., .2, []), self.approved)
        self.fsm._set_state("FINAL_POSE_LOCK")
        self.fsm._stable_observation.reset_mock()
        self.fsm._observe_pose = Mock(return_value=(replace(self.pose, measurement_mono=100.1), "ok"))
        self.fsm._vision_values = Mock(return_value=(0., .2))
        self.fsm._near_insertion_step = Mock(return_value=True)
        with patch("calib.fsm_v4.top.time.monotonic", return_value=100.1):
            lines = self.fsm.step(True, None, 2., 0., None)
        self.fsm._stable_observation.assert_not_called()
        self.fsm._near_insertion_step.assert_called_once()
        self.assertTrue(any("OBSERVE REUSE" in line[0] for line in lines))

    def test_one_missing_observation_invalidates_window_before_loss_timeout(self):
        self.fsm._set_state("STAGING_PLAN")
        self.fsm._observe_pose = Mock(return_value=(None, "detection unavailable"))
        with patch("calib.fsm_v4.top.time.monotonic", return_value=100.1):
            self.fsm.step(False, None, None, None, None)
        self.assertIsNone(self.fsm._approved_stopped_observation)

    def test_debug_resume_clears_approved_window(self):
        from main_rec import _FSMStageDebugGate
        gate = _FSMStageDebugGate(self.fsm, enabled=False)
        gate._reset_perception_window()
        self.assertIsNone(self.fsm._approved_stopped_observation)


if __name__ == "__main__":
    unittest.main()
