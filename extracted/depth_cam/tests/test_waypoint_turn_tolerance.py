"""No hardware: settled-turn acceptance and safe forward continuation."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4 import config as cfg, planner
from calib.fsm_v4.top import CalibrationFSMV4

TOP = 'calib.fsm_v4.top.'
PLAN = 'calib.fsm_v4.planner.'


class WaypointToleranceTests(unittest.TestCase):
    def setUp(self):
        self.fsm = object.__new__(CalibrationFSMV4)
        self.fsm.state = 'WAYPOINT_TURN_SETTLE'
        self.fsm._rotation = Mock()
        self.fsm._waypoint = SimpleNamespace(forward_m=.8)
        self.fsm._forward_used_m = 0.
        self.fsm._begin_translation = Mock()
        self.fsm._set_state = Mock()
        self.fsm._fail = Mock()
        self.pose = SimpleNamespace(pallet_z_m=3.0)

    def check(self, error, distance=.7, reached=False):
        with patch(TOP+'staging_position_reached', return_value=reached), \
             patch(TOP+'safe_straight_continuation_m', return_value=distance), \
             patch(TOP+'goal_vector_vehicle', return_value=(.2, 1.5, 1.6)):
            return self.fsm._continue_accepted_waypoint_turn(self.pose, error, {}, [])

    def test_actual_overshoot_and_inclusive_boundaries_continue(self):
        self.assertEqual(cfg.WAYPOINT_SETTLED_YAW_TOL_DEG, 3.)
        for error in (-2.156997, -3., 0., 3.):
            with self.subTest(error=error):
                self.fsm._begin_translation.reset_mock()
                self.assertTrue(self.check(error))
                self.fsm._begin_translation.assert_called_once_with(
                    'WAYPOINT_DRIVE', 'FWD', self.pose, .7, 'v4_waypoint_forward')
                self.fsm._fail.assert_not_called()

    def test_outside_tolerance_returns_to_existing_replan(self):
        for error in (-3.001, 3.001, float('nan')):
            self.assertFalse(self.check(error))
        self.fsm._rotation.reset.assert_not_called()
        self.fsm._begin_translation.assert_not_called()

    def test_other_rotation_modes_are_unchanged(self):
        for state in ('FACE_SETTLE', 'RECENTER_SETTLE', 'FINAL_SETTLE'):
            self.fsm.state = state
            self.assertFalse(self.check(2.))
        self.fsm._begin_translation.assert_not_called()

    def test_unsafe_continuation_replans_without_centering(self):
        self.assertTrue(self.check(2., distance=0.))
        self.fsm._fail.assert_not_called()
        self.fsm._set_state.assert_called_once_with('STAGING_PLAN')
        self.fsm._begin_translation.assert_not_called()

    def test_reached_position_goes_to_final_lock(self):
        self.assertTrue(self.check(2., reached=True))
        self.fsm._set_state.assert_called_once_with('FINAL_POSE_LOCK')
        self.fsm._begin_translation.assert_not_called()

    def test_error_is_relative_to_target_yaw_not_absolute_yaw(self):
        self.fsm._rotation_mode = 'waypoint'
        self.fsm._rotation_target_yaw_deg = -16.4427747
        error = self.fsm._rotation_error(SimpleNamespace(yaw_deg=-18.5997717), 0.)
        self.assertAlmostEqual(error, -2.156997)


class StraightContinuationSafetyTests(unittest.TestCase):
    def check(self, visible=.6, staging=.5, remaining=.8, longitudinal=-1.):
        with patch(PLAN+'goal_vector_vehicle', return_value=(.2, 1.5, (2.29)**.5)), \
             patch(PLAN+'staging_position_errors', return_value=(.2, longitudinal)), \
             patch(PLAN+'_maximum_visible_forward_m', side_effect=lambda p,t,d,m: (min(visible,d),1.)), \
             patch(PLAN+'_maximum_staging_safe_forward_m', side_effect=lambda p,t,d: min(staging,d)):
            return planner.safe_straight_continuation_m(
                SimpleNamespace(rot_x_pallet_m=.3), .8, remaining, {})

    def test_rechecks_visibility_and_staging_and_remaining_budget(self):
        self.assertAlmostEqual(self.check(), .5)
        self.assertAlmostEqual(self.check(visible=.3), .3)
        self.assertAlmostEqual(self.check(remaining=.25), .25)

    def test_insufficient_safe_distance_or_overtravel_blocks(self):
        self.assertEqual(self.check(visible=.05), 0.)
        self.assertEqual(self.check(remaining=0.), 0.)
        self.assertEqual(self.check(longitudinal=0.), 0.)


if __name__ == '__main__':
    unittest.main()
