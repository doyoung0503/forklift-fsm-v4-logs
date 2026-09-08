"""Start-distance routing uses stopped poses without hardware commands."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4.top import CalibrationFSMV4
from calib.fsm_v4.pose import VisualPoseFilter
from calib.fsm_v4.controllers import RotationController


class StartDistanceRoutingTests(unittest.TestCase):
    def run_start(self, z, x=0., yaw=0., state='ACQUIRE_VERIFY', staging_started=False):
        f = object.__new__(CalibrationFSMV4)
        f.tracer = None
        f.execu = Mock()
        f.status = Mock()
        f._pose_filter = VisualPoseFilter()
        f._rotation = RotationController()
        with patch('calib.fsm_v4.top.time.monotonic', return_value=100.):
            f.reset()
        if staging_started:
            f._alignment_started_mono = 100.
        f.state = state
        f._state_deadline_mono = 110.
        f._initial_visibility_complete = True
        f._begin_rotation = Mock(return_value=True)
        f._begin_translation = Mock()
        pose = VisualPoseFilter().seed(yaw, x, z, 101.)
        with patch.object(f, '_observe_pose', return_value=(pose, 'test')), \
             patch.object(f, '_stable_observation', return_value=(pose, 12., .2)), \
             patch('calib.fsm_v4.top.time.monotonic', return_value=101.):
            f.step(True, 1.1, z, yaw, x, vision_meta={'bbox_margin_norm': .2})
        return f

    def test_reacquired_staging_pose_never_restarts_face_even_above_four_m(self):
        f = self.run_start(4.3, x=.5, staging_started=True)
        self.assertEqual(f.state, 'STAGING_PLAN')
        f._begin_rotation.assert_not_called()
        f._begin_translation.assert_not_called()

    def test_middle_distance_enters_plan_without_face_turn_or_backing(self):
        for state in ('ACQUIRE_VERIFY', 'STANDOFF_VERIFY'):
            for z in (2.2, 2.201, 3., 3.9, 3.95, 4., 4.1):
                with self.subTest(state=state, z=z):
                    f = self.run_start(z, state=state)
                    self.assertEqual(f.state, 'STAGING_PLAN')
                    self.assertEqual(f._alignment_started_mono, 101.)
                    f._begin_translation.assert_not_called()
                    f._begin_rotation.assert_not_called()

    def test_near_aligned_pose_is_ready_for_insertion(self):
        f = self.run_start(1.8, yaw=-9.2)
        self.assertEqual(f.state, 'READY_TO_INSERT')
        f._begin_translation.assert_not_called()

    def test_near_correctable_pose_rotates_before_insertion(self):
        f = self.run_start(1.8, x=-.1, yaw=-9.2)
        self.assertEqual(f._begin_rotation.call_args.args[:2],
                         ('FINAL_ROTATE', 'insert_align'))
        f._begin_translation.assert_not_called()
        self.assertIsNone(f.failure_reason)

    def test_near_infeasible_pose_fails_without_translation(self):
        with patch('calib.fsm_v4.top.insertion_alignment_turn', return_value=None):
            f = self.run_start(2.199)
        self.assertEqual(f.state, 'FAILED')
        self.assertIn('insertion rotation', f.failure_reason)
        f._begin_rotation.assert_not_called()
        f._begin_translation.assert_not_called()

    def test_far_standoff_still_moves_forward(self):
        f = self.run_start(4.5, state='STANDOFF_VERIFY')
        self.assertEqual(f._begin_translation.call_args.args[:2],
                         ('STANDOFF_MOVE', 'FWD'))


if __name__ == '__main__':
    unittest.main()
