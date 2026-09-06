"""Initial cropped detections trigger a bounded right sweep, without hardware."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4.top import CalibrationFSMV4
from calib.fsm_v4.pose import VisualPoseFilter
from calib.fsm_v4 import config as cfg
from calib.fsm_v4.controllers import RotationController


class InitialVisibilitySweepTests(unittest.TestCase):
    def setUp(self):
        self.f = object.__new__(CalibrationFSMV4)
        self.f.tracer = None
        self.f.execu = Mock()
        self.f.status = Mock()
        self.f._pose_filter = VisualPoseFilter()
        self.f._rotation = RotationController()
        with patch('calib.fsm_v4.top.time.monotonic', return_value=100.):
            self.f.reset()
        self.f.execu.reset_mock()
        self.cropped = VisualPoseFilter().seed(-35.57367, -1.55004, 3.96004, 101.)
        self.visible = VisualPoseFilter().seed(0., 0., 3.96, 101.)

    def step(self, pose, now, margin=.2):
        meta = {'bbox_margin_norm': margin}
        with patch.object(self.f, '_observe_pose', return_value=(pose, 'test')), \
             patch('calib.fsm_v4.top.time.monotonic', return_value=now):
            return self.f.step(pose is not None, 1.1, 3.96, 0., None, vision_meta=meta)

    def test_cropped_detection_corrects_toward_detected_side(self):
        for x, command in ((-1.55, 'ROT_LEFT'), (1.55, 'ROT_RIGHT')):
            with self.subTest(x=x):
                self.setUp()
                self.f.state = 'SEARCH_SWEEP'
                pose = VisualPoseFilter().seed(0., x, 3.96, 101.)
                self.step(pose, 101., margin=.01)
                self.assertEqual(self.f.state, 'INITIAL_POSE_ROTATE')
                self.f.execu.exec.assert_called_with(command)
                self.assertIsNone(self.f.failure_reason)

    def test_right_search_only_when_not_detected(self):
        self.f.state = 'SEARCH_SWEEP'
        self.step(None, 101.)
        self.f.execu.exec.assert_called_with('ROT_RIGHT')
        with patch.object(self.f, '_observe_pose', return_value=(None, 'no pose')), \
             patch('calib.fsm_v4.top.time.monotonic', return_value=102.):
            self.f.step(True, 1.1, None, None, None)
        self.f.execu.exec.assert_called_with('STOP')

    def test_loss_during_search_does_not_trigger_pnp_recovery(self):
        with patch('calib.fsm_v4.top.time.monotonic', return_value=101.):
            self.f._start_initial_visibility_sweep(101., [])
        self.step(None, 103.)
        self.step(None, 105.)
        self.assertEqual(self.f.state, 'INITIAL_VISIBILITY_SWEEP')
        self.f.execu.exec.assert_called_with('ROT_RIGHT')

    def test_only_full_front_stops_then_verifies(self):
        with patch('calib.fsm_v4.top.time.monotonic', return_value=101.):
            self.f._start_initial_visibility_sweep(101., [])
        self.step(None, 102., margin=.01)
        self.assertEqual(self.f.state, 'INITIAL_VISIBILITY_SWEEP')
        self.step(self.visible, 103.)
        self.assertEqual(self.f.state, 'ACQUIRE_VERIFY')
        self.f.execu.exec.assert_called_with('STOP')
        self.assertFalse(self.f._initial_visibility_complete)

    def test_timeout_stops_and_budget_cannot_restart(self):
        with patch('calib.fsm_v4.top.time.monotonic', return_value=101.):
            self.f._start_initial_visibility_sweep(101., [])
        deadline = self.f._initial_visibility_deadline
        self.step(self.visible, 102.)
        with patch('calib.fsm_v4.top.time.monotonic', return_value=103.):
            self.f._start_initial_visibility_sweep(103., [])
        self.assertEqual(self.f._initial_visibility_deadline, deadline)
        self.step(None, deadline)
        self.assertEqual(self.f.state, 'FAILED')
        self.f.execu.exec.assert_called_with('STOP')
        self.assertIn('time budget exhausted', self.f.failure_reason)

    def test_stable_initial_crop_enters_search_and_reset_clears_budget(self):
        self.f.state = 'ACQUIRE_VERIFY'
        self.f._state_deadline_mono = 110.
        with patch.object(self.f, '_stable_observation',
                          return_value=(self.cropped, -21., .01)):
            self.step(self.cropped, 101., margin=.01)
        self.assertEqual(self.f.state, 'INITIAL_POSE_ROTATE')
        self.f.execu.exec.assert_called_with('ROT_LEFT')
        with patch('calib.fsm_v4.top.time.monotonic', return_value=105.):
            self.f.reset()
        self.assertIsNone(self.f._initial_visibility_deadline)
        self.assertFalse(self.f._initial_visibility_complete)

    def test_saved_full_pose_drives_bounded_recovery_after_stop_loss(self):
        self.f._rotation = RotationController()
        saved = VisualPoseFilter().seed(0., .8, 3.96, 102.)
        with patch('calib.fsm_v4.top.time.monotonic', return_value=101.):
            self.f._start_initial_visibility_sweep(101., [])
        self.step(saved, 102.)
        self.assertEqual(self.f.state, 'ACQUIRE_VERIFY')
        self.step(None, 103.9)
        self.f.execu.exec.assert_called_with('STOP')
        self.step(None, 104.)
        self.assertEqual(self.f.state, 'INITIAL_POSE_ROTATE')
        self.f.execu.exec.assert_called_with('ROT_RIGHT')
        hold = self.f._rotation.plan.hold_sec
        self.assertGreater(hold, 0.)
        self.assertLessEqual(hold, cfg.ROT_MAX_COMMAND_HOLD_SEC)
        self.step(None, 104.2)
        self.assertEqual(self.f.state, 'INITIAL_POSE_ROTATE')
        self.assertEqual(self.f._initial_visible_snapshot[0].measurement_mono, 102.)
        self.step(None, 104. + hold + .001)
        self.assertEqual(self.f.state, 'INITIAL_POSE_SETTLE')
        self.f.execu.exec.assert_called_with('STOP')
        self.assertEqual(self.f._rotation_stop_reason, 'fitted_hold_elapsed')
        self.step(None, 104. + hold + cfg.STOP_MIN_SETTLE_SEC + .01)
        self.assertEqual(self.f.state, 'ACQUIRE_VERIFY')
        self.assertIsNone(self.f._initial_visible_snapshot)

    def test_expired_snapshot_is_not_used_and_reset_clears_it(self):
        self.f._initial_visible_snapshot = (self.visible, None)
        self.f._initial_visibility_stop_mono = 101.
        with patch.object(self.f, '_begin_rotation') as begin:
            self.assertFalse(self.f._recover_initial_snapshot(120., []))
            begin.assert_not_called()
        with patch('calib.fsm_v4.top.time.monotonic', return_value=120.):
            self.f.reset()
        self.assertIsNone(self.f._initial_visible_snapshot)


if __name__ == '__main__':
    unittest.main()
