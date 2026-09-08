"""Exercise real FSM step branches with mocked actuators and observations."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4 import config as cfg
from calib.fsm_v4.top import CalibrationFSMV4
from calib.fsm_v4.motion import forward_seconds


class SingleInsertionTests(unittest.TestCase):
    def setUp(self):
        self.f = object.__new__(CalibrationFSMV4)
        self.f._pipeline_started_mono = 100.
        self.pose = SimpleNamespace(yaw_deg=19., rot_x_pallet_m=.5, pallet_z_m=1.8)
        self.f._observe_pose = Mock(return_value=(self.pose, 'ok'))
        self.f._vision_values = Mock(return_value=(0., .2))
        self.f._exec = Mock()
        self.f._set_state = Mock()
        self.f._fail = Mock()
        self.f._begin_translation = Mock()
        self.f._forward_segment_step = Mock()
        self.f._trace_begin = Mock()
        self.f.execu = Mock()
        self.f.status = Mock()
        self.f._insertion_started_mono = 100.

    def step(self):
        with patch('calib.fsm_v4.top.time.monotonic', return_value=101.):
            self.f.step(True, 1.1, 1.4, 19., None)

    def test_acceptance_freezes_z_minus_point_three_without_one_metre_cap(self):
        self.f._accept_insertion(self.pose, [])
        self.assertAlmostEqual(self.f._insert_remaining_m, 1.5)
        self.f.state = 'READY_TO_INSERT'
        self.f._insertion_started_mono = None
        self.pose.pallet_z_m = 1.7  # subsequent frame must not change accepted target
        with patch.object(cfg, 'AUTO_INSERT_ENABLED', True):
            self.step()
        self.assertAlmostEqual(self.f._translation_target_m, 1.5)
        self.assertAlmostEqual(self.f._insert_hold_sec, forward_seconds(1.5))
        self.f.execu.exec.assert_called_once_with('FWD')
        self.f._observe_pose.assert_not_called()

    def test_invalid_acceptance_does_not_start(self):
        for z in (0., .3, cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M + .001, float('nan')):
            self.pose.pallet_z_m = z
            self.f._accept_insertion(self.pose, [])
        self.assertEqual(self.f._fail.call_count, 4)
        self.f._set_state.assert_not_called()

    def test_two_metre_acceptance_means_one_point_seven(self):
        self.pose.pallet_z_m = 2.
        self.f._accept_insertion(self.pose, [])
        self.assertAlmostEqual(self.f._insert_remaining_m, 1.7)

    def test_old_yaw_and_lateral_limits_do_not_abort(self):
        self.f.state = 'INSERT_DRIVE'
        self.f._translation_deadline_mono = 105.
        self.step()
        self.f._fail.assert_not_called()
        self.f._forward_segment_step.assert_not_called()
        self.f._observe_pose.assert_not_called()
        self.f.execu.exec.assert_called_once_with('FWD')

    def test_settled_action_finishes_without_residual_retry(self):
        self.f.state = 'INSERT_SETTLE'
        self.f._settle_observation = Mock(return_value=(self.pose, 0., .2))
        self.f._translation_progress = Mock(return_value=.97)
        self.f._insert_remaining_m = 1.
        self.f._translation_stop_info = {}
        self.f._trace_end = Mock()
        self.f._settle_started_mono = 98.
        self.f._translation_target_m = 1.
        self.f._insert_hold_sec = 3.
        self.step()
        self.f._exec.assert_called_with('STOP')
        self.f._set_state.assert_called_once_with('DONE')
        self.f._begin_translation.assert_not_called()
        self.f._settle_observation.assert_not_called()

    def test_done_remains_stopped(self):
        self.f.state = 'DONE'
        self.step()
        self.f._exec.assert_called_once_with('STOP')
        self.f._begin_translation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
