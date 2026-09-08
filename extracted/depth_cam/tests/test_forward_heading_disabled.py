"""Forward guards tested with mocks only; never construct a CAN executor."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4 import config as cfg
from calib.fsm_v4.top import CalibrationFSMV4

MODULE = 'calib.fsm_v4.top'


class ForwardHeadingDisabledTests(unittest.TestCase):
    def setUp(self):
        self.fsm = object.__new__(CalibrationFSMV4)
        self.fsm.state = 'WAYPOINT_DRIVE'
        self.fsm._translation_deadline_mono = 110.
        self.fsm._translation_started_mono = 99.
        self.fsm._translation_target_m = .8
        self.fsm._translation_motion_hits = 0
        self.fsm._translation_progress = Mock(return_value=0.)
        self.fsm._translation_ready = Mock(return_value=False)
        self.fsm._stop_translation = Mock()
        self.fsm._fail = Mock()
        self.fsm.execu = Mock()
        self.fsm.status = Mock()

    def run_step(self, margin=.5):
        with patch(MODULE + '.time.monotonic', return_value=100.), \
             patch(MODULE + '.goal_vector_vehicle', return_value=(.2, 1., 1.02)), \
             patch(MODULE + '.staging_position_errors', return_value=(.2, -1.)), \
             patch(MODULE + '.measurement_age', return_value=0.), \
             patch(MODULE + '.bounded_forward_lookahead', return_value=(0., 0.)):
            self.fsm._forward_segment_step(
                SimpleNamespace(rotation_center_speed_m_s=0.), 0., margin, {},
                'WAYPOINT_DRIVE_SETTLE', [], enforce_visibility=True,
                enforce_staging_limit=True)

    def test_default_ignores_eleven_degree_goal_bearing(self):
        self.assertFalse(cfg.DRIVE_HEADING_GUARD_ENABLED)
        self.run_step()
        self.fsm._stop_translation.assert_not_called()
        self.fsm.execu.exec.assert_called_once_with('FWD')

    def test_image_edge_margin_does_not_interrupt_forward(self):
        with patch.object(cfg, 'IMAGE_EDGE_VISIBILITY_GUARD_ENABLED', True), \
             patch.object(cfg, 'PALLET_CENTER_VISIBILITY_GUARD_ENABLED', True):
            self.run_step(margin=-.1)
        self.fsm._stop_translation.assert_not_called()
        self.fsm.execu.exec.assert_called_once_with('FWD')

    def test_timeout_still_fails(self):
        self.fsm._translation_deadline_mono = 99.
        self.run_step()
        self.fsm._fail.assert_called_once()
        self.fsm.execu.exec.assert_not_called()

    def test_distance_target_still_stops(self):
        self.fsm._translation_ready.return_value = True
        self.fsm._translation_progress.return_value = .81
        self.run_step()
        self.assertEqual(self.fsm._stop_translation.call_args.kwargs['target_m'], .8)
        self.fsm.execu.exec.assert_not_called()


if __name__ == '__main__':
    unittest.main()
