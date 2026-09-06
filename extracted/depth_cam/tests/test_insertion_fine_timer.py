"""No hardware; scoped minimum, adaptive slope and inclusive fine timeout."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4 import config as cfg
from calib.fsm_v4.controllers import RotationController
from calib.fsm_v4.adaptive_slope import SessionSlope
from calib.fsm_v4.top import CalibrationFSMV4


class FineInsertionTests(unittest.TestCase):
    def test_half_degree_only_with_explicit_insertion_scope_and_adaptive(self):
        with patch.object(cfg, 'ROT_ADAPTIVE_SLOPE_ENABLED', True):
            ordinary, fine = RotationController(), RotationController()
            ordinary.start(.5, 'ROT_RIGHT', 1., 1., domain='heading')
            fine.start(.5, 'ROT_RIGHT', 1., 1., domain='heading',
                       insertion_fine=True, slope_multiplier=1.5)
        self.assertEqual(ordinary.plan.hold_sec, 0.)
        self.assertGreater(fine.plan.hold_sec, cfg.ROTATION_RESPONSE.startup_delay_sec)
        self.assertAlmostEqual(fine.plan.hold_sec, cfg.ROTATION_RESPONSE.startup_delay_sec
                               + .5 / (cfg.ROTATION_RESPONSE.slope_deg_s * 1.5))
        self.assertEqual(cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG, 2.5)

    def test_micro_observation_updates_with_small_but_nonzero_active_time(self):
        learner = SessionSlope(alpha=.5)
        r = learner.observe('ROT_LEFT', 10., 1., 1.04, .6,
                            min_active_sec=cfg.INSERT_FINE_MIN_ACTIVE_SEC)
        self.assertTrue(r['slope_update_accepted'])
        self.assertAlmostEqual(r['slope_next_deg_s'], 12.5)
        self.assertFalse(learner.observe('ROT_LEFT', 10., 1., 1.001, .6,
                         min_active_sec=cfg.INSERT_FINE_MIN_ACTIVE_SEC)['slope_update_accepted'])

    def test_timer_counts_settle_wait_and_stops_at_120_seconds(self):
        f = object.__new__(CalibrationFSMV4)
        f._insertion_fine_started_mono = 100.
        f._exec = Mock()
        f._fail = Mock()
        for state in ('FINAL_ROTATE', 'FINAL_SETTLE', 'FINAL_POSE_LOCK'):
            f.state = state
            self.assertFalse(f._insertion_fine_timed_out(219.999, []))
            self.assertTrue(f._insertion_fine_timed_out(220., []))
        f._exec.assert_called_with('STOP')
        self.assertEqual(f._fail.call_count, 3)

    def test_ordinary_run_has_no_fine_deadline(self):
        f = object.__new__(CalibrationFSMV4)
        f._insertion_fine_started_mono = None
        f.state = 'WAYPOINT_TURN'
        self.assertFalse(f._insertion_fine_timed_out(1000., []))


if __name__ == '__main__':
    unittest.main()
