"""Synthetic session data only. No CAN, camera or external log input."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4 import config as cfg
from calib.fsm_v4.adaptive_slope import SessionSlope
from calib.fsm_v4.controllers import RotationController
from calib.fsm_v4.top import CalibrationFSMV4


class SessionSlopeTests(unittest.TestCase):
    def test_active_time_not_total_including_wait(self):
        learner = SessionSlope(alpha=1.)
        result = learner.observe('ROT_RIGHT', 10., 1., 1.5, 8.)
        self.assertAlmostEqual(result['slope_active_sec'], .5)
        self.assertAlmostEqual(result['slope_next_deg_s'], 16.)
        self.assertEqual(learner.multiplier('ROT_LEFT'), 1.)
        self.assertEqual(SessionSlope().multiplier('ROT_RIGHT'), 1.)

    def test_measured_command_overrun_is_in_denominator(self):
        a, b = SessionSlope(alpha=1.), SessionSlope(alpha=1.)
        a.observe('ROT_RIGHT', 10., 1., 1.5, 8.)
        b.observe('ROT_RIGHT', 10., 1., 1.8, 8.)
        self.assertAlmostEqual(a.multiplier('ROT_RIGHT'), 1.6)
        self.assertAlmostEqual(b.multiplier('ROT_RIGHT'), 1.)

    def test_smoothing_bounds_and_invalid_samples(self):
        learner = SessionSlope()
        learner.observe('ROT_RIGHT', 10., 1., 1.5, 100.)
        self.assertEqual(learner.multiplier('ROT_RIGHT'), 1.25)
        for _ in range(100):
            learner.observe('ROT_RIGHT', 10., 1., 1.5, 100.)
        self.assertLessEqual(learner.multiplier('ROT_RIGHT'), 2.)
        old = learner.multiplier('ROT_RIGHT')
        for hold, angle in ((1.01, 4.), (float('nan'), 4.), (1.5, -4.)):
            self.assertFalse(learner.observe('ROT_RIGHT', 10., 1., hold, angle)['slope_update_accepted'])
        self.assertEqual(old, learner.multiplier('ROT_RIGHT'))

    def test_flag_off_unchanged_and_on_preserves_delay_and_base(self):
        base = cfg.ROTATION_RESPONSE
        off = RotationController()
        with patch.object(cfg, 'ROT_ADAPTIVE_SLOPE_ENABLED', False):
            off.start(4., 'ROT_RIGHT', 100., 100., domain='heading', slope_multiplier=1.5)
        self.assertEqual(off.response, base)
        with patch.object(cfg, 'ROT_ADAPTIVE_SLOPE_ENABLED', True):
            on = RotationController()
            on.start(4., 'ROT_RIGHT', 100., 100., domain='heading', slope_multiplier=1.5)
            self.assertAlmostEqual(on.response.slope_deg_s, base.slope_deg_s*1.5)
            self.assertEqual(on.response.startup_delay_sec, base.startup_delay_sec)
            self.assertAlmostEqual(on.plan.hold_sec, base.startup_delay_sec+4/(base.slope_deg_s*1.5))
            self.assertIs(cfg.ROTATION_RESPONSE, base)
            bearing = RotationController()
            bearing.start(4., 'ROT_RIGHT', 100., 100., domain='bearing', range_m=3., slope_multiplier=1.5)
            self.assertAlmostEqual(bearing.response.slope_deg_s,
                                   base.in_bearing_domain(3., cfg.ROT_BEARING_CENTRE_OFFSET_M).slope_deg_s)

    def make_fsm(self):
        fsm = object.__new__(CalibrationFSMV4)
        fsm._rotation = SimpleNamespace(
            response=cfg.ROTATION_RESPONSE, plan=SimpleNamespace(target_deg=4., hold_sec=1.6),
            start_error_deg=4., expected_delta_sign=-1., command='ROT_RIGHT')
        fsm._rotation_actual_hold_sec = 1.7
        fsm._rotation_mode = 'waypoint'
        fsm._rotation_stop_reason = 'fitted_hold_elapsed'
        fsm._session_slope = SessionSlope()
        return fsm

    def test_fsm_consumes_settled_sample_once_and_reports_overrun(self):
        fsm = self.make_fsm()
        with patch.object(cfg, 'ROT_ADAPTIVE_SLOPE_ENABLED', True):
            fsm._update_rotation_adaptation(-2.)
            self.assertTrue(fsm._rotation_adaptive_result['slope_update_accepted'])
            self.assertAlmostEqual(fsm._rotation_adaptive_result['slope_hold_overrun_sec'], .1)
            self.assertEqual(fsm._rotation_adaptive_result['slope_settled_angle_deg'], 6.)
            previous = fsm._session_slope.multiplier('ROT_RIGHT')
            fsm._update_rotation_adaptation(-2.)
            self.assertEqual(fsm._session_slope.multiplier('ROT_RIGHT'), previous)

    def test_disabled_or_abnormal_stop_does_not_learn(self):
        fsm = self.make_fsm()
        with patch.object(cfg, 'ROT_ADAPTIVE_SLOPE_ENABLED', False):
            fsm._update_rotation_adaptation(-2.)
        self.assertEqual(fsm._session_slope.multiplier('ROT_RIGHT'), 1.)
        for reason in ('timeout', 'direction_change_budget', 'visibility'):
            fsm = self.make_fsm()
            fsm._rotation_stop_reason = reason
            with patch.object(cfg, 'ROT_ADAPTIVE_SLOPE_ENABLED', True):
                fsm._update_rotation_adaptation(-2.)
            self.assertEqual(fsm._session_slope.multiplier('ROT_RIGHT'), 1.)


if __name__ == '__main__':
    unittest.main()
