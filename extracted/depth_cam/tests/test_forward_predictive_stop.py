from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


DEPTH_CAM_DIR = Path(__file__).resolve().parents[1]
if str(DEPTH_CAM_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_CAM_DIR))

from calib.fsm_v4 import config as cfg  # noqa: E402
from calib.fsm_v4.motion import (  # noqa: E402
    bounded_forward_lookahead,
    fitted_forward_speed,
    forward_predictive_stop_ready,
)
from calib.fsm_v4.controllers import (  # noqa: E402
    RotationController,
    adaptive_overshoot_scale,
)
from calib.fsm_v4.planner import visibility_turn_interval  # noqa: E402


class ForwardPredictiveStopTests(unittest.TestCase):
    def test_fitted_speed_has_start_delay_and_cruise_cap(self):
        self.assertEqual(fitted_forward_speed(0.2), 0.0)
        expected_max = cfg.FWD_ACCEL_M_S2 * cfg.FWD_ACCEL_DURATION_SEC
        self.assertAlmostEqual(fitted_forward_speed(20.0), expected_max)

    def test_pose_speed_spike_cannot_create_large_fictitious_advance(self):
        advance, used_speed = bounded_forward_lookahead(
            elapsed_sec=2.781,
            measured_speed_m_s=1.5,
            horizon_sec=0.50,
        )
        self.assertLessEqual(advance, cfg.FWD_PREDICTIVE_MAX_ADVANCE_M)
        self.assertLess(used_speed, 0.4)

        # Reproduces 20260903_190743 step 5: the old code predicted 0.845 m
        # from only 0.083 m real displacement and stopped a 0.8 m command.
        travelled = 0.08264837017429189
        self.assertFalse(forward_predictive_stop_ready(travelled, 0.8))
        self.assertLess(travelled + advance, 0.8)

    def test_predictive_stop_is_available_near_target(self):
        advance, _speed = bounded_forward_lookahead(4.0, 0.25, 0.50)
        travelled = 0.70
        self.assertTrue(forward_predictive_stop_ready(travelled, 0.8))
        self.assertGreaterEqual(travelled + advance, 0.8)


class RotationAdaptiveTimeTests(unittest.TestCase):
    def test_thirty_requested_forty_actual_produces_three_quarter_scale(self):
        self.assertAlmostEqual(adaptive_overshoot_scale(1.0, 30.0, 40.0), 0.75)

    def test_only_post_delay_active_time_is_scaled(self):
        # The runtime feature is disabled in the field config. Enable it only
        # for this unit test so the test exercises the scaling branch it names.
        # Adaptive scaling belongs to the legacy physical response, not the
        # newly selected endpoint-only regression (tested separately).
        with patch.object(cfg, "ROT_ADAPTIVE_OVERSHOOT_ENABLED", True), patch.object(
            cfg, "ROTATION_RESPONSE", cfg.ROTATION_RESPONSE_FALLBACK
        ):
            base = RotationController()
            base.start(15.0, "ROT_RIGHT", 10.0, 10.0, domain="heading")
            adjusted = RotationController()
            adjusted.start(
                15.0, "ROT_RIGHT", 10.0, 10.0,
                domain="heading", adaptive_time_scale=0.75,
            )
        self.assertIsNotNone(base.plan)
        self.assertIsNotNone(adjusted.plan)
        expected = (
            adjusted.response.startup_delay_sec
            + (base.plan.hold_sec - adjusted.response.startup_delay_sec) * 0.75
        )
        self.assertAlmostEqual(adjusted.plan.hold_sec, expected)

    def test_undertravel_does_not_reduce_scale(self):
        self.assertAlmostEqual(adaptive_overshoot_scale(0.75, 30.0, 25.0), 0.75)


class PalletCenterConstraintTests(unittest.TestCase):
    def test_center_bearing_does_not_clip_visibility_turn_interval(self):
        self.assertFalse(cfg.PALLET_CENTER_VISIBILITY_GUARD_ENABLED)
        lower, upper = visibility_turn_interval(-15.0, None)
        self.assertEqual(lower, -cfg.ROT_MAX_WAYPOINT_TURN_DEG)
        self.assertEqual(upper, cfg.ROT_MAX_WAYPOINT_TURN_DEG)

if __name__ == "__main__":
    unittest.main()
