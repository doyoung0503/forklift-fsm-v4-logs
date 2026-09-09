"""Distance-adaptive initial correction routing, no hardware."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4 import config as cfg
from calib.fsm_v4.coarse import CoarseAlignmentMixin, coarse_lateral_limit_m


class AdaptiveCoarseTests(unittest.TestCase):
    def setUp(self):
        self.f = CoarseAlignmentMixin()
        self.f._coarse_done = False

    def pose(self, distance, lateral, yaw=0.):
        return SimpleNamespace(rot_z_pallet_m=-distance,
                               rot_x_pallet_m=lateral, yaw_deg=yaw)

    def test_formula_uses_pallet_normal_pivot_distance(self):
        for distance in (3., 3.5, 4., 4.5, 5.):
            self.assertAlmostEqual(coarse_lateral_limit_m(distance),
                cfg.COARSE_LATERAL_BASE_M + cfg.COARSE_LATERAL_GAIN *
                (distance - cfg.STAGING_DISTANCE_M))

    def test_both_signs_strict_threshold_boundary(self):
        for distance in (3., 4., 5.):
            limit = coarse_lateral_limit_m(distance)
            for side in (-1., 1.):
                self.assertFalse(self.f._coarse_required(self.pose(distance, side*limit)))
                self.assertFalse(self.f._coarse_required(self.pose(distance, side*(limit-.001))))
                self.assertTrue(self.f._coarse_required(self.pose(distance, side*(limit+.001))))

    def test_same_lateral_can_require_correction_only_when_close(self):
        lateral = (coarse_lateral_limit_m(3.) + coarse_lateral_limit_m(5.))/2
        self.assertTrue(self.f._coarse_required(self.pose(3., lateral)))
        self.assertFalse(self.f._coarse_required(self.pose(5., lateral)))

    def test_yaw_is_not_the_trigger(self):
        self.assertTrue(self.f._coarse_required(self.pose(3., .8, yaw=0.)))
        self.assertFalse(self.f._coarse_required(self.pose(3., .01, yaw=40.)))

    def test_uncalibrated_far_distance_cannot_expand_limit(self):
        self.assertEqual(coarse_lateral_limit_m(50.), coarse_lateral_limit_m(5.))
        self.assertEqual(coarse_lateral_limit_m(1.), cfg.COARSE_LATERAL_BASE_M)
        self.assertLess(coarse_lateral_limit_m(2.5), coarse_lateral_limit_m(3.))

    def test_disabled_and_once_only_still_apply(self):
        pose = self.pose(3., 1.)
        with patch.object(cfg, 'COARSE_IMU_ENABLED', False):
            self.assertFalse(self.f._coarse_required(pose))
        self.f._coarse_done = True
        self.assertFalse(self.f._coarse_required(pose))

    def test_invalid_pose_does_not_silently_skip_initial_correction(self):
        self.f._exec = Mock()
        self.f._fail = Mock()
        for field in ('rot_x_pallet_m', 'rot_z_pallet_m', 'yaw_deg'):
            pose = self.pose(3., .5)
            setattr(pose, field, math.nan)
            self.assertTrue(self.f._maybe_begin_coarse(pose, 100., []))
        self.assertEqual(self.f._fail.call_count, 3)
        self.f._exec.assert_called_with('STOP')

    def test_threshold_settings_are_logged_and_validated(self):
        self.assertEqual(cfg.metadata()['v4_coarse_lateral_gain'], cfg.COARSE_LATERAL_GAIN)
        for bad_gain in (-.1, math.nan, math.inf):
            with patch.object(cfg, 'COARSE_LATERAL_GAIN', bad_gain):
                with self.assertRaisesRegex(ValueError, 'coarse lateral threshold'):
                    cfg.validate()


if __name__ == '__main__':
    unittest.main()
