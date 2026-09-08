"""Horizontal fork-ray geometry only; no hardware."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4 import config as cfg
from calib.fsm_v4.planner import fork_opening_alignment, staging_position_reached


class ForkOpeningTests(unittest.TestCase):
    def setUp(self):
        width = patch.object(cfg, 'FORK_WIDTH_M', .10)
        width.start()
        self.addCleanup(width.stop)

    def pose(self, x=0., z=1.5, yaw=0.):
        return SimpleNamespace(pallet_x_m=x, pallet_z_m=z, yaw_deg=yaw)

    def test_centred_hits_and_40mm_margin_boundary(self):
        ok, hits = fork_opening_alignment(self.pose())
        self.assertTrue(ok)
        self.assertEqual(hits, (-.3, .3))
        self.assertTrue(fork_opening_alignment(self.pose(x=.039))[0])
        self.assertFalse(fork_opening_alignment(self.pose(x=.040))[0])

    def test_yaw_widens_intersections(self):
        ok, hits = fork_opening_alignment(self.pose(yaw=20.))
        self.assertFalse(ok)
        self.assertAlmostEqual(hits[1], .3192533317)
        self.assertFalse(fork_opening_alignment(self.pose(x=.04, yaw=20.))[0])

    def test_previous_front_only_acceptance_is_now_rejected(self):
        pose = self.pose(x=-.0044616656, z=1.8001815, yaw=-12.0525857)
        self.assertFalse(fork_opening_alignment(pose)[0])
        self.assertLess(abs(pose.yaw_deg), cfg.FINAL_YAW_TOL_DEG)

    def test_camera_z_activation_boundary_is_inclusive(self):
        limit = cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M
        self.assertTrue(fork_opening_alignment(self.pose(z=limit))[0])
        self.assertTrue(fork_opening_alignment(self.pose(z=limit - .0001))[0])
        for z in (limit + .0001, limit + .5, 0., -1., float('nan'), float('inf')):
            self.assertEqual(fork_opening_alignment(self.pose(z=z)), (False, ()))

    def test_face_behind_tips_parallel_or_invalid_rejected(self):
        for pose in (self.pose(z=1.), self.pose(yaw=90.), self.pose(yaw=float('nan'))):
            self.assertFalse(fork_opening_alignment(pose)[0])

    def test_fork_centre_offset_is_used(self):
        with patch.object(cfg, 'CAMERA_TO_FORK_TIP_X_M', .1):
            self.assertFalse(fork_opening_alignment(self.pose())[0])
            self.assertTrue(fork_opening_alignment(self.pose(x=.1))[0])

    def test_large_yaw_is_rejected_by_both_geometry_and_yaw_gate(self):
        pose = self.pose(yaw=21.)
        self.assertFalse(fork_opening_alignment(pose)[0])
        self.assertGreater(abs(pose.yaw_deg), cfg.FINAL_YAW_TOL_DEG)


if __name__ == '__main__':
    unittest.main()
