"""Measured pallet layout; measured 11.5 cm forks, no hardware or inference."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4 import config as cfg
from calib.fsm_v4.insertion_geometry import check_insertion_sweep, pallet_blocks, polygons_touch
from calib.fsm_v4.top import CalibrationFSMV4


class SweepTests(unittest.TestCase):
    def setUp(self):
        fixture = patch.object(cfg, 'FORK_WIDTH_M', .115)
        fixture.start()
        self.addCleanup(fixture.stop)

    def pose(self, x=0., yaw=0., z=1.8):
        return SimpleNamespace(pallet_x_m=x, pallet_z_m=z, yaw_deg=yaw)

    def test_nine_blocks_and_two_empty_rows_have_measured_coordinates(self):
        blocks = list(pallet_blocks())
        self.assertEqual({(r, c) for r, c, _ in blocks},
                         {(r, c) for r in (1, 3, 5) for c in (1, 3, 5)})
        middle_right = next(poly for r, c, poly in blocks if (r, c) == (3, 5))
        for actual, expected in zip(middle_right, ((.35, .45), (.55, .45), (.55, .65), (.35, .65))):
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e)

    def test_centred_straight_insertion_passes(self):
        self.assertTrue(check_insertion_sweep(self.pose()).ok)

    def test_old_gate_passes_but_right_middle_wall_blocks(self):
        pose = self.pose(yaw=-5.)
        self.assertLess(.3 / math.cos(math.radians(pose.yaw_deg)), .355)
        self.assertLess(abs(pose.yaw_deg), cfg.FINAL_YAW_TOL_DEG)
        check = check_insertion_sweep(pose)
        self.assertFalse(check.ok)
        self.assertIn('right fork sweep hits row 3 column 5', check.reason)

    def test_inner_edge_cannot_cross_middle_support(self):
        with patch.object(cfg, 'FORK_WIDTH_M', .21):
            check = check_insertion_sweep(self.pose())
        self.assertFalse(check.ok)
        self.assertEqual(check.reason, 'front pocket clearance')

    def test_exact_margin_contact_rejected(self):
        self.assertTrue(check_insertion_sweep(self.pose(x=.039)).ok)
        self.assertFalse(check_insertion_sweep(self.pose(x=.040)).ok)

    def test_unknown_or_invalid_fork_width_blocks(self):
        for width in (None, 0., -.1, .3, float('nan'), float('inf')):
            with self.subTest(width=width), patch.object(cfg, 'FORK_WIDTH_M', width):
                self.assertFalse(check_insertion_sweep(self.pose()).ok)

    def test_inconsistent_layout_blocks(self):
        with patch.object(cfg, 'INSERT_HOLE_WIDTH_M', .26):
            self.assertFalse(check_insertion_sweep(self.pose()).ok)

    def test_nonfinite_pose_blocks(self):
        for field in ('pallet_x_m', 'pallet_z_m', 'yaw_deg'):
            for value in (float('nan'), float('inf')):
                pose = self.pose()
                setattr(pose, field, value)
                self.assertFalse(check_insertion_sweep(pose).ok)

    def test_longer_travel_includes_rear_row(self):
        # Narrower synthetic forks and small yaw clear the middle row, but not
        # the back row once the commanded tip is extended beyond it.
        with patch.object(cfg, 'FORK_WIDTH_M', .05):
            pose = self.pose(yaw=-3.)
            self.assertTrue(check_insertion_sweep(pose).ok)
            with patch.object(cfg, 'INSERT_CAMERA_Z_REMAINDER_M', -.2):
                check = check_insertion_sweep(pose)
                self.assertFalse(check.ok)
                self.assertIn('row 1', check.reason)

    def test_sat_detects_edge_crossing_without_inside_vertices_and_touch(self):
        a = ((-2., -.1), (2., -.1), (2., .1), (-2., .1))
        b = ((-.1, -2.), (.1, -2.), (.1, 2.), (-.1, 2.))
        self.assertTrue(polygons_touch(a, b))
        self.assertTrue(polygons_touch(a, ((2., -.1), (3., -.1), (3., .1), (2., .1))))
        self.assertFalse(polygons_touch(a, ((2.01, -.1), (3., -.1), (3., .1), (2.01, .1))))

    def test_accepted_grid_has_no_independently_sampled_block_points_in_forks(self):
        # Independent inverse-transform point oracle across all real block areas.
        accepted = 0
        for yaw in (-6., -3., 0., 3., 6.):
            for x in (-.03, 0., .03):
                pose = self.pose(x=x, yaw=yaw)
                if not check_insertion_sweep(pose).ok:
                    continue
                accepted += 1
                c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                for ix in range(3):
                    for iz in range(3):
                        for j in range(11):
                            for k in range(11):
                                u, d = -.55 + ix*.45 + j*.02, iz*.45 + k*.02
                                vx, vz = x+c*u+s*d, 1.8-s*u+c*d
                                inside_fork = -.3 <= vx <= -.185 or .185 <= vx <= .3
                                self.assertFalse(inside_fork and vz <= 1.18+1.5)
        self.assertGreater(accepted, 0)

    def test_direct_approval_cannot_bypass_sweep(self):
        f = object.__new__(CalibrationFSMV4)
        f._fail, f._set_state = Mock(), Mock()
        f._accept_insertion(self.pose(yaw=-5.), [])
        f._fail.assert_called_once()
        self.assertIn('row 3 column 5', f._fail.call_args.args[0])
        f._set_state.assert_not_called()


if __name__ == '__main__':
    unittest.main()
