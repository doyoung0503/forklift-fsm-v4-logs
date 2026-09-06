"""Independent geometry and original-planner checks for the analysis adapter."""
import math
import unittest
from unittest.mock import patch

import visualize_fsm_v4_feasibility as a


class FeasibilityAnalysisTests(unittest.TestCase):
    def test_map_coordinates_roundtrip_and_camera_distance(self):
        for radius in (2., 3.43, 6., 10.):
            for beta in (-45., -20., 0., 20., 45.):
                pose = a.initial_pose(radius, beta)
                self.assertAlmostEqual(pose.rot_x_pallet_m, radius*math.sin(math.radians(beta)))
                self.assertAlmostEqual(pose.rot_z_pallet_m-a.HALF, -radius*math.cos(math.radians(beta)))
        self.assertAlmostEqual(a.initial_pose(3.43, 0.).pallet_z_m, 2.2)

    def test_vectorized_visibility_matches_original_sampled_model(self):
        self.assertLess(a.validate_fast_margin(), 1e-10)

    def test_sufficient_impossibility_bound_does_not_reject_a_valid_ray_pose(self):
        checked = 0
        for radius in (1., 2., 2.3, 2.4, 2.8, 3., 3.3):
            for beta in (-45., -30., -20., -10., 0., 10., 20., 30., 45.):
                pose = a.initial_pose(radius, beta)
                if not a.rotation_only_impossible(pose):
                    continue
                checked += 1
                for alpha in range(-200, 201):
                    predicted = a.p._pose_after_action(pose, beta-alpha/10., 0.)
                    self.assertFalse(a.p.fork_opening_alignment(predicted, enforce_entry_distance=False)[0],
                                     (radius, beta, alpha/10.))
        self.assertGreater(checked, 20)

    def test_original_planner_and_fast_rollouts_agree(self):
        for radius, beta in ((2.5,0.),(3.,15.),(3.5,10.),(4.,15.),(5.,20.),
                             (6.,30.),(4.,45.),(8.,0.),(9.,20.),(10.,0.)):
            with self.subTest(radius=radius,beta=beta):
                original = a.evaluate(radius,beta)
                with patch.object(a.p,'action_visibility_margin_deg',a.fast_margin):
                    fast = a.evaluate(radius,beta)
                self.assertEqual(original['category'],fast['category'])
                self.assertEqual(original['reason'],fast['reason'])
                self.assertEqual(original.get('states'),fast.get('states'))
                self.assertAlmostEqual(original.get('turn_deg',0),fast.get('turn_deg',0),places=8)

    def test_representative_distance_failures_and_successes(self):
        self.assertEqual(a.evaluate(2.,0.)['category'],0)
        self.assertTrue(a.evaluate(3.,0.)['success'])
        self.assertTrue(a.evaluate(4.,0.)['success'])
        self.assertIn('standoff translation timeout',a.evaluate(10.,0.)['reason'])


if __name__ == '__main__':
    unittest.main()
