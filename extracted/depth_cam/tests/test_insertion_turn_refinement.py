"""Real saved failure poses; refine insertion turns without relaxing guards."""
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4 import config as cfg, planner as p
from calib.fsm_v4.pose import VisualPoseFilter


class RefinementTests(unittest.TestCase):
    def pose(self, side=1.):
        return VisualPoseFilter().seed(side*8.895732011011319,
            side*.13083585537427211, 1.7838915466950969, 100.)

    def test_missed_interval_is_recovered_on_both_sides(self):
        with patch.object(cfg, 'FORK_WIDTH_M', .115):
            for side in (1., -1.):
                pose=self.pose(side)
                for angle in (3.5, 4.):
                    self.assertFalse(p.fork_opening_alignment(
                        p._pose_after_action(pose, side*angle, 0.),
                        enforce_entry_distance=False)[0])
                turn=p.insertion_alignment_turn(pose)
                self.assertGreater(side*turn, 3.57)
                self.assertLess(side*turn, 3.91)
                self.assertTrue(p.fork_opening_alignment(
                    p._pose_after_action(pose, turn, 0.), enforce_entry_distance=False)[0])
                self.assertTrue(p.action_keeps_front_visible(pose, turn, 0., None))

    def test_refinement_still_requires_rotation_visibility(self):
        with patch.object(cfg, 'FORK_WIDTH_M', .115), \
             patch.object(p, 'action_keeps_front_visible', side_effect=lambda pose,turn,forward,meta: turn==0.):
            self.assertIsNone(p.insertion_alignment_turn(self.pose()))

    def test_existing_valid_candidate_is_preserved(self):
        pose=VisualPoseFilter().seed(-5.,0.,1.8,100.)
        with patch.object(cfg, 'FORK_WIDTH_M', .10):
            self.assertEqual(p.insertion_alignment_turn(pose),-.5)

    def test_search_is_bounded_when_all_geometry_fails(self):
        with patch.object(p, 'fork_opening_alignment', return_value=(False,())) as checks:
            self.assertIsNone(p.insertion_alignment_turn(self.pose()))
        self.assertLess(checks.call_count, 1000)

    def test_narrow_interior_is_searched_even_when_endpoints_and_midpoint_fail(self):
        pose=VisualPoseFilter().seed(0.,0.,1.8,100.)
        def geometry(predicted, **kwargs):
            return (3.59 < -predicted.yaw_deg < 3.61, (0.,0.))
        with patch.object(p, 'fork_opening_alignment', side_effect=geometry), \
             patch.object(p, 'action_keeps_front_visible', return_value=True):
            turn=p.insertion_alignment_turn(pose)
        self.assertGreater(turn,3.59)
        self.assertLess(turn,3.61)

    def test_resolution_is_validated_and_recorded(self):
        self.assertEqual(cfg.metadata()['v4_insertion_turn_refine_resolution_deg'], .05)
        for value in (0., -.01, math.nan, math.inf, .5):
            with patch.object(cfg, 'INSERT_TURN_REFINE_RESOLUTION_DEG', value):
                with self.assertRaisesRegex(ValueError, 'refinement resolution'):
                    cfg.validate()


if __name__=='__main__':
    unittest.main()
