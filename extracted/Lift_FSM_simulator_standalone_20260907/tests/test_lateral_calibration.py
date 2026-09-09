"""Check experiment coordinates and guard against misleading binary boundaries."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate_lateral import Experiment, placement
from protocol_world import Plant, SPEC, scenario_options


class FakeExperiment(Experiment):
    def __init__(self, predicate):
        self.rows = []
        self.predicate = predicate

    def trial(self, distance, lateral, facing=True):
        row = dict(distance_m=distance, lateral_m=lateral, facing=facing,
                   success=self.predicate(abs(lateral)))
        self.rows.append(row)
        return row


class CalibrationTests(unittest.TestCase):
    def test_world_placement_preserves_pallet_coordinates(self):
        for distance in (3., 4., 5.):
            for lateral in (-1., 0., 1.):
                plant = Plant(scenario_options(placement(distance, lateral)))
                a = math.radians(plant.yaw)
                dx = SPEC.CAMERA_TO_ROT_CENTER_X_M - plant.x
                dz = SPEC.CAMERA_TO_ROT_CENTER_Z_M - plant.z
                self.assertAlmostEqual(math.cos(a)*dx-math.sin(a)*dz, lateral)
                self.assertAlmostEqual(math.sin(a)*dx+math.cos(a)*dz, -distance)
                self.assertAlmostEqual(plant.x, 0.)

    def test_monotonic_bracket_contains_boundary(self):
        result = FakeExperiment(lambda x: x <= .277).boundary(3., 1)
        self.assertEqual(result['status'], 'sampled_monotonic')
        self.assertLessEqual(result['success_m'], .277)
        self.assertGreater(result['failure_m'], .277)
        self.assertLessEqual(result['resolution_m'], .01)

    def test_interior_failure_invalidates_monotonicity(self):
        result = FakeExperiment(lambda x: x <= .277 and not .06 < x < .08).boundary(3., 1)
        self.assertEqual(result['status'], 'nonmonotonic')
        self.assertLess(result['first_sampled_failure_m'], result['success_m'])
        self.assertLessEqual(result['conservative_success_m'], .06)
        self.assertGreater(result['conservative_failure_m'], .06)

    def test_failed_origin_has_no_boundary(self):
        result = FakeExperiment(lambda x: False).boundary(3., 1)
        self.assertEqual(result['status'], 'zero_lateral_failed')


if __name__ == '__main__':
    unittest.main()
