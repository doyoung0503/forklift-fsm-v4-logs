"""Simulator no longer applies its own continuous-opening collision gate."""
import unittest
from protocol_world import World, SPEC


class MeasuredGeometryTests(unittest.TestCase):
    def test_old_wall_crossings_do_not_produce_collision_judgments(self):
        self.assertEqual(SPEC.PALLET_FRONT_VISIBILITY_WIDTH_M,1.1)
        self.assertEqual(SPEC.INSERT_OPENING_SPAN_M,.71)
        self.assertEqual(SPEC.FORK_OUTER_SPAN_M,.60)
        for offset in (0.,.055,-.055,.056,-.056):
            world=World(dict(x=offset,z=.8,yaw=0,opening='continuous'))
            world.advance(.02, [])
            self.assertIsNone(world.plant.minimum_clearance)
            self.assertIsNone(world.plant.collision)


if __name__=='__main__':
    unittest.main()
