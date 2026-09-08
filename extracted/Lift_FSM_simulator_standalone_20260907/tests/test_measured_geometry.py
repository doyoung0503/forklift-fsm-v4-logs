"""Metric acceptance limits independent of the FSM's completion decision."""
import unittest
from protocol_world import World, SPEC


class MeasuredGeometryTests(unittest.TestCase):
    def test_centered_and_grazing_forks(self):
        self.assertEqual(SPEC.PALLET_FRONT_VISIBILITY_WIDTH_M,1.1)
        self.assertEqual(SPEC.INSERT_OPENING_SPAN_M,.71)
        self.assertEqual(SPEC.FORK_OUTER_SPAN_M,.60)
        for offset in (0.,.055,-.055,.056,-.056):
            world=World(dict(x=offset,z=.8,yaw=0,opening='continuous'))
            world.plant.check_collision(0)
            self.assertAlmostEqual(world.plant.minimum_clearance,.055-abs(offset))
            self.assertEqual(world.plant.collision is not None,abs(offset)>.055)

    def test_tip_enters_at_measured_camera_distance(self):
        self.assertAlmostEqual(SPEC.CAMERA_TO_FORK_TIP_Z_M-SPEC.CAMERA_TO_ROT_CENTER_Z_M,1.86)
        for distance,engaged in ((1.181,False),(1.179,True)):
            world=World(dict(x=0,z=distance,yaw=0))
            world.plant.check_collision(0)
            self.assertEqual(world.plant.minimum_clearance is not None,engaged)


if __name__=='__main__':
    unittest.main()
