import unittest
import numpy as np
from protocol_world import World,PROFILE
from simulation_camera import world_geometry


class SceneGeometryTests(unittest.TestCase):
    def test_camera_offset_and_stationary_pallet_after_motion(self):
        world=World(dict(placement_mode='world',forklift_x=1,forklift_z=-.68,
                         forklift_heading=30,pallet_x=.3,pallet_z=2.2,pallet_heading=165))
        config=PROFILE['parameters']
        initial=world_geometry(world.plant.truth(),config)
        np.testing.assert_allclose(initial['pallet'],[.3,2.2],atol=1e-10)
        self.assertAlmostEqual(np.linalg.norm(initial['camera']-initial['pivot']),.68)
        # Re-express the same fixed pallet after changing the camera pose.
        camera=np.array([.7,.9]);heading=-20
        a=np.radians(heading);R=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
        local=R.T@(np.array([.3,2.2])-camera)
        pivot=camera+R@np.array([0,-.68])
        truth=dict(x=local[0],z=local[1],yaw=5,heading=heading,world_x=pivot[0],world_z=pivot[1])
        moved=world_geometry(truth,config)
        np.testing.assert_allclose(moved['pallet'],initial['pallet'],atol=1e-10)
        np.testing.assert_allclose(moved['camera'],camera,atol=1e-10)


if __name__=='__main__':unittest.main()
