import math
import random
import statistics
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol_world import Plant, SPEC, scenario_options
from process_runtime import Session
from batched_session import LocalController

class ZeroMeanErrors(unittest.TestCase):
    def test_fixed_means_and_central_90_percent(self):
        o=scenario_options(dict(drive_scale=.8,rotation_scale=1.2,imu_scale=95/90,imu_bias_deg_s=1))
        self.assertEqual([o[k] for k in ['drive_scale','rotation_scale','imu_scale']],[1.,1.,1.])
        self.assertEqual(o['imu_bias_deg_s'],0)
        self.assertAlmostEqual(math.erf(.1/(o['drive_scale_sd']*math.sqrt(2))),.9)
        plant=Plant(o);drive=[];turn=[]
        for _ in range(20000):
            plant._set_axes(SPEC.ROTATE_REFERENCE_DEFLECTION,SPEC.DRIVE_REFERENCE_DEFLECTION)
            drive.append(plant.drive_gain-1);turn.append(plant.rotation_gain-1)
            plant._set_axes(0,0);plant.coasts.clear()
        for samples in [drive,turn]:
            self.assertLess(abs(statistics.mean(samples)),.002)
            self.assertAlmostEqual(sum(abs(x)<=.1 for x in samples)/len(samples),.9,delta=.015)

    def test_imu_bias_is_seeded_once_and_stationary_readings_are_stable(self):
        s=Session(dict(seed=123,imu_angle_sd_deg=5),controller_factory=LocalController,log_root=None)
        rng=random.Random(123 ^ 0x494d55)
        try:
            self.assertAlmostEqual(s.imu_gain,1+rng.gauss(0,5/90))
            self.assertEqual(s.provenance['imu_error_model']['sampled_gain'],s.imu_gain)
            for _ in range(20):
                s.step();imu=s.inputs[-1]['imu']
                self.assertEqual(imu['angle_error_deg'],0)
                self.assertEqual(imu['rate_deg_s'],0)
                self.assertEqual(imu['yaw_deg'],s.imu_initial_yaw)
        finally:s.close()

    def test_smooth_endpoint_bias_wrap_stop_and_reverse(self):
        s=Session(dict(imu_angle_sd_deg=5),controller_factory=LocalController,log_root=None)
        try:
            s.imu_gain=95/90
            s.imu_initial_yaw=s.imu_yaw=170.
            s.imu_previous=(0.,170.)
            previous_yaw=170.
            for step in range(1,91):
                s.now=step*.1;s.plant.heading=(170.+step+180.)%360.-180.
                imu=s._observe_imu()
                self.assertAlmostEqual(imu['yaw_deg']-previous_yaw,95/90)
                self.assertAlmostEqual(imu['rate_deg_s'],10*95/90)
                previous_yaw=imu['yaw_deg']
            self.assertAlmostEqual(imu['yaw_deg'],265.)
            self.assertAlmostEqual(imu['angle_error_deg'],5.)
            for _ in range(20):
                s.now+=.1;imu=s._observe_imu()
                self.assertAlmostEqual(imu['yaw_deg'],265.)
                self.assertEqual(imu['rate_deg_s'],0.)
            s.now+=9.;s.plant.heading=170.
            imu=s._observe_imu()
            self.assertAlmostEqual(imu['yaw_deg'],170.)
            self.assertAlmostEqual(imu['rate_deg_s'],-10*95/90)
        finally:
            s.now=s.world.now
            s.close()

    def test_ideal_imu_and_sample_frequency_invariance(self):
        def endpoint(hz,sigma):
            s=Session(dict(seed=123,imu_angle_sd_deg=sigma),controller_factory=LocalController,log_root=None)
            try:
                start=s.plant.heading
                for step in range(1,hz+1):
                    s.now=step/hz;s.plant.heading=(start+90*step/hz)%360
                    imu=s._observe_imu()
                return imu
            finally:
                s.now=s.world.now
                s.close()
        a,b=endpoint(30,5),endpoint(120,5)
        for key in ('yaw_deg','rate_deg_s','angle_error_deg'):
            self.assertAlmostEqual(a[key],b[key])
        ideal=endpoint(30,0)
        self.assertEqual(ideal['angle_error_deg'],0)
        self.assertAlmostEqual(ideal['rate_deg_s'],90)
