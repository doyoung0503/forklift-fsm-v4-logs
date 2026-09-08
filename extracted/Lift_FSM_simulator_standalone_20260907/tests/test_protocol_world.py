"""CAN-only control, absolute-error sampling, timing and public HTTP contract."""
import ast
import contextlib
import io
import json
import math
import statistics
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_result_model import GaussianResultModel
from protocol_world import World,can_frame,model_step,scenario_options,wrap
from serve_world import WorldHandler
from protocol_client import run


def drive(world,seconds,steer=127,forward=127,heartbeat=True,mode=0x42):
    for _ in range(round(seconds/.02)):
        now=world.now
        frames=[can_frame(0x2e3,[mode,0,0,10,0,64,105,147],now)]
        if heartbeat:
            frames.append(can_frame(0x764,[0],now))
        frames.append(can_frame(0x1e3,[127,steer,forward,127,127,127,127,127],now))
        world.advance(.02,frames)


class ProtocolTests(unittest.TestCase):
    def test_world_placement_converts_vehicle_and_pallet_headings(self):
        w=World(dict(placement_mode='world',forklift_x=1,forklift_z=2,forklift_heading=90,
                     pallet_x=3.68,pallet_z=2,pallet_heading=-90))
        self.assertAlmostEqual(w.plant.x,0)
        self.assertAlmostEqual(w.plant.z,2)
        self.assertAlmostEqual(w.plant.yaw,0)
        self.assertEqual((w.plant.world_x,w.plant.world_z,w.plant.heading),(1,2,90))
        drive(w,3,forward=67)
        self.assertGreater(w.plant.world_x,1)
        self.assertAlmostEqual(w.plant.world_z,2)

    def test_can_motion_keeps_world_pallet_fixed(self):
        from protocol_world import SPEC
        w=World(dict(placement_mode='world',forklift_x=-1,forklift_z=-2,forklift_heading=20,
                     pallet_x=2,pallet_z=3,pallet_heading=165))
        for steer,forward in ((97,127),(127,67),(157,187)):
            drive(w,2,steer=steer,forward=forward)
            p=w.plant;a=math.radians(p.heading);c,s=math.cos(a),math.sin(a)
            dx,dz=p.x-SPEC.CAMERA_TO_ROT_CENTER_X_M,p.z-SPEC.CAMERA_TO_ROT_CENTER_Z_M
            self.assertAlmostEqual(p.world_x+c*dx+s*dz,2)
            self.assertAlmostEqual(p.world_z-s*dx+c*dz,3)
            self.assertAlmostEqual(wrap(p.yaw+p.heading+180),165)

    def test_raw_can_directions_and_analog_strength(self):
        outcomes=[]
        for steer,forward in ((127,67),(127,187),(157,127),(97,127),(127,97)):
            world=World()
            drive(world,3,steer,forward)
            outcomes.append(world.plant.truth())
        self.assertGreater(outcomes[0]['world_z'],0)
        self.assertLess(outcomes[1]['world_z'],0)
        self.assertLess(outcomes[2]['heading'],0)
        self.assertGreater(outcomes[3]['heading'],0)
        self.assertAlmostEqual(outcomes[4]['world_z'],outcomes[0]['world_z']/2)

    def test_mode_heartbeat_watchdog_and_unknown_frames(self):
        for heartbeat,mode in ((False,0x42),(True,0)):
            w=World();drive(w,2,forward=67,heartbeat=heartbeat,mode=mode)
            self.assertEqual(w.plant.world_z,0)
        w=World();drive(w,2,forward=67)
        before=w.plant.world_z
        w.advance(.5,[])
        self.assertGreater(w.plant.world_z,before)
        stopped=w.plant.world_z
        w.advance(.5,[can_frame(0x123,[67]*8,w.now),can_frame(0x1e3,[67]*8,w.now,extended=True)])
        self.assertEqual(w.plant.world_z,stopped)
        self.assertNotEqual(w.plant.bus_status,'ready')

    def test_absolute_bias_not_accumulated_and_yaw_wrap(self):
        w=World(dict(perception='oracle',x_bias=.12,y_bias=-.03,z_bias=.2,yaw_bias=10,yaw=175,face_selection='fixed',latency=0))
        for _ in range(50):
            packet=w.observe();diag=w.last_diagnostic
            for axis in ('x','y','z'):
                self.assertAlmostEqual(diag[axis]-diag['ground_truth_pose'][axis],w.options[axis+'_bias'])
            self.assertAlmostEqual(wrap(diag['yaw']-diag['ground_truth_pose']['yaw']),10)
            self.assertEqual(packet['step']['offset_smooth'],[diag[k] for k in ('x','y','z')])
            w.advance(.04,[])

    def test_empirical_gaussian_mean_std_and_miss_probability(self):
        options=scenario_options(dict(x_bias=.15,x_noise=.03,y_bias=-.1,y_noise=.04,
                                      z_bias=.2,z_noise=.05,yaw_bias=2,yaw_noise=1,dropout=.2))
        sampler=GaussianResultModel(options)
        samples=[sampler.sample() for _ in range(20000)]
        for axis in ('x','y','z','yaw'):
            values=[s['error'][axis] for s in samples]
            self.assertAlmostEqual(statistics.mean(values),options[axis+'_bias'],delta=options[axis+'_noise']*.035)
            self.assertAlmostEqual(statistics.stdev(values),options[axis+'_noise'],delta=options[axis+'_noise']*.035)
        self.assertAlmostEqual(statistics.mean(s['missed'] for s in samples),.2,delta=.012)

    def test_fresh_results_only_at_model_cadence(self):
        w=World(dict(model_hz=10,x_noise=.1))
        initial=json.dumps(w.observe(),sort_keys=True)
        w.advance(.09,[])
        self.assertEqual(json.dumps(w.observe(),sort_keys=True),initial)
        w.advance(.01,[])
        self.assertEqual(w.observe()['sequence'],1)
        self.assertAlmostEqual(w.observe()['result_fps'],10)
        self.assertNotEqual(json.dumps(w.observe(),sort_keys=True),initial)
        w.advance(.9,[])
        self.assertEqual(w.model_count,11)

    def test_missed_result_has_no_stale_pose_and_truth_not_in_packet(self):
        w=World(dict(dropout=1))
        packet=w.observe();step=model_step(packet)
        self.assertFalse(step['det_ok'])
        for key in ('dist_z','yaw_smooth','offset_smooth','detected_length','target_bearing_deg'):
            self.assertIsNone(step[key])
        for key in ('face_corners_px','face_corners_camera_m','pos_x_m','pos_z_m'):
            self.assertIsNone(step['vision_meta'][key])
        self.assertNotIn('ground_truth',json.dumps(packet))
        self.assertNotIn('sampled_error',json.dumps(packet))

    def test_fov_is_based_on_true_geometry_not_pose_error(self):
        normal=World(dict(perception='fov',x=4,z=2))
        biased=World(dict(perception='fov',x=4,z=2,x_bias=-2,z_bias=2))
        self.assertFalse(normal.observe()['step']['det_ok'])
        self.assertFalse(biased.observe()['step']['det_ok'])
        self.assertEqual(normal.last_diagnostic['fraction'],biased.last_diagnostic['fraction'])

    def test_fov_has_no_distance_or_projected_size_cutoff(self):
        from protocol_world import observation
        options=scenario_options(dict(camera_range=6,face_selection='fixed'))
        self.assertNotIn('camera_range',options)
        sample=GaussianResultModel(options).sample()
        for distance in (6.01,10,100,1_000_000):
            packet,diagnostic=observation(dict(x=0,z=distance,yaw=0),0,0,0,options,sample)
            self.assertTrue(packet['step']['det_ok'],distance)
            self.assertAlmostEqual(diagnostic['fraction'],1)
        for x,z,missed in ((100,100,False),(0,100,True),(0,-100,False)):
            packet,_=observation(dict(x=x,z=z,yaw=0),0,0,0,options,{**sample,'missed':missed})
            self.assertFalse(packet['step']['det_ok'])

    def test_invalid_can_transaction_does_not_move_or_partially_write(self):
        w=World();w.observe()
        before=json.dumps(w.diagnostics(),sort_keys=True)
        with self.assertRaises(ValueError):
            w.advance(.1,[can_frame(0x764,[0],0),can_frame(0x1e3,[127]*8,.2)])
        self.assertEqual(before,json.dumps(w.diagnostics(),sort_keys=True))
        for frames in ([can_frame(0x764,[0],.05),can_frame(0x764,[0],0)],
                       [{**can_frame(0x764,[0],0),'dlc':8}]):
            with self.assertRaises(ValueError):
                w.advance(.1,frames)

    def test_model_matches_actual_launcher_keyword_contract(self):
        tree=ast.parse((ROOT.parent/'depth_cam/main_rec.py').read_text(encoding='utf-8-sig'))
        kwargs=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=='fsm_step_kwargs' for t in n.targets))
        meta=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and
                      t.value.id=='fsm_step_kwargs' and isinstance(t.slice,ast.Constant) and
                      t.slice.value=='vision_meta' for t in n.targets))
        step=World().observe()['step']
        self.assertEqual(set(step),{k.arg for k in kwargs.keywords}|{'vision_meta'})
        self.assertEqual(set(step['vision_meta']),{k.value for k in meta.keys})

    def test_standalone_environment_does_not_import_fsm_or_inference(self):
        code="import serve_world,sys; serve_world.protocol_info(); assert not any(k.split('.')[0] in ('calib','torch','cv2','ultralytics') for k in sys.modules)"
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,capture_output=True)

    def test_seed_reproduces_errors_misses_and_variable_timing(self):
        options=dict(model_hz=13,model_interval_sd=.015,latency_sd=.02,x_noise=.1,dropout=.2)
        a,b=World(options),World(options)
        seen=[]
        for _ in range(100):
            self.assertEqual(a.observe(),b.observe())
            seen.append(a.observe()['result_interval_s'])
            a.advance(.03,[]);b.advance(.03,[])
        self.assertGreater(len(set(seen)),10)


class HttpTests(unittest.TestCase):
    def test_different_controllers_use_same_world_http_interface(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),WorldHandler)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}'
            with contextlib.redirect_stdout(io.StringIO()):
                demo=run(url,'demo',6)
                current=run(url,'current',15)
            self.assertGreater(demo['diagnostics']['truth']['world_z'],.2)
            self.assertLess(demo['diagnostics']['truth']['world_z'],.4)
            self.assertEqual(current['controller_state'],'DONE')
            for result in (demo,current):
                self.assertEqual(result['model']['protocol'],'lift.model.v1')
                self.assertGreater(result['diagnostics']['can']['received_frames'],100)
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__':
    unittest.main()
