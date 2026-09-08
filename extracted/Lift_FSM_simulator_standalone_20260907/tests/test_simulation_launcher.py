import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from controller_process import ControllerProcess
from protocol_world import World

ROOT=Path(__file__).resolve().parents[1]
LAUNCHER=ROOT.parent/'depth_cam/main_rec_v4.py'


class SimulationLauncherTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows native window startup regression')
    def test_native_startup_keeps_ipc_responsive_and_does_not_change_can(self):
        packet=World().observe()
        outputs=[]
        for visible in (False,True):
            with ControllerProcess(timeout=5) as worker:
                worker.request('init',show_window=visible)
                result=worker.request('tick',t=0,until=.03,model=packet)
                self.assertEqual(result['telemetry']['window_open'],visible)
                self.assertEqual(worker.request('info')['info']['loaded_camera_inference_modules'],[])
                outputs.append((result['current_can'],result['future_can'],result['lifecycle']))
        self.assertEqual(outputs[0],outputs[1])

    def test_actual_entrypoint_is_mode_specific_and_does_not_load_inference(self):
        with ControllerProcess() as worker:
            self.assertIn(str(LAUNCHER),worker.process.args)
            self.assertIn('simulation',worker.process.args)
            info=worker.request('init',show_window=False)['info']
            self.assertEqual(info['entrypoint'],'main_rec_v4.py')
            self.assertEqual(info['runtime_mode'],'simulation')
            self.assertEqual(info['loaded_camera_inference_modules'],[])
            step=worker.request('tick',t=0,until=.01,model=World().observe())
            self.assertEqual(step['telemetry']['runtime_mode'],'simulation')
            self.assertFalse(step['telemetry']['window_open'])

    def test_simulation_branch_precedes_real_loader_and_model_validation(self):
        code="""
import sys
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import main_rec_v4 as launcher
with patch('simulation_runtime.run_simulation',return_value=0) as simulation, patch.object(launcher,'_configured_runtime_model',side_effect=AssertionError('real loader')):
    assert launcher.main(mode='simulation',simulation_ipc=True,simulation_window=False)==0
    simulation.assert_called_once_with(ipc=True,show_window=False,port=8766,open_browser=True)
assert not any(k.split('.')[0] in ('main_rec','pyrealsense2','torch','ultralytics') for k in sys.modules)
"""
        subprocess.run([sys.executable,'-c',code,str(LAUNCHER.parent)],check=True,capture_output=True)

    def test_invalid_mode_combinations_fail_before_hardware_startup(self):
        for args in (['--mode','simulation','--calibrate'],['--mode','real','--ipc'],['--mode','invalid']):
            result=subprocess.run([sys.executable,str(LAUNCHER),*args],capture_output=True)
            self.assertEqual(result.returncode,2)

    def test_window_cancel_uses_stop_and_has_reproducible_worker_contract(self):
        packet=World().observe()
        outputs=[]
        for _ in range(2):
            with ControllerProcess() as worker:
                worker.request('init',show_window=False)
                outputs.append(worker.request('tick',t=0,until=.03,model=packet,cancel=True))
        self.assertEqual(outputs[0],outputs[1])
        result=outputs[0]
        self.assertEqual(result['lifecycle']['outcome'],'cancelled')
        self.assertEqual(result['advance_until'],0)
        self.assertEqual(result['current_can'][-1]['data'],[127]*8)
        self.assertFalse(result['fsm_updated'])

    def test_native_window_uses_original_diagram_without_camera_launcher(self):
        code="""
import sys
from unittest.mock import patch
from current_fsm_driver import CurrentFSMDriver
from protocol_world import World
from fsm_window import FSMWindow
import cv2
driver=CurrentFSMDriver();packet=World().observe()
with patch.object(cv2,'namedWindow'),patch.object(cv2,'resizeWindow'),patch.object(cv2,'imshow') as show,patch.object(cv2,'destroyWindow'),patch.object(cv2,'waitKey',return_value=-1):
    window=FSMWindow()
    window.render(driver,packet,0,driver.lifecycle(),[])
    image=show.call_args.args[1]
    assert image.shape[0]==720 and image.shape[1]>=1180 and image.shape[2]==3
    assert window.diagram.__module__=='ui.diagram_v4'
    window.close()
assert not any(k.split('.')[0] in ('main_rec','torch','ultralytics','pyrealsense2') for k in sys.modules)
"""
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,capture_output=True)


if __name__=='__main__':
    unittest.main()
