"""Presentation must use recorded frames without driving or resampling the FSM."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from process_runtime import Session

ROOT=Path(__file__).resolve().parents[1]


class DisplaySyncTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows native presentation regression')
    def test_native_seek_does_not_advance_fsm_world_randomness_or_logs(self):
        with tempfile.TemporaryDirectory() as logs:
            live=Session(dict(perception='fov',seed=25,x_noise=.01,yaw_noise=1,dropout=.1),show_window=True,log_root=logs)
            reference=Session(live.options,log_root=None)
            try:
                initial=copy.deepcopy(live.initial)
                ack=live.present(initial,1)
                self.assertTrue(ack['window_open'])
                self.assertIsNone(live.logger)
                latest=None
                for _ in range(12):
                    latest=live.step();expected=reference.step()
                    for key in ('t','state','command','truth','observation','can','lifecycle'):
                        self.assertEqual(latest[key],expected[key])
                before=(live.now,copy.deepcopy(live.world.can_log),copy.deepcopy(live.inputs),live.logging_info())
                for serial,frame in enumerate((latest,initial,latest,initial),2):
                    ack=live.present(frame,serial)
                    self.assertEqual((ack['presentation_id'],ack['t']),(serial,frame['t']))
                    self.assertTrue(ack['window_open'])
                self.assertEqual(before,(live.now,live.world.can_log,live.inputs,live.logging_info()))
                # A bad image is a presentation error, not a controller failure.
                failed=live.present(latest,99,jpeg='invalid jpeg')
                self.assertIn('error',failed)
                self.assertFalse(live.controller.closed)
                self.assertEqual(before,(live.now,live.world.can_log,live.inputs,live.logging_info()))
                actual=live.step();expected=reference.step()
                for key in ('t','state','command','truth','observation','can','lifecycle'):
                    self.assertEqual(actual[key],expected[key])
            finally:
                live.close();reference.close()

    def test_snapshot_uses_exact_jpeg_current_can_and_recorded_state_without_throttle(self):
        code="""
import base64
from unittest.mock import patch
import cv2
import numpy as np
from current_fsm_driver import CurrentFSMDriver
from fsm_window import FSMWindow
from protocol_world import World
driver=CurrentFSMDriver()
frame=dict(t=8.5,**driver.telemetry(),truth=World().plant.truth(),model_packet=World().observe(),
           can=dict(id=0x1e3,data=[67]*8),lifecycle=dict(outcome='running'))
encoded=base64.b64encode(cv2.imencode('.jpg',np.full((480,640,3),77,dtype=np.uint8))[1]).decode()
with patch.object(cv2,'namedWindow'),patch.object(cv2,'resizeWindow'),patch.object(cv2,'imshow') as show,patch.object(cv2,'destroyWindow'),patch.object(cv2,'waitKey',return_value=-1),patch.object(cv2,'getWindowProperty',return_value=1),patch('fsm_window.time.monotonic',return_value=1):
    window=FSMWindow()
    window.render_snapshot(frame,{}, {}, {},encoded)
    assert show.call_args.args[1].shape==(720,1500,3)
    assert np.all(show.call_args.args[1][:360,:480]==77)
    assert window.last_can=='43 43 43 43 43 43 43 43'
    frame['t']=2.0
    frame['can']['data']=[127]*8
    window.render_snapshot(frame,{}, {}, {},encoded)
    assert show.call_count==2  # same state/command and wall time, backwards virtual time
    assert window.last_can=='7F 7F 7F 7F 7F 7F 7F 7F'
    assert window.last_render_args is None  # background camera cannot redraw a stale frame
    assert window.camera_future is None  # JPEG is shared, not rendered a second time
    window.close()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':
    unittest.main()
