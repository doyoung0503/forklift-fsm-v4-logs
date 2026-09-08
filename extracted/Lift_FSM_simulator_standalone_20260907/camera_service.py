"""Lazy isolated render worker. The web/world process does not import calib/GL."""
import sys
import threading
from pathlib import Path
from controller_process import ControllerProcess
from protocol_world import scenario_options, PROFILE


class CameraService:
    def __init__(self):
        self.worker=None
        self.lock=threading.Lock()

    def render(self, frame, options):
        options=scenario_options(options)
        with self.lock:
            if self.worker is None or self.worker.closed:
                self.worker=ControllerProcess(command=[sys.executable,'-u','-B',str(Path(__file__).with_name('camera_worker.py'))],timeout=45.)
            return self.worker.request('render',frame=frame,options=options,config=PROFILE['parameters'])

    def close(self):
        with self.lock:
            if self.worker: self.worker.close()


CAMERA=CameraService()
