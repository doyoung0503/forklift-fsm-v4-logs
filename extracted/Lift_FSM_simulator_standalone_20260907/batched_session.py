"""One world+FSM worker per live run; IPC batches never skip inner ticks."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from controller_process import ControllerProcess


class BatchedSession:
    def __init__(self,options=None,overrides=None,capture=True,log_root=None,show_window=False):
        self.worker=ControllerProcess(command=[sys.executable,'-u','-B',str(Path(__file__).with_name('batch_worker.py'))],timeout=30)
        self.transport_perf=dict(requests=0,roundtrip_ms=0.,worker_operation_ms=0.,transport_residual_ms=0.,merge_ms=0.)
        self.frames=[]
        self.closed=False
        self._update(self.worker.request('create',options=options,overrides=overrides,capture=capture,
            log_root=str(log_root or Path(__file__).parent/'run_logs'),show_window=show_window))

    def _update(self,data):
        self.frames.extend(data.pop('frames',[]))
        self.__dict__.update(data)
        self.logger=SimpleNamespace(directory=Path(self.log_directory)) if self.log_directory else None
        return self.frames[-1] if self.frames else self.initial

    def step(self):
        return self._update(self.worker.request('advance',ticks=1))

    def step_batch(self,target):
        start=time.perf_counter()
        data=self.worker.request('advance',target=min(target,self.now+.1),ticks=120)
        elapsed=(time.perf_counter()-start)*1000
        worker_ms=data.pop('worker_operation_ms',0.)
        mark=time.perf_counter()
        frame=self._update(data)
        p=self.transport_perf
        p['requests']+=1;p['roundtrip_ms']+=elapsed;p['worker_operation_ms']+=worker_ms
        p['transport_residual_ms']+=elapsed-worker_ms
        p['merge_ms']+=(time.perf_counter()-mark)*1000
        return frame

    def present(self,frame,presentation_id,jpeg=None,options=None):
        return self.worker.request('present',frame=frame,presentation_id=presentation_id,jpeg=jpeg,options=options)

    def report(self):
        path=self.worker.request('report')['path']
        return json.loads(Path(path).read_text(encoding='utf8'))

    def close(self):
        if not self.closed:
            self.closed=True
            self.worker.close()


class LocalController:
    """Same adapter methods as the tick IPC worker, but called in-process."""
    def __init__(self):
        import os
        self.pid=os.getpid();self.closed=False;self.window=None
        self.phase_perf=dict(window_pump_ms=0.,window_pump_calls=0,fsm_ms=0.,fsm_calls=0)
        self.process=SimpleNamespace(poll=lambda:0 if self.closed else None)

    def request(self,op,**data):
        from current_fsm_driver import CurrentFSMDriver
        if op=='init':
            self.driver=CurrentFSMDriver(data.get('overrides'))
            self.show=data.get('show_window',False)
            return dict(info=CurrentFSMDriver.info(),pid=self.pid,telemetry=self.driver.telemetry())
        if op=='tick':
            if self.window:
                mark=time.perf_counter()
                pumped=self.window.pump()
                self.phase_perf['window_pump_executed']=self.phase_perf.get('window_pump_executed',0)+int(bool(pumped))
                self.phase_perf['window_pump_ms']+=(time.perf_counter()-mark)*1000
                self.phase_perf['window_pump_calls']+=1
            if self.window and self.window.closed:
                stopped=self.driver.stop(data['t'])
                return dict(current_can=stopped['can_frames'],future_can=[],advance_until=data['t'],
                    model_updated=False,fsm_updated=False,inputs=None,
                    telemetry={**self.driver.telemetry(),'command':'STOP'},
                    lifecycle=dict(finished=True,outcome='cancelled',reason='simulation window closed'),cancel_requested=True)
            mark=time.perf_counter()
            result=self.driver.tick(data['t'],data['until'],data.get('model'),data.get('halt',False),data.get('imu'))
            self.phase_perf['fsm_ms']+=(time.perf_counter()-mark)*1000
            self.phase_perf['fsm_calls']+=1
            result['telemetry'].update(runtime_mode='simulation',window_open=bool(self.window and not self.window.closed))
            return result
        if op=='stop':return self.driver.stop(data['t'])
        if op=='present':
            if self.show and self.window is None:
                from fsm_window import FSMWindow
                self.window=FSMWindow()
            if self.window and not self.window.closed:
                self.window.render_snapshot(data['frame'],data['options'],data['config'],self.driver.overrides,data.get('jpeg'))
            return dict(presentation_id=data['presentation_id'],t=data['frame']['t'],window_open=bool(self.window and not self.window.closed))
        raise ValueError(op)

    def close(self):
        self.closed=True
        if self.window:self.window.close()
