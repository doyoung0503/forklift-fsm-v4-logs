"""World/UI runtime. Never imports the FSM, its configuration or CAN driver."""
import hashlib
import json
import math
import os
import random
import threading
import time
from collections import Counter
from pathlib import Path

from controller_process import ControllerProcess,ControllerProcessError
from protocol_world import World,RequestedWorld,PROFILE,SPEC,scenario_options
from run_logging import RunLogger

HERE=Path(__file__).resolve().parent
LOCK=threading.RLock()
SIMULATOR_NAMES=('process_runtime.py','controller_process.py','fsm_worker.py','current_fsm_driver.py','batched_session.py','batch_worker.py',
    'v4_runtime.py','current_can_adapter.py','serve_v4.py','serve_world.py','protocol_client.py',
    'protocol_world.py','model_result_model.py','vehicle_profile.json','view_geometry.js','v4_sim.js','run_logging.py','fsm_window.py',
    'camera_view.js','camera_service.py','camera_worker.py','simulation_camera.py','camera_preview.py','runtime_camera_overlay.py')
# Parent code is pinned for its lifetime; a fresh FSM process is used per run.
SIMULATOR_HASHES={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in SIMULATOR_NAMES}
SIMULATOR_ID=hashlib.sha256(json.dumps(SIMULATOR_HASHES,sort_keys=True).encode()).hexdigest()


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [clean(v) for v in value]
    if isinstance(value,Path):
        return str(value)
    return value


def metadata(info,pid):
    return {**info,'simulator_id':SIMULATOR_ID,'simulator_files':SIMULATOR_HASHES,
            'execution_mode':'subprocess','controller_pid':pid,'server_pid':os.getpid(),
            'vehicle_profile':PROFILE,
            'boundary':'World/UI process exchanges model packets and CAN frames with a separate FSM process. '
                       'Virtual time and lifecycle use lift.worker.v1; diagnostic telemetry is optional. '
                       'Current-FSM-specific clock/CAN bindings remain inside the worker.'}


def fingerprint():
    with ControllerProcess() as controller:
        result=controller.request('info')
        return metadata(result['info'],result['pid'])


def telemetry(value):
    return {**dict(state='RUNNING',command='UNKNOWN',failure_reason=None,failure_state=None,
                  remaining=None,stable_frames=0,target=None,lines=[]),**(value or {})}


class Session:
    def __init__(self,options=None,overrides=None,capture=True,*,controller_factory=ControllerProcess,log_root=HERE/'run_logs',show_window=False):
        self.options=scenario_options({'inference_mode':'requested', **(options or {})})
        self.overrides=dict(overrides or {})
        self.capture=capture
        self.show_window=bool(show_window)
        self.now=0.
        self.controller_wall_ms=0.
        self.done=False
        self.frames,self.inputs,self.events=[],[],[]
        self.digest=hashlib.sha256()
        self.lifecycle=dict(finished=False,outcome='running',reason=None)
        self.controller_error=None
        self.log_root=Path(log_root) if log_root is not None else None
        self.logger=None
        self.logging_error=None
        self.log_attempted=False
        self.logged_can=0
        self.world=(RequestedWorld if self.options['inference_mode']=='requested' else World)(self.options)
        self.consumed_model_sequence=-1
        self.imu_previous=(0.,self.world.plant.heading)
        self.imu_yaw=self.world.plant.heading
        self.imu_initial_yaw=self.imu_yaw
        self.imu_random=random.Random(self.options['seed'] ^ 0x494d55)
        # One calibration error per run: sigma is the endpoint error at 90 deg.
        # A fixed absolute yaw offset would cancel in relative turn feedback.
        self.imu_gain=1.+self.imu_random.gauss(0.,self.options['imu_angle_sd_deg']/90.)
        self.plant=self.world.plant
        self.world.observe()
        self.controller=controller_factory()
        try:
            result=self.controller.request('init',overrides=self.overrides,show_window=self.show_window,sync_display=True)
            self.provenance=metadata(result['info'],result['pid'])
            self.provenance['imu_error_model']=dict(kind='run_constant_gain_v1',
                reference_turn_deg=90.,endpoint_sd_deg=self.options['imu_angle_sd_deg'],
                sampled_gain=self.imu_gain)
            self.input_id=self.provenance['source_id']
            self.config={**self.provenance['config'],**self.overrides}
            self.status=telemetry(result.get('telemetry'))
            self.initial=self.snapshot()
        except Exception:
            self.controller.close()
            raise

    def logging_info(self):
        return dict(run_id=self.logger.run_id if self.logger else None,
                    directory=str(self.logger.directory) if self.logger else None,
                    status='error' if self.logging_error else 'closed' if self.logger and self.logger.closed else
                           'recording' if self.logger else 'disabled' if self.log_root is None else 'waiting',
                    error=self.logging_error,counts=dict(self.logger.counts) if self.logger else {})

    def _log_call(self,method,*args):
        if not self.logger or self.logger.closed or self.logging_error:
            return
        try:
            getattr(self.logger,method)(*args)
        except Exception as error:
            # Disk failure must be visible, but must not alter FSM/CAN decisions.
            self.logging_error=f'{type(error).__name__}: {error}'
            self.logger.close()

    def _start_logging(self):
        if self.log_attempted or self.log_root is None:
            return
        self.log_attempted=True
        try:
            self.logger=RunLogger(self.log_root,dict(options=self.options,overrides=self.overrides,
                config=self.config,provenance=self.provenance),self.snapshot())
        except Exception as error:
            self.logging_error=f'{type(error).__name__}: {error}'
            return
        self.world.model_observer=lambda packet,diagnostic:self._log_call('model',packet,diagnostic,self.status['state'])
        if not self.world.last_packet.get('pending',False):
            self.world.model_observer(self.world.last_packet,self.world.last_diagnostic)
        self._log_call('flush')

    def _log_can(self):
        self._log_call('can',self.world.can_log[self.logged_can:])
        self.logged_can=len(self.world.can_log)

    def _finish_logging(self):
        self._log_can()
        self._log_call('finish',self.report())
        self.world.model_observer=None

    def snapshot(self):
        return clean(dict(t=self.now,**self.status,truth=self.plant.truth(),
            observation=self.world.last_diagnostic,model_packet=self.world.last_packet,collision=self.plant.collision,
            clearance=self.plant.minimum_clearance,can=dict(id=0x1e3,data=list(self.plant.movement)),
            bus_status=self.plant.bus_status,can_frame_count=len(self.world.can_log),terminal=self.done,
            controller_pid=self.controller.pid,execution_mode='subprocess',lifecycle=self.lifecycle,logging=self.logging_info()))

    def present(self,frame,presentation_id,jpeg=None,options=None):
        if not self.show_window or self.controller.closed or self.controller.process.poll() is not None:
            return dict(presentation_id=presentation_id,t=frame['t'],window_open=False)
        return self.controller.request('present',frame=frame,presentation_id=presentation_id,
            jpeg=jpeg,options=options or self.options,config=PROFILE['parameters'])

    def _fault(self,error):
        self.controller_error=str(error)
        self.controller.close(graceful=False)
        self.lifecycle=dict(finished=True,outcome='error',reason='controller process error: '+str(error))
        self.done=True
        # No synthetic STOP is injected. Loss of the CAN sender triggers the
        # receiver's configured watchdog, then any configured coast settles.
        duration=max(SPEC.HEARTBEAT_TIMEOUT_SEC,SPEC.CONTROL_TIMEOUT_SEC,SPEC.MOVEMENT_TIMEOUT_SEC)
        duration+=max(self.options['drive_coast_sec'],self.options['rotation_coast_sec'])+.02
        while duration>1e-9:
            dt=min(1.,duration)
            self.world.advance(dt,[])
            duration-=dt
        self.now=self.world.now
        self.status['lines']=[self.lifecycle['reason']]
        frame=self.snapshot()
        self.events.append(dict(t=self.now,state=self.status['state'],command=self.status['command'],
                                lines=self.status['lines'],can=frame['can']))
        if self.capture:
            self.frames.append(frame)
        self._log_call('state',frame)
        self._finish_logging()
        frame['logging']=self.logging_info()
        return frame

    def _observe_imu(self):
        previous_t,previous_heading=self.imu_previous
        delta=(self.plant.heading-previous_heading+180.)%360.-180.
        dt=self.now-previous_t
        self.imu_yaw+=delta
        angle_error=(self.imu_yaw-self.imu_initial_yaw)*(self.imu_gain-1.)
        self.imu_previous=(self.now,self.plant.heading)
        return dict(t=self.now,yaw_deg=self.imu_yaw+angle_error,
                    rate_deg_s=(delta/dt if dt>0 else 0.)*self.imu_gain,
                    angle_error_deg=angle_error)

    def step(self):
        if self.done:
            return self.snapshot()
        perf=getattr(self,'phase_perf',None)
        if perf is None:
            perf=self.phase_perf=dict(ticks=0,sensors_ms=0.,controller_ms=0.,world_snapshot_ms=0.,world_advance_ms=0.,trace_ms=0.,logging_ms=0.,tick_total_ms=0.)
        tick_start=mark=time.perf_counter()
        self._start_logging()
        requesting=isinstance(self.world,RequestedWorld) and self.status.get('model_requested',True)
        if requesting and self.world.last_packet['sequence']==self.consumed_model_sequence:
            self.world.request_model()
        packet=self.world.observe()
        imu=self._observe_imu()
        before=(self.status['state'],self.status['command'])
        halt=self.now>=self.options['max_seconds']-1e-9
        until=self.now if halt else min(self.now+1/self.options['hz'],self.world.next_model_time,self.options['max_seconds'])
        if requesting and self.world.pending is None and self.world.next_request_time>self.now+1e-9:
            until=min(until,self.world.next_request_time)
        try:
            perf["sensors_ms"]+=(time.perf_counter()-mark)*1000
            tick_started=time.perf_counter()
            result=self.controller.request('tick',t=self.now,until=until,model=packet,halt=halt,imu=imu,
                scene=dict(truth=self.plant.truth(),options=self.options,config=PROFILE['parameters']))
            self.controller_wall_ms+=(time.perf_counter()-tick_started)*1000
            perf["controller_ms"]+=(time.perf_counter()-tick_started)*1000
            mark=time.perf_counter()
            if result['model_updated']:
                self.consumed_model_sequence=packet['sequence']
            end=result['advance_until']
            if not self.now-1e-8<=end<=until+1e-8:
                raise ValueError('Controller returned invalid time boundary')
            self.world.advance(0.,result['current_can'])
            self.status=telemetry(result.get('telemetry'))
            self.lifecycle=result.get('lifecycle') or dict(finished=False,outcome='running',reason=None)
            self.done=bool(self.lifecycle.get('finished'))
            frame=self.snapshot()
            if before!=(self.status['state'],self.status['command']):
                self.events.append(dict(t=self.now,state=self.status['state'],command=self.status['command'],
                                        lines=self.status['lines'],can=frame['can']))
            perf["world_snapshot_ms"]+=(time.perf_counter()-mark)*1000
            mark=time.perf_counter()
            row=dict(t=self.now,inputs=result.get('inputs',packet['step']),state=self.status['state'],command=self.status['command'],
                     failure_reason=self.status['failure_reason'],failure_state=self.status['failure_state'],
                     can=frame['can'],model_packet=packet,model_updated=result['model_updated'],
                     can_frames=result['current_can']+result['future_can'],advance_until=end,
                     halt_requested=halt,lifecycle=self.lifecycle)
            row['fsm_updated']=result.get('fsm_updated',result['model_updated'])
            row['imu']=imu
            if result.get('cancel_requested'):
                row['cancel_requested']=True
            perf["trace_ms"]+=(time.perf_counter()-mark)*1000
            mark=time.perf_counter()
            self.world.advance(end-self.now,result['future_can'])
            perf["world_advance_ms"]+=(time.perf_counter()-mark)*1000
            mark=time.perf_counter()
            self.digest.update(json.dumps(row,sort_keys=True,allow_nan=False).encode())
            if self.capture:
                self.frames.append(frame);self.inputs.append(row)
            self.now=end
            perf["trace_ms"]+=(time.perf_counter()-mark)*1000
            mark=time.perf_counter()
            self._log_can()
            self._log_call('tick',row,frame)
            if self.done:
                if not self.show_window:
                    self.controller.close()
                self._finish_logging()
            frame['logging']=self.logging_info()
            perf["logging_ms"]+=(time.perf_counter()-mark)*1000
            perf["ticks"]+=1
            perf["tick_total_ms"]+=(time.perf_counter()-tick_start)*1000
            return frame
        except (ControllerProcessError,ValueError,TypeError,KeyError) as error:
            return self._fault(error)

    def run(self):
        try:
            while not self.done:
                self.step()
            return self.report()
        finally:
            if not self.show_window or not self.done or self.controller_error:
                self.close()

    def close(self):
        if self.controller.closed:
            if self.logger and not self.logger.closed:
                self._finish_logging()
            return
        try:
            if not self.done:
                stopped=self.controller.request('stop',t=self.now)
                self.world.advance(0.,stopped['can_frames'])
                self.done=True
                self.lifecycle=dict(finished=True,outcome='cancelled',reason='session closed')
                self.status['command']='STOP'
                self._log_call('state',self.snapshot())
        except (ControllerProcessError,ValueError,KeyError,TypeError) as error:
            self._fault(error)
        finally:
            self.controller.close()
            self._finish_logging()

    def report(self):
        outcome=self.lifecycle.get('outcome','running')
        reasons=[]
        if outcome in ('failure','error','timeout','cancelled'):
            reasons.append(self.lifecycle.get('reason') or outcome)
        remaining=max(0.,self.plant.z-SPEC.INSERT_CAMERA_Z_REMAINDER_M)
        if outcome=='success' and remaining>self.options['completion_tolerance']:
            reasons.append('FSM DONE with insertion distance remaining')
        if not self.done:
            reasons.append('run in progress')
        return clean(dict(options=self.options,overrides=self.overrides,source_id=self.input_id,
            simulator_id=SIMULATOR_ID,provenance=self.provenance,trace_hash=self.digest.hexdigest(),logging=self.logging_info(),
            state=self.status['state'],elapsed=self.now,finished=self.done,
            success=outcome=='success' and not reasons,reasons=reasons,
            failure_reason=self.status['failure_reason'],failure_state=self.status['failure_state'],
            lifecycle=self.lifecycle,controller_error=self.controller_error,
            controller_pid=self.controller.pid,controller_exitcode=self.controller.process.poll(),
            replayable=self.controller_error is None,collision=self.plant.collision,
            minimum_clearance=self.plant.minimum_clearance,fsm_remaining=self.status['remaining'],
            true_remaining=remaining,final=self.plant.truth(),events=self.events,config=self.config,
            vehicle_profile=PROFILE,can_frames=self.world.can_log if self.capture else [],
            can_frame_count=len(self.world.can_log),frames=self.frames,trace=self.inputs))


def replay_trace(trace,overrides=None,source_id=None,simulator_id=None):
    mismatches=[]
    movement=[127]*8
    with ControllerProcess() as controller:
        result=controller.request('init',overrides=overrides or {})
        if source_id is not None and source_id!=result['info']['source_id']:
            raise ValueError('Trace source differs from the new FSM process')
        if simulator_id is not None and simulator_id!=SIMULATOR_ID:
            raise ValueError('Trace simulator version differs')
        for index,row in enumerate(trace):
            result=controller.request('tick',t=row['t'],until=row['advance_until'],
                model=row['model_packet'] if row['model_updated'] else None,
                halt=row.get('halt_requested',False),cancel=row.get('cancel_requested',False),imu=row.get('imu'))
            for f in result['current_can']:
                if f['id']==0x1e3:
                    movement=f['data']
            status=telemetry(result.get('telemetry'))
            actual={k:status[k] for k in ('state','command','failure_reason','failure_state')}
            actual.update(can=dict(id=0x1e3,data=list(movement)),
                          can_frames=result['current_can']+result['future_can'])
            if 'lifecycle' in row:
                actual['lifecycle']=result['lifecycle']
            if 'fsm_updated' in row:
                actual['fsm_updated']=result.get('fsm_updated',result['model_updated'])
            for f in result['future_can']:
                if f['id']==0x1e3:
                    movement=f['data']
            if any(row[k]!=v for k,v in actual.items()):
                mismatches.append(dict(frame=index,t=row['t'],actual=actual,expected={k:row[k] for k in actual}))
    return dict(frames=len(trace),passed=not mismatches,mismatches=mismatches)


def summarize(reports):
    return dict(total=len(reports),success=sum(r['success'] for r in reports),
                fsm_done=sum(r['lifecycle']['outcome']=='success' for r in reports),
                collisions=sum(bool(r['collision']) for r in reports),
                reasons=dict(Counter(reason for r in reports for reason in r['reasons'])))
