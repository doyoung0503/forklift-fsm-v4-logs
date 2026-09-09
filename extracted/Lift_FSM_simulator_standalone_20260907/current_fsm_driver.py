"""Current-FSM-specific bindings, loaded only inside the controller process.

The original decision code is unchanged. Private attributes below are optional
diagnostics; class/step, virtual clock and CAN-driver bindings remain adapter
dependencies and are intentionally not imported by the world/UI server.
"""
import ast
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

from v4_runtime import environment,control,top,fingerprint,clean,DEPTH_CAM
from current_can_adapter import CurrentCanTransport
from protocol_world import model_step

# Read the actual runtime diagram's phase definitions and pure state helpers.
# OpenCV rendering and camera/model imports are not needed to display them in HTML.
DIAGRAM_PATH=DEPTH_CAM/'ui/diagram_v4.py'
DIAGRAM={}
DIAGRAM_ERROR=None
DIAGRAM_SHA=None
try:
    source=DIAGRAM_PATH.read_bytes()
    DIAGRAM_SHA=hashlib.sha256(source).hexdigest()
    tree=ast.parse(source.decode('utf-8-sig'))
    nodes=[n for n in tree.body if
           (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='V4_FLOW' for t in n.targets)) or
           (isinstance(n,ast.FunctionDef) and n.name in ('_phase_index','_activity'))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(DIAGRAM_PATH),'exec'),DIAGRAM)
except Exception as error:
    DIAGRAM_ERROR=str(error)


class CurrentFSMDriver:
    @staticmethod
    def info():
        info=fingerprint()
        if DIAGRAM_SHA:
            info['files']={**info['files'],'ui/diagram_v4.py':DIAGRAM_SHA}
            info['source_id']=hashlib.sha256(json.dumps(info['files'],sort_keys=True).encode()).hexdigest()
        info['diagram']=dict(source='ui/diagram_v4.py',
            phases=[dict(label=label,states=sorted(states)) for label,states in DIAGRAM.get('V4_FLOW',[])],
            error=DIAGRAM_ERROR)
        return info

    def __init__(self,overrides=None):
        self.overrides=overrides or {}
        self.transport=CurrentCanTransport(control)
        self.sequence=None
        self.now=0.
        self.lines=[]
        self.input_mode='waiting'
        with environment(0.,self.overrides),self.transport.bind(0.):
            self.fsm=top.CalibrationFSMV4()

    def telemetry(self):
        f=self.fsm
        diagram={}
        try:
            diagram=dict(phase_index=DIAGRAM['_phase_index'](f),activity=DIAGRAM['_activity'](f.state),
                         recovery_origin=getattr(f,'_visual_recovery_origin_state',None))
        except Exception:
            pass  # Optional UI definitions must never change the controller decision.
        return clean(dict(state=getattr(f,'state','RUNNING'),diagram=diagram,input_mode=self.input_mode,
            vision_independent=bool(getattr(f,'vision_independent',False)),
            skip_detection=bool(getattr(f,'skip_detection',False)),
            model_requested=not bool(getattr(f,'skip_detection',False) or getattr(f,'vision_independent',False)) and not self.lifecycle()['finished'],
            command=getattr(getattr(f,'execu',None),'last_cmd',None) or 'STOP',
            failure_reason=getattr(f,'failure_reason',None),failure_state=getattr(f,'failure_state',None),
            remaining=getattr(f,'_insert_remaining_m',None),
            stable_frames=len(getattr(f,'_samples',())),
            target=getattr(f,'motion_target_status',None),lines=self.lines))

    def lifecycle(self):
        # Only this adapter knows the current controller's terminal state names.
        state=getattr(self.fsm,'state',None)
        return dict(finished=state in ('DONE','FAILED'),
                    outcome='success' if state=='DONE' else 'failure' if state=='FAILED' else 'running',
                    reason=getattr(self.fsm,'failure_reason',None))

    def tick(self,now,until,model=None,halt=False,imu=None):
        self.now=now
        if imu is not None:
            sample=(imu['yaw_deg'],1000.+imu['t'],imu['rate_deg_s'],None)
            self.fsm.imu_source=SimpleNamespace(snapshot=lambda:sample)
        fresh=model is not None and not model.get('pending',False) and model['sequence']!=self.sequence
        with environment(now,self.overrides),self.transport.bind(now):
            blind=bool(getattr(self.fsm,'vision_independent',False))
            inputs=(dict(det_ok=False,detected_length=None,dist_z=None,yaw_smooth=None,
                         offset_smooth=None,target_bearing_deg=None,vision_meta=None) if blind else
                    model_step(model) if model is not None else None)
            self.input_mode='clock' if blind else 'model' if fresh else 'held'
            if blind or fresh:
                self.lines=[line[0] for line in self.fsm.step(**inputs)]
            if fresh:
                self.sequence=model['sequence']
            lifecycle=self.lifecycle()
            if halt and not lifecycle['finished']:
                control.issue_command_stop()
                lifecycle=dict(finished=True,outcome='timeout',reason='simulation time limit')
            current=self.transport.pump(now)
            telemetry=self.telemetry()
            if halt:
                telemetry['command']='STOP'
            end=now if lifecycle['finished'] else until
            if not lifecycle['finished'] and getattr(self.fsm,'vision_independent',False):
                end=min(end,now+.01)  # Actual launcher's no-vision loop uses waitKey(10).
            future=self.transport.pump(end)
        self.now=end
        return dict(current_can=current,future_can=future,advance_until=end,
                    model_updated=fresh,fsm_updated=blind or fresh,inputs=inputs,
                    telemetry=telemetry,lifecycle=lifecycle)

    def stop(self,now):
        with environment(now,self.overrides),self.transport.bind(now):
            control.issue_command_stop()
            return dict(can_frames=self.transport.pump(now))
