"""Audit added error, endpoint mapping, and actual FSM early-stop behaviour."""
import ast
import json
import math
from pathlib import Path

from calibrate_lateral import placement
from process_runtime import Session
from protocol_world import Plant, SPEC, scenario_options
from v4_runtime import cfg
from calib.fsm_v4.motion import forward_seconds


def main():
    output = Path('verification/lateral_motion_audit_20260909')
    output.mkdir(parents=True, exist_ok=True)
    plant = Plant(scenario_options(placement(4., .55)))
    plant.drive_gain = plant.rotation_gain = 1.
    forward = []
    for distance in (.1, .2, .8, 1.9):
        hold = forward_seconds(distance)
        actual = plant.drive_distance(hold)
        assert abs(actual-distance) < 1e-10
        forward.append(dict(target_m=distance, hold_s=hold, modeled_m=actual))
    rotation = []
    for angle in (.5, 3.5, 10.):
        hold = cfg.ROTATION_RESPONSE.command_seconds(angle)
        actual = plant.rotation_angle(hold)
        assert abs(actual-angle) < 1e-10
        rotation.append(dict(target_deg=angle, hold_s=hold, modeled_deg=actual))
    runs=[]
    for disable in (False, True):
        overrides={'COARSE_IMU_ENABLED':False}
        if disable:
            overrides['FWD_PREDICTIVE_MAX_ADVANCE_M']=0.
        session=Session(placement(4.,.55),overrides,capture=True,log_root=None)
        report=session.run()
        segments=[]
        active=None
        last=None
        for frame in report['frames']:
            if frame['state']==last:
                continue
            last=frame['state']
            if frame['state']=='WAYPOINT_DRIVE':
                active=frame
            elif frame['state']=='WAYPOINT_DRIVE_SETTLE' and active is not None:
                start,end=active['truth'],frame['truth']
                displacement=math.hypot(end['world_x']-start['world_x'],end['world_z']-start['world_z'])
                info={}
                for line in frame['lines']:
                    if '[V4 MOVE] predictive STOP: ' in line:
                        info=ast.literal_eval(line.split('predictive STOP: ',1)[1])
                segments.append(dict(start_t=active['t'],stop_t=frame['t'],
                    actual_travel_m=displacement,stop_info=info,
                    start_lines=active['lines'],start_truth=start,stop_truth=end))
                active=None
        # The world receives only CAN frames. Pair movement changes to check
        # the measured motion against the independent endpoint time law.
        intervals=[]
        prev=None
        for wire in report['can_frames']:
            if wire['id']!=0x1e3:
                continue
            axes=(127-wire['data'][1],127-wire['data'][2])
            if prev and axes==prev['axes']:
                continue
            if prev and prev['axes']!=(0,0):
                duration=wire['t']-prev['t']
                steer,drive=prev['axes']
                start_frame=min(report['frames'],key=lambda f:abs(f['t']-prev['t']))
                stop_frame=min(report['frames'],key=lambda f:abs(f['t']-wire['t']))
                a,b=start_frame['truth'],stop_frame['truth']
                observed_m=math.hypot(b['world_x']-a['world_x'],b['world_z']-a['world_z'])
                observed_deg=abs(b['heading']-a['heading'])
                intervals.append(dict(start_t=prev['t'],stop_t=wire['t'],hold_s=duration,
                    steer=steer,drive=drive,
                    observed_m=observed_m, observed_deg=observed_deg,
                    frame_time_error_s=max(abs(start_frame['t']-prev['t']),abs(stop_frame['t']-wire['t'])),
                    endpoint_deg=plant.rotation_angle(duration)*abs(steer)/SPEC.ROTATE_REFERENCE_DEFLECTION if steer else 0.,
                    endpoint_m=plant.drive_distance(duration)*abs(drive)/SPEC.DRIVE_REFERENCE_DEFLECTION if drive else 0.))
            prev=dict(t=wire['t'],axes=axes)
        result=dict(disable_predictive=disable,segments=segments,can_intervals=intervals,
            **{k:report[k] for k in ('options','overrides','success','state','reasons','elapsed','source_id','simulator_id','trace_hash')})
        runs.append(result)
        print(json.dumps(result,indent=2),flush=True)
    audit=dict(forward_endpoint_checks=forward,rotation_endpoint_checks=rotation,
               rotation_extra_coast_reduction_deg=cfg.ROT_ACTIVE_COAST_HEURISTIC_REDUCTION_DEG,
               rotation_response=vars(cfg.ROTATION_RESPONSE),runs=runs)
    audit['max_can_endpoint_translation_error_m']=max(abs(i['observed_m']-i['endpoint_m'])
        for r in runs for i in r['can_intervals'] if i['drive'])
    audit['max_can_endpoint_rotation_error_deg']=max(abs(i['observed_deg']-i['endpoint_deg'])
        for r in runs for i in r['can_intervals'] if i['steer'])
    (output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
