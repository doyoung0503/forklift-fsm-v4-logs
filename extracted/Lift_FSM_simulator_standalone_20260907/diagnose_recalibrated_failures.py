"""Read saved truth poses; compare collision shapes and insertion turn grids."""
import json
import hashlib
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v4_runtime import cfg
from protocol_world import clip, SPEC
from calib.fsm_v4.insertion_geometry import check_insertion_sweep
from calib.fsm_v4.pose import VisualPoseFilter
from calib.fsm_v4 import planner as p


ROOT = Path('verification/lateral_recalibrated_20260909')
OUT = Path('verification/recalibrated_failure_diagnosis_20260909')


def pose(t):
    return VisualPoseFilter().seed(t['yaw'], t['x'], t['z'], 100.)


def monitor(t, options):
    # Historical monitor reproduced ONLY for offline diagnosis of saved runs.
    # The active simulator no longer runs this continuous-wall calculation.
    c,s=math.cos(math.radians(t['yaw'])),math.sin(math.radians(t['yaw']))
    half=SPEC.FORK_OUTER_SPAN_M/2
    limit=SPEC.INSERT_OPENING_SPAN_M/2
    result=[]
    for lo,hi in ((-half,-half+options['fork_width']),(half-options['fork_width'],half)):
        points=[]
        for x,z in ((lo,0.),(hi,0.),(hi,SPEC.CAMERA_TO_FORK_TIP_Z_M),(lo,SPEC.CAMERA_TO_FORK_TIP_Z_M)):
            dx,dz=x-t['x'],z-t['z']
            points.append((c*dx-s*dz,s*dx+c*dz))
        engaged=clip(clip(points,1,0.,True),1,options['pallet_depth'],False)
        if engaged:
            result.append(min(min(p[0] for p in engaged)+limit,limit-max(p[0] for p in engaged)))
    return min(result) if result else None


def main():
    provenance=json.loads((ROOT/'provenance.json').read_text(encoding='utf8'))
    for name in ('calib/fsm_v4/planner.py','calib/fsm_v4/insertion_geometry.py'):
        path=Path('../depth_cam')/name
        hashes={k.replace('\\','/'):v for k,v in provenance['files'].items()}
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[name]:
            raise RuntimeError('Geometry/planner source changed since calibration: '+name)
    rows = [json.loads(f.read_text(encoding='utf8')) for f in ROOT.glob('trial_*.json')]
    collisions = []
    for r in rows:
        if not r['collision']:
            continue
        accepted = next(c['truth'] for c in r['changes'] if c['state']=='READY_TO_INSERT')
        final = r['final']
        with patch.object(cfg, 'FORK_WIDTH_M', r['options']['fork_width']):
            nominal = check_insertion_sweep(pose(accepted), enforce_entry_distance=False)
            # Same straight sweep extended to the observed stop position.
            with patch.object(cfg, 'INSERT_CAMERA_Z_REMAINDER_M', final['z']):
                actual = check_insertion_sweep(pose(accepted), enforce_entry_distance=False)
        hit = r['collision']
        a = math.radians(hit['yaw'])
        fx = (-1 if hit['side']=='left' else 1)*SPEC.FORK_OUTER_SPAN_M/2
        dx, dz = fx-hit['x'], SPEC.CAMERA_TO_FORK_TIP_Z_M-hit['z']
        tip = (math.cos(a)*dx-math.sin(a)*dz, math.sin(a)*dx+math.cos(a)*dz)
        collisions.append(dict(distance_m=r['distance_m'], lateral_m=r['lateral_m'],
            accepted=accepted, final=final, first_collision=hit,
            colliding_tip_pallet_coordinates_m=tip,
            nominal_nine_block=vars(nominal), actual_nine_block=vars(actual),
            nominal_continuous_clearance_m=monitor(dict(accepted,z=.3),r['options']),
            final_continuous_clearance_m=monitor(final,r['options']),
            forward_overrun_m=.3-final['z']))
    scans=[]
    for d,l,which in ((3.,.28125,0),(3.,.3,0),(5.,.91125,-1),(5.,1.0125,-1)):
        r=next(r for r in rows if r['distance_m']==d and r['lateral_m']==l)
        stops=[c for c in r['changes'] if c['state']=='STAGING_PLAN' and c['truth']['z']<2.2]
        t=stops[which]['truth']
        pp=pose(t)
        intervals=[]
        with patch.object(cfg,'FORK_WIDTH_M',r['options']['fork_width']):
            for i in range(-2000,2001):
                angle=i*.01
                predicted=p._pose_after_action(pp,angle,0.)
                if abs(predicted.yaw_deg)<=cfg.FINAL_YAW_TOL_DEG and check_insertion_sweep(
                        predicted,enforce_entry_distance=False).ok:
                    if not intervals or angle-intervals[-1][1]>.01001:
                        intervals.append([angle,angle])
                    else:
                        intervals[-1][1]=angle
            candidates=set(p._candidate_turns(pp.yaw_deg))
            for i in range(1,int(math.ceil(cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG/cfg.INSERT_FINE_MIN_TURN_DEG))):
                candidates.update((i*cfg.INSERT_FINE_MIN_TURN_DEG,-i*cfg.INSERT_FINE_MIN_TURN_DEG))
            passing=[v for v in sorted(candidates) if abs(v)>=cfg.INSERT_FINE_MIN_TURN_DEG
                     and abs(p._pose_after_action(pp,v,0.).yaw_deg)<=cfg.FINAL_YAW_TOL_DEG
                     and check_insertion_sweep(p._pose_after_action(pp,v,0.),enforce_entry_distance=False).ok]
        scans.append(dict(distance_m=d,lateral_m=l,truth=t,
                          geometry_valid_intervals_deg=intervals,grid_geometry_valid_deg=passing))
    result=dict(failure_counts=dict(Counter(' / '.join(r['reasons']) for r in rows if not r['success'])),
                note='Saved truth geometry only; angle scans exclude motion visibility checks. No runtime edits.',
                collisions=collisions,turn_scans=scans)
    OUT.mkdir(exist_ok=True)
    (OUT/'diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(result['failure_counts'])
    print('collision cases',len(collisions),'nominal nine-block clear',sum(c['nominal_nine_block']['ok'] for c in collisions),
          'actual nine-block clear',sum(c['actual_nine_block']['ok'] for c in collisions),
          'nominal continuous collisions',sum(c['nominal_continuous_clearance_m']<0 for c in collisions))
    for c in collisions:
        if (c['distance_m'],c['lateral_m']) in ((3.,.28125),(5.,.855625)):
            print(json.dumps(c))
    print(json.dumps(scans,indent=2))


if __name__=='__main__':
    main()
