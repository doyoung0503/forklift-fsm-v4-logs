"""Read-only diagnosis from saved truth poses; no simulated/physical movement."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'depth_cam'))
from calib.fsm_v4 import config as cfg
from calib.fsm_v4.planner import _pose_after_action
from calib.fsm_v4.insertion_geometry import check_insertion_sweep


def main():
    folder = HERE/'verification/lateral_no_predictive_20260909'
    rows = [json.loads(p.read_text(encoding='utf8')) for p in folder.glob('trial_*.json')]
    selected = [r for r in rows if r['distance_m']==4.5 and r['lateral_m'] in (.64125,.7)]
    results=[]
    for row in sorted(selected,key=lambda r:r['lateral_m']):
        locks=[]
        for c in row['changes']:
            if c['state']!='FINAL_POSE_LOCK':
                continue
            truth=c['truth']
            pose=SimpleNamespace(pallet_x_m=truth['x'],pallet_z_m=truth['z'],
                                 yaw_deg=truth['yaw'],measurement_mono=0.)
            good=[]
            for i in range(-20000,20001):
                angle=i*.001
                predicted=_pose_after_action(pose,angle,0.)
                if abs(predicted.yaw_deg)<=cfg.FINAL_YAW_TOL_DEG and check_insertion_sweep(
                        predicted,enforce_entry_distance=False).ok:
                    good.append(angle)
            intervals=[]
            for angle in good:
                if not intervals or angle-intervals[-1][1]>.00101:
                    intervals.append([angle,angle])
                else:
                    intervals[-1][1]=angle
            examples={}
            for angle in (0.,.25,.5,4.,4.5,-.5):
                p=_pose_after_action(pose,angle,0.)
                examples[str(angle)]=check_insertion_sweep(p,enforce_entry_distance=False).__dict__
            locks.append(dict(t=c['t'],truth=truth,geometry_valid_turn_intervals_deg=intervals,
                              geometry_examples=examples))
        events=[c for c in row['changes'] if c['state'] in (
            'WAYPOINT_TURN','WAYPOINT_DRIVE_SETTLE','FINAL_ROTATE','FINAL_SETTLE','FAILED','READY_TO_INSERT')]
        results.append(dict(initial_lateral_m=row['lateral_m'],success=row['success'],
                            reasons=row['reasons'],locks=locks,events=events,
                            source_id=row['source_id'],trace_hash=row['trace_hash']))
    out=HERE/'verification/lateral_nonmonotonic_diagnosis_20260909'
    out.mkdir(exist_ok=True)
    (out/'diagnosis.json').write_text(json.dumps(dict(
        scan_step_deg=.001, scan_domain_deg=[-20,20],
        note='Geometric endpoint scan only; not a new execution or full swept-rotation visibility proof.',
        minimum_commanded_fine_turn_deg=cfg.INSERT_FINE_MIN_TURN_DEG,
        cases=results),indent=2),encoding='utf8')
    for result in results:
        print(result['initial_lateral_m'],result['success'],
              [p['geometry_valid_turn_intervals_deg'] for p in result['locks']])


if __name__=='__main__':
    main()
