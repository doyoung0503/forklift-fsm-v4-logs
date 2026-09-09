"""Offline regression scenarios for TX-timed rotation and near-range replanning."""
import json
from pathlib import Path
from process_runtime import Session
from calibrate_lateral import placement


def main():
    output=Path('verification/rotation_near_replan_20260909')
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for distance,lateral in ((4.5,.64125),(4.5,.7),(3.,.4),(3.,-.4),
                             (3.5,.5),(4.,.6),(4.,-.6),(3.,0.)):
        report=Session(placement(distance,lateral),
            {'COARSE_IMU_ENABLED':False,'FWD_PREDICTIVE_MAX_ADVANCE_M':0.},
            capture=True,log_root=None).run()
        changes=[]
        for frame in report['frames']:
            if not changes or frame['state']!=changes[-1]['state']:
                changes.append({k:frame[k] for k in ('t','state','truth','lines')})
        result=dict(distance_m=distance,lateral_m=lateral,
            near_replans=sum(any('NEAR REPLAN' in line for line in e['lines']) for e in report['events']),
            timed_stops=sum(any('TX-timed STOP' in line for line in e['lines']) for e in report['events']),
            changes=changes,events=report['events'],
            **{k:report[k] for k in ('success','state','reasons','elapsed','source_id','simulator_id',
                                    'trace_hash','options','overrides','collision','final')})
        results.append(result)
        (output/f'case_{len(results):02d}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps({k:result[k] for k in ('distance_m','lateral_m','success','reasons',
                                             'near_replans','timed_stops','elapsed')}),flush=True)
    (output/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
