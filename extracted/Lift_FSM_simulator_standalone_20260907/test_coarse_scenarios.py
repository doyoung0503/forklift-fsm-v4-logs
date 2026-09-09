"""Offline yaw sweep using the current adaptive lateral trigger; virtual CAN only."""
import json
from pathlib import Path
from process_runtime import Session,replay_trace


def main():
    output=Path('verification/coarse_adaptive_yaw_sweep')
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for yaw in (10,15,20,25,30,35,40):
        options=dict(x=0,z=4,yaw=yaw,latency=0,latency_sd=0,dropout=0,
                     model_hz=30,vertical_offset=.575,face_selection='fixed')
        session=Session(options,{},log_root=output/'logs')
        report=session.run()
        changes=[]
        last=None
        for f in report['frames']:
            if f['state']!=last:
                changes.append({k:f.get(k) for k in ('t','state','command','truth','target','lines')})
                last=f['state']
        replay=replay_trace(report['trace'],report['overrides'],report['source_id'])
        result=dict(yaw=yaw,state=report['state'],success=report['success'],
            elapsed=report['elapsed'],reasons=report['reasons'],changes=changes,
            coarse_started=any(f['state']=='COARSE_PREPARE' for f in changes),
            coarse_completed=any(f['state']=='COARSE_REACQUIRE' for f in changes)
                and any(a['state']=='COARSE_REACQUIRE' and b['state']=='ACQUIRE_VERIFY'
                        for a,b in zip(changes,changes[1:])),
            replay=replay,options=report['options'],overrides=report['overrides'],
            logging=report['logging'],final=report['final'],source_id=report['source_id'])
        (output/f'yaw_{yaw}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        results.append(result)
        print(json.dumps({k:result[k] for k in ('yaw','state','success','coarse_started','coarse_completed','elapsed','reasons')}) ,flush=True)
    (output/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
