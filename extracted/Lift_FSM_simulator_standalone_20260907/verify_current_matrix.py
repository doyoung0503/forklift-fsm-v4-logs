"""Reproducible current FSM audit; no UI or physical hardware."""
import concurrent.futures as cf
import contextlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from parallel_batch import run_case

OUT=Path(__file__).parent/'verification/current_matrix_20260909'

def execute(case):
    with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
        result=run_case(case)
    result['group']=case['group']
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base=dict(x=0,z=4,yaw=0,seed=20260908,max_seconds=185,latency=0,latency_sd=0,dropout=0,
        model_hz=30,x_noise=0,y_noise=0,z_noise=0,yaw_noise=0,vertical_offset=.425,face_selection='fixed')
    cases=[]
    def add(group,**options):cases.append(dict(case_id=len(cases),group=group,options={**base,**options},overrides={}))
    for z,x,yaw in itertools.product([2.,3.,4.],[-.3,0,.3],[-10,-5,0,5,10]):add('nominal',z=z,x=x,yaw=yaw)
    for yaw in [-40,-35,-30,-25,-20,-15,-10,10,15,20,25,30,35,40]:add('coarse',yaw=yaw)
    for yaw,seed in itertools.product([-25,-15,0,15,25],[11,12,13]):
        add('drive_imu_noise',yaw=yaw,seed=seed,drive_scale_sd=.15*math.sqrt(math.pi/2),rotation_scale_sd=.05,imu_scale=95/90)
    for yaw,seed in itertools.product([-15,0,15],[11,12,13]):
        add('perception_noise',yaw=yaw,seed=seed,x_noise=.01,y_noise=.01,z_noise=.02,yaw_noise=1,latency_sd=.05,dropout=.05)
    for z,x in itertools.product([2.,3.,4.],[-.6,.6]):add('lateral_stress',z=z,x=x)
    for yaw in [-15,0,15]:add('observation_loss',yaw=yaw,loss_start=8,loss_duration=3)
    (OUT/'manifest.json').write_text(json.dumps(cases,indent=2),encoding='utf8')
    start=time.perf_counter();results=[]
    with cf.ProcessPoolExecutor(max_workers=min(4,os.cpu_count() or 1),mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
        futures={pool.submit(execute,c):c for c in cases}
        with (OUT/'results.jsonl').open('w',encoding='utf8') as out:
            for future in cf.as_completed(futures):
                c=futures[future]
                try:r=future.result()
                except Exception as e:r=dict(case_id=c['case_id'],group=c['group'],options=c['options'],success=False,state='ERROR',reasons=[str(e)])
                results.append(r);out.write(json.dumps(r)+'\n');out.flush()
                if len(results)%10==0:print(f'{len(results)}/{len(cases)} finished',flush=True)
    print('completed',len(results),'wall',round(time.perf_counter()-start,2),flush=True)

if __name__=='__main__':main()
