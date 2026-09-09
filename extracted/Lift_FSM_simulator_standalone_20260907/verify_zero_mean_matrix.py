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

OUT=Path(__file__).parent/'verification/zero_mean_matrix_20260909'

def execute(case):
    with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
        result=run_case(case)
    result['group']=case['group']
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base=dict(x=0,z=4,yaw=0,seed=20260908,max_seconds=185,latency=0,latency_sd=0,dropout=0,
        drive_scale_sd=.0607956831911769,rotation_scale_sd=.0607956831911769,imu_angle_sd_deg=5.,
        model_hz=30,x_noise=0,y_noise=0,z_noise=0,yaw_noise=0,vertical_offset=.425,face_selection='fixed')
    cases=[]
    def add(group,**options):cases.append(dict(case_id=len(cases),group=group,options={**base,**options},overrides={}))
    for seed in [11,12,13]:
        for z,x,yaw in itertools.product([2.,3.,4.],[-.3,0,.3],[-10,-5,0,5,10]):add('basic',z=z,x=x,yaw=yaw,seed=seed)
        for yaw in [-40,-35,-30,-25,-20,-15,-10,10,15,20,25,30,35,40]:add('large_yaw',yaw=yaw,seed=seed)
        for z,x in itertools.product([2.,3.,4.],[-.6,.6]):add('lateral',z=z,x=x,seed=seed)
        for yaw in [-15,0,15]:
            add('perception_noise',yaw=yaw,seed=seed,x_noise=.01,y_noise=.01,z_noise=.02,yaw_noise=1,latency_sd=.05,dropout=.05)
            add('observation_loss',yaw=yaw,seed=seed,loss_start=8,loss_duration=3)
        # Diagnostic control only: isolate the influence of IMU angle noise.
        for yaw in [-25,-15,0,15,25]:add('control_imu_zero',yaw=yaw,seed=seed,imu_angle_sd_deg=0.)
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
