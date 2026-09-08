"""Run real FSM sessions, then render two synchronized views from saved logs."""
from pathlib import Path
import argparse
import bisect
import csv
import hashlib
import json
import math
import subprocess

ROOT=Path(__file__).resolve().parent
CASES=[('01_front',0.,2.,180.),('02_right',.30,2.2,165.),('03_left',-.30,2.2,-165.)]


def run_cases(output,height=.65):
    from process_runtime import Session,replay_trace
    output.mkdir(parents=True,exist_ok=True);cases=[]
    for index,(name,x,z,heading) in enumerate(CASES):
        options=dict(placement_mode='world',forklift_x=0.,forklift_z=-.68,forklift_heading=0.,
            pallet_x=x,pallet_z=z,pallet_heading=heading,vertical_offset=height-.075,
            model_hz=10,latency=.12,dropout=.05,x_noise=.01,y_noise=.01,z_noise=.02,
            yaw_noise=1,seed=20260907+index)
        print(f'Running {name}: {options}',flush=True)
        session=Session(options,log_root=output/'logs');report=session.run()
        print(f'{name}: {report["state"]}, {report["elapsed"]:.2f}s, {report["reasons"]}',flush=True)
        replay=replay_trace(report['trace'],report['overrides'],report['source_id'],report['simulator_id'])
        if not replay['passed']:raise AssertionError(f'{name} replay mismatch')
        record=dict(name=name,options=report['options'],state=report['state'],elapsed=report['elapsed'],
            success=report['success'],reasons=report['reasons'],trace_hash=report['trace_hash'],
            log_directory=report['logging']['directory'],replay_passed=True,replay_frames=replay['frames'])
        (output/f'{name}_report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        cases.append(record)
        (output/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')


def render_cases(output,fps,speed):
    import cv2
    import numpy as np
    import imageio_ffmpeg
    from simulation_camera import SimulationCamera,composite,world_geometry
    cases=json.loads((output/'cases.json').read_text(encoding='utf-8'));renderer=SimulationCamera()
    for case in cases:
        log=Path(case['log_directory']);metadata=json.loads((log/'run_meta.json').read_text(encoding='utf-8'))
        source=log/'run_fsm_steps.jsonl'
        rows=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
        frames=[row['telemetry'] for row in rows]
        for frame,row in zip(frames,rows):
            frame['model_packet']=row['model_packet'];assert abs(frame['t']-row['t'])<1e-8
        options=metadata['options'];config=metadata['provenance']['vehicle_profile']['parameters']
        times=[f['t'] for f in frames];end=times[-1];count=math.ceil(end/speed*fps)+fps
        path=output/f'{case["name"]}_2d_camera.mp4'
        writer=imageio_ffmpeg.write_frames(str(path),(1280,720),fps=fps,codec='libx264',quality=8,macro_block_size=2)
        writer.send(None);trail=[];previous=-1;render_times=[]
        with (output/f'{case["name"]}_video_frames.csv').open('w',newline='',encoding='utf-8-sig') as out:
            manifest=csv.writer(out);manifest.writerow(['video_frame','video_time_s','requested_sim_time_s','log_row','log_time_s','model_sequence','state','camera_world_x','camera_world_z','pallet_world_x','pallet_world_z'])
            try:
                for i in range(count):
                    wanted=min(end,i/fps*speed);j=max(0,bisect.bisect_right(times,wanted)-1);frame=frames[j]
                    if j!=previous:
                        trail.extend(world_geometry(frames[k]['truth'],config)['camera'] for k in range(previous+1,j+1));previous=j
                    camera,info=renderer.render(frame,options,config)
                    composed=composite(frame,camera,options,config,trail,case['name'],speed)
                    writer.send(cv2.cvtColor(composed,cv2.COLOR_BGR2RGB));render_times.append(info['render_ms'])
                    manifest.writerow([i,i/fps,wanted,j,frame['t'],info['model_sequence'],frame['state'],*info['camera_world'],*info['pallet_world']])
                    if i in (0,count//3,2*count//3,count-1):cv2.imencode('.jpg',composed)[1].tofile(output/f'{case["name"]}_{i:05d}.jpg')
                    if i%(fps*10)==0:print(f'{case["name"]}: video {i}/{count}, sim {frame["t"]:.2f}s {frame["state"]}',flush=True)
            finally:writer.close()
        case.update(video=path.name,video_frames=count,video_fps=fps,playback_speed=speed,
            source_log_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            render_ms_median=float(np.median(render_times)),render_ms_p95=float(np.percentile(render_times,95)))
        (output/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'concat.txt').write_text(''.join(f"file '{c['video']}'\n" for c in cases),encoding='utf-8')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','concat','-safe','0',
                    '-i','concat.txt','-c','copy','three_conditions.mp4'],cwd=output,check=True)
    sections='<h2>세 조건 연속 재생</h2><video controls preload="metadata" src="three_conditions.mp4"></video>'
    sections+=''.join(f'<h2>{c["name"]}: 팔레트 ({c["options"]["pallet_x"]}, {c["options"]["pallet_z"]})m / {c["options"]["pallet_heading"]}°</h2><p>FSM {c["state"]} · {c["elapsed"]:.2f}초 · 검증 {"통과" if c["success"] else "미통과"} · {" / ".join(c["reasons"])}</p><video controls preload="metadata" src="{c["video"]}"></video>' for c in cases)
    (output/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>FSM 세 조건 실행</title><style>body{background:#12181d;color:#e4ebf1;max-width:1280px;margin:30px auto;font:16px/1.7 system-ui;padding:20px}video{width:100%}h2{margin-top:40px}</style><h1>실제 FSM 실행 로그 · 2D 좌표 + 가상 카메라</h1><p>두 화면은 같은 로그 시각으로 동기화합니다. 실제 신경망 추론은 없습니다. 재생 '+str(speed)+'배속. 각 영상 마지막 1초는 최종 상태입니다.</p>'+sections,encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'fsm_camera_runs')
    p.add_argument('--render-only',action='store_true');p.add_argument('--run-only',action='store_true')
    p.add_argument('--fps',type=int,default=30);p.add_argument('--speed',type=float,default=1)
    p.add_argument('--height',type=float,default=.65,help='Camera height above ground, metres')
    args=p.parse_args()
    if args.fps<=0 or args.speed<=0:p.error('fps and speed must be positive')
    if not args.render_only:run_cases(args.output,args.height)
    if not args.run_only:render_cases(args.output,args.fps,args.speed)


if __name__=='__main__':main()
