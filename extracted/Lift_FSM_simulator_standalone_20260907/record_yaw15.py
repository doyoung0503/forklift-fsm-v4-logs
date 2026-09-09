"""One actual FSM simulation; export synchronized world and native FSM panels."""
import base64
import bisect
import csv
import json
import math
import sys
from pathlib import Path

from process_runtime import Session, replay_trace


def main():
    drive15='--drive-mae15' in sys.argv
    noisy='--noisy' in sys.argv or drive15
    folder='videos/yaw15_drive_mae15_h50' if drive15 else ('videos/yaw15_imu_mae5' if noisy else 'videos/yaw15_synced')
    output=Path(__file__).parent/folder
    output.mkdir(parents=True,exist_ok=True)
    options=dict(x=0,z=4,yaw=15,model_hz=30,latency=0,latency_sd=0,
                 dropout=0,vertical_offset=.425,face_selection='fixed')
    if noisy:
        # Use a representative +5deg error per90deg for this single video.
        # Store the equivalent zero-mean Gaussian sigma for comparison only.
        sigma=5.*math.sqrt(math.pi/2)
        error90=5.0  # representative absolute error, not a lucky near-zero draw
        options.update(imu_scale=1.+error90/90.,drive_scale_sd=.05,
                       rotation_scale_sd=.05,seed=20260908)
        if drive15:
            options['drive_scale_sd']=.15*math.sqrt(math.pi/2)
        (output/'error_model.json').write_text(json.dumps(dict(
            imu_error_model='representative +5deg at 90deg true rotation; fixed gyro gain',
            expected_abs_error_at_90_deg=5,normal_sigma_deg=sigma,
            representative_error_at_90_deg=error90,imu_scale=options['imu_scale'],
            drive_gain_sigma=options['drive_scale_sd'],
            drive_expected_abs_relative_error=options['drive_scale_sd']*math.sqrt(2/math.pi),
            rotation_gain_sigma=.05,seed=20260908),indent=2),encoding='utf8')
    if '--render-only' in sys.argv:
        report=json.loads((output/'run.json').read_text(encoding='utf8'))
    else:
        session=Session(options,log_root=output/'logs')
        report=session.run()
        (output/'run.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf8')
        check=replay_trace(report['trace'],report['overrides'],report['source_id'],report['simulator_id'])
        assert check['passed'],check
    print('RUN',report['state'],report['elapsed'],report['reasons'],flush=True)
    import cv2
    import numpy as np
    import imageio_ffmpeg
    from simulation_camera import SimulationCamera,top_view,world_geometry,fov_sector
    from fsm_window import FSMWindow
    renderer=SimulationCamera()
    window=FSMWindow(offscreen=True)
    frames=report['frames'];times=[f['t'] for f in frames]
    config=report['provenance']['vehicle_profile']['parameters']
    options=report['options'];fps=15
    points=[]
    for f in frames:
        g=world_geometry(f['truth'],config)
        points.extend(fov_sector(g['camera'],g['rotation'],options['camera_hfov_deg']))
        points.extend([g['pivot']+[-2,-2],g['pivot']+[2,2],g['pallet']+[-1,-1],g['pallet']+[1,1]])
    lo=np.min(points,axis=0)-.5;hi=np.max(points,axis=0)+.5
    center=(lo+hi)/2
    world_options={**options,'top_view_center_x':center[0],'top_view_center_z':center[1],
                   'top_view_scale':min(600/(hi[0]-lo[0]),400/(hi[1]-lo[1]))}
    count=math.ceil(report['elapsed']*fps)+fps
    writer=imageio_ffmpeg.write_frames(str(output/'yaw15_world_fsm.mp4'),(2220,780),fps=fps,
                                      codec='libx264',quality=8,macro_block_size=2)
    writer.send(None)
    with (output/'frame_sync.csv').open('w',newline='',encoding='utf8') as handle:
        manifest=csv.writer(handle);manifest.writerow(['video_frame','video_time_s','world_sim_s','fsm_sim_s','camera_sim_s','state'])
        try:
            for i in range(count):
                wanted=min(report['elapsed'],i/fps)
                frame=frames[max(0,bisect.bisect_right(times,wanted)-1)]
                camera,info=renderer.render(frame,options,config)
                assert abs(info['t']-frame['t'])<1e-9
                jpeg=base64.b64encode(cv2.imencode('.jpg',camera)[1]).decode()
                window.render_snapshot(frame,options,config,report['overrides'],jpeg)
                world=top_view(frame,world_options,config)
                canvas=np.full((780,2220,3),24,np.uint8)
                canvas[180:720,:720]=cv2.resize(world,(720,540))
                canvas[60:,720:]=window.last_image
                cv2.putText(canvas,f'WORLD X-Z | yaw +15 deg | t={frame["t"]:.3f}s',(18,38),0,.8,(240,240,240),2,cv2.LINE_AA)
                cv2.putText(canvas,'FSM + synthetic camera | same logged frame | 1x',(740,38),0,.8,(240,240,240),2,cv2.LINE_AA)
                cv2.putText(canvas,frame['state'],(18,100),0,.7,(100,230,150),2,cv2.LINE_AA)
                if noisy:
                    note='IMU +5deg @90deg; drive MAE=15%; camera50cm' if drive15 else 'IMU +5deg @90deg; drive/turn sigma=5%'
                    cv2.putText(canvas,note,(18,142),0,.55,(220,220,240),1,cv2.LINE_AA)
                writer.send(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
                manifest.writerow([i,i/fps,frame['t'],frame['t'],info['t'],frame['state']])
                if i in (0,count//2,count-1):cv2.imencode('.jpg',canvas)[1].tofile(output/f'preview_{i}.jpg')
                if i%(fps*10)==0:print('VIDEO',i,count,frame['state'],flush=True)
        finally:
            writer.close();window.close()
    print('SAVED',output/'yaw15_world_fsm.mp4',flush=True)


if __name__=='__main__':
    main()
