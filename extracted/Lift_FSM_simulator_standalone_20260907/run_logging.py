"""Streaming simulator logs; no imports from the real camera/FSM logger.

Common real-run fields (t_iso, t_mono, frame_i, det_ok, phase) are retained.
All monotonic timestamps use the simulation clock; inference is synthetic.
"""
import csv
import json
import math
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from protocol_world import EPOCH


class RunLogger:
    def __init__(self, root, metadata, initial):
        self.started=datetime.now(timezone.utc)
        self.run_id=self.started.strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid4().hex[:8]
        self.directory=root/self.run_id
        self.directory.mkdir(parents=True,exist_ok=False)
        self.handles={}
        self.geometry=metadata['provenance']['vehicle_profile']['parameters']
        self.options=metadata['options']
        self.counts=dict(can_frames=0,model_results=0,fsm_steps=0,transitions=0,relative_poses=0)
        self.closed=False
        self.last_state=None
        self.last_command=None
        try:
            for name in ('can_tx.jsonl','control_seq.jsonl','model_results.jsonl','fsm_steps.jsonl'):
                self.handles[name]=open(self.directory/('run_'+name),'w',encoding='utf-8',newline='')
            for name in ('pallet_state.csv','inference_timing.csv','relative_pose.csv'):
                self.handles[name]=open(self.directory/('run_'+name),'w',encoding='utf-8-sig',newline='')
            self.state_writer=csv.DictWriter(self.handles['pallet_state.csv'],fieldnames=[
                't_iso','t_mono','sim_time_s','frame_i','fsm_state','align_sub','det_ok',
                'pos_x','pos_y','pos_z','yaw_deg','yaw_model_raw_deg','width_m','fps',
                'measurement_sim_time_s','latency_s','truth_x','truth_y','truth_z','truth_yaw_deg',
                'error_x','error_y','error_z','error_yaw_deg'])
            self.timing_writer=csv.DictWriter(self.handles['inference_timing.csv'],fieldnames=[
                't_iso','t_mono','sim_time_s','frame_i','fsm_state','camera_frame_number',
                'inference_ran','pose_src','model_det_ok','pnp_ok','result_fps',
                'result_interval_s','inference_start_host_mono_ms','inference_end_host_mono_ms',
                'model_inference_ms','pose_result_host_mono_ms','pos_x_m','pos_z_m','yaw_deg'])
            self.state_writer.writeheader();self.timing_writer.writeheader()
            self.pose_writer=csv.DictWriter(self.handles['relative_pose.csv'],fieldnames=[
                't_iso','t_mono','sim_time_s','phase','fsm_state','command',
                'forklift_world_x_m','forklift_world_z_m','forklift_heading_deg',
                'camera_world_x_m','camera_world_z_m','pallet_world_x_m','pallet_world_z_m','pallet_heading_deg',
                'pallet_camera_x_m','pallet_camera_y_m','pallet_camera_z_m','pallet_relative_yaw_deg',
                'pallet_pivot_x_m','pallet_pivot_z_m','pivot_to_pallet_distance_m',
                'fork_tip_world_x_m','fork_tip_world_z_m',
                'fork_tip_pallet_x_m','fork_tip_pallet_z_m',
                'left_outer_tip_pallet_x_m','left_outer_tip_pallet_z_m',
                'right_outer_tip_pallet_x_m','right_outer_tip_pallet_z_m',
                'tip_insertion_depth_m','insertion_remaining_m','collision','minimum_clearance_m'])
            self.pose_writer.writeheader()
            self.write_json('run_meta.json',dict(schema='lift.run.v1',run_id=self.run_id,
                started_utc=self.started.isoformat(),clock=dict(domain='simulation_monotonic',
                    epoch=EPOCH,t_iso='run start UTC + simulation time; not execution wall time'),
                inference_ran=False,notes=[
                    'Model result/timing files describe generated results, not camera or inference execution.',
                    'control_seq records observed states and commands; it does not fabricate real tracer begin/end events.',
                    'A running manifest without a final summary indicates an interrupted process.'],**metadata))
            self.write_json('run_summary.json',dict(run_id=self.run_id,finished=False,lifecycle={'outcome':'running'}))
            self.state(initial,initial=True)
            self.relative_pose(initial,'initial')
            self.flush()
        except Exception:
            self.close()
            raise

    def stamp(self,t):
        return dict(t_iso=(self.started+timedelta(seconds=t)).isoformat(timespec='milliseconds'),
                    t_mono=EPOCH+t,sim_time_s=t)

    def emit(self,name,record):
        self.handles[name].write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')

    def write_json(self,name,record):
        target=self.directory/name
        temp=target.with_suffix(target.suffix+'.tmp')
        temp.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
        temp.replace(target)

    def can(self,frames):
        for frame in frames:
            self.emit('can_tx.jsonl',dict(**self.stamp(frame['t']),phase='can_tx',
                direction='fsm_to_world',id_hex=f"0x{frame['id']:03X}",
                data_hex=' '.join(f'{value:02X}' for value in frame['data']),**frame))
            self.counts['can_frames']+=1

    def model(self,packet,diagnostic,state):
        step=packet['step'];t=packet['sim_time_s'];stamp=self.stamp(t)
        pose=step['offset_smooth'] or [None,None,None]
        truth=diagnostic['ground_truth_pose'];error=diagnostic['sampled_error']
        self.emit('model_results.jsonl',dict(**stamp,frame_i=packet['sequence'],fsm_state=state,
            inference_ran=False,pose_src='simulation',packet=packet,diagnostic=diagnostic))
        self.state_writer.writerow(dict(**stamp,frame_i=packet['sequence'],fsm_state=state,align_sub='',
            det_ok=int(step['det_ok']),pos_x=pose[0],pos_y=pose[1],pos_z=pose[2],
            yaw_deg=step['yaw_smooth'],yaw_model_raw_deg=step['vision_meta'].get('yaw_raw_deg'),
            width_m=step['detected_length'],fps=packet['result_fps'],
            measurement_sim_time_s=diagnostic['measured'],latency_s=diagnostic['latency'],
            truth_x=truth['x'],truth_y=truth['y'],truth_z=truth['z'],truth_yaw_deg=truth['yaw'],
            error_x=error['x'],error_y=error['y'],error_z=error['z'],error_yaw_deg=error['yaw']))
        meta=step['vision_meta']
        self.timing_writer.writerow(dict(**stamp,frame_i=packet['sequence'],fsm_state=state,
            camera_frame_number=packet['sequence'],inference_ran=0,pose_src='simulation',
            model_det_ok=int(step['det_ok']),pnp_ok='',result_fps=packet['result_fps'],
            result_interval_s=packet['result_interval_s'],
            inference_start_host_mono_ms=meta['inference_start_mono']*1000,
            inference_end_host_mono_ms=meta['inference_end_mono']*1000,
            model_inference_ms=diagnostic['latency']*1000,
            pose_result_host_mono_ms=meta['pose_result_mono']*1000,
            pos_x_m=pose[0],pos_z_m=pose[2],yaw_deg=step['yaw_smooth']))
        self.counts['model_results']+=1

    def state(self,frame,initial=False):
        state,command=frame['state'],frame['command']
        if initial or state!=self.last_state:
            self.emit('control_seq.jsonl',dict(**self.stamp(frame['t']),phase='state',
                previous_state=self.last_state,state=state,cmd=command,lines=frame['lines'],
                target=frame.get('target'),failure_reason=frame.get('failure_reason')))
            self.counts['transitions']+=1
        if initial or command!=self.last_command:
            self.emit('control_seq.jsonl',dict(**self.stamp(frame['t']),phase='cmd',cmd=command,state=state))
        self.last_state,self.last_command=state,command

    def tick(self,row,frame):
        self.state(frame)
        self.relative_pose(frame,'tick')
        self.emit('fsm_steps.jsonl',dict(**self.stamp(row['t']),**row,telemetry=frame))
        self.counts['fsm_steps']+=1
        self.flush()

    def relative_pose(self,frame,phase):
        """Current ground truth, independent of delayed/noisy model output.

        World vehicle position is the pivot; pallet position is front centre.
        Pallet-local +Z points into the pallet, +X along its front width.
        """
        truth=frame['truth'];g=self.geometry
        angle=math.radians(truth['heading']);c,s=math.cos(angle),math.sin(angle)
        px,pz=g['CAMERA_TO_ROT_CENTER_X_M'],g['CAMERA_TO_ROT_CENTER_Z_M']
        def world(x,z):
            dx,dz=x-px,z-pz
            return truth['world_x']+c*dx+s*dz,truth['world_z']-s*dx+c*dz
        camera=world(0,0);pallet=world(truth['x'],truth['z'])
        tip_z=g['CAMERA_TO_FORK_TIP_Z_M'];tip=world(0,tip_z)
        a=math.radians(truth['yaw']);pc,ps=math.cos(a),math.sin(a)
        def local(x):
            dx,dz=x-truth['x'],tip_z-truth['z']
            return pc*dx-ps*dz,ps*dx+pc*dz
        local_tip=local(0);left=local(-g['FORK_OUTER_SPAN_M']/2);right=local(g['FORK_OUTER_SPAN_M']/2)
        self.pose_writer.writerow(dict(**self.stamp(frame['t']),phase=phase,fsm_state=frame['state'],command=frame['command'],
            forklift_world_x_m=truth['world_x'],forklift_world_z_m=truth['world_z'],forklift_heading_deg=truth['heading'],
            camera_world_x_m=camera[0],camera_world_z_m=camera[1],pallet_world_x_m=pallet[0],pallet_world_z_m=pallet[1],
            pallet_heading_deg=(truth['heading']+truth['yaw']+360)%360-180,
            pallet_camera_x_m=truth['x'],pallet_camera_y_m=self.options['vertical_offset'],pallet_camera_z_m=truth['z'],
            pallet_relative_yaw_deg=truth['yaw'],pallet_pivot_x_m=truth['x']-px,pallet_pivot_z_m=truth['z']-pz,
            pivot_to_pallet_distance_m=math.hypot(truth['x']-px,truth['z']-pz),
            fork_tip_world_x_m=tip[0],fork_tip_world_z_m=tip[1],fork_tip_pallet_x_m=local_tip[0],fork_tip_pallet_z_m=local_tip[1],
            left_outer_tip_pallet_x_m=left[0],left_outer_tip_pallet_z_m=left[1],
            right_outer_tip_pallet_x_m=right[0],right_outer_tip_pallet_z_m=right[1],
            tip_insertion_depth_m=max(0.,local_tip[1]),
            insertion_remaining_m=max(0.,truth['z']-g['INSERT_CAMERA_Z_REMAINDER_M']),
            collision=int(bool(frame['collision'])),minimum_clearance_m=frame['clearance']))
        self.counts['relative_poses']+=1

    def finish(self,report):
        self.relative_pose(dict(t=report['elapsed'],truth=report['final'],state=report['state'],
            command=self.last_command,collision=report['collision'],clearance=report['minimum_clearance']),'final')
        self.emit('control_seq.jsonl',dict(**self.stamp(report['elapsed']),phase='lifecycle',
            state=report['state'],lifecycle=report['lifecycle'],controller_error=report['controller_error']))
        self.flush()
        summary={k:v for k,v in report.items() if k not in ('frames','trace','can_frames')}
        summary['logging']={**summary['logging'],'status':'closed','counts':dict(self.counts)}
        self.write_json('run_summary.json',dict(**summary,log_counts=self.counts,ended_utc=datetime.now(timezone.utc).isoformat()))
        self.close()

    def flush(self):
        for handle in self.handles.values():
            handle.flush()

    def close(self):
        self.closed=True
        for handle in self.handles.values():
            try:
                handle.close()
            except OSError:
                pass
