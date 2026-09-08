"""Deterministic OBJ camera preview + Gaussian synthetic labels; no inference/CAN.

pip install moderngl trimesh imageio-ffmpeg
python camera_preview.py --seconds 18 --height .65
"""
from pathlib import Path
import argparse
import json
import math
import time
import zipfile

import cv2
import moderngl
import numpy as np
import trimesh
import imageio_ffmpeg
from protocol_world import scenario_options, observation
from model_result_model import GaussianResultModel
from runtime_camera_overlay import draw_packet_overlay

ROOT = Path(__file__).resolve().parent


class PalletCamera:
    """Y-up world, +Z optical forward; pallet front centre is (0,.075,2).

    Original OBJ scale and height retained; X centre and front Z normalized.
    render_relative matches protocol_world truth.x/z/yaw (camera-relative).
    """
    def __init__(self, hfov=55):
        asset = ROOT / 'assets/green_pallet'
        asset.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ROOT / 'green_pallet_obj.zip') as archive:
            for name in ('green_pallet_repaired.obj', 'green_pallet_repaired.mtl',
                         'textures/tex_u0_v0_diffuse.jpg', 'export_report.json'):
                if not (asset / name).exists():
                    archive.extract(name, asset)
        scene = trimesh.load(asset / 'green_pallet_repaired.obj', force='scene', process=False)
        self.bounds = scene.bounds.tolist()
        self.ctx = moderngl.create_standalone_context()
        self.fx = 320 / math.tan(math.radians(hfov / 2))
        self.fbo = self.ctx.simple_framebuffer((640, 480), components=3, samples=4)
        self.resolved = self.ctx.simple_framebuffer((640, 480), components=3)
        self.program = self.ctx.program(vertex_shader='''#version 330
            uniform mat4 mvp; in vec3 position; in vec3 normal; in vec2 uv;
            out vec3 n; out vec2 texcoord; out vec3 world;
            void main(){gl_Position=mvp*vec4(position,1); n=normal; texcoord=uv; world=position;}
        ''', fragment_shader='''#version 330
            uniform sampler2D tex; uniform vec3 color; uniform int textured;
            in vec3 n; in vec2 texcoord; in vec3 world; out vec4 frag;
            void main(){
                vec3 base=color;
                if(textured==1) base=texture(tex,texcoord).rgb;
                if(textured==2){vec2 g=abs(fract(world.xz+.5)-.5);
                    base=vec3(.64,.66,.67)*(min(g.x,g.y)<.009 ? .82:1.0);}
                float light=.62+.38*abs(dot(normalize(n),normalize(vec3(-.4,1,-.6))));
                frag=vec4(base*light,1);
            }
        ''')
        self.parts = []
        shift = np.array([(scene.bounds[0,0]+scene.bounds[1,0])/2, 0, scene.bounds[0,2]-2])
        for mesh in scene.geometry.values():
            mat = mesh.visual.material
            verts = mesh.vertices - shift
            texcoords = mesh.visual.uv
            data = np.column_stack((verts, mesh.vertex_normals,
                                    texcoords if texcoords is not None else np.zeros((len(verts),2))))
            texture = None
            if mat.image is not None and mat.name == 'default_tex0':
                im = mat.image.convert('RGB').transpose(1)
                texture = self.ctx.texture(im.size, 3, im.tobytes())
                texture.build_mipmaps()
            self._part(data, mesh.faces, tuple(np.array(mat.diffuse[:3])/255), 1 if texture else 0, texture)
        floor=np.array([[-20,-.002,-20,0,1,0,0,0],[20,-.002,-20,0,1,0,0,0],
                        [20,-.002,30,0,1,0,0,0],[-20,-.002,30,0,1,0,0,0]])
        self._part(floor, [[0,1,2],[0,2,3]], (.65,.65,.65), 2, None)

    def _part(self, data, faces, color, mode, tex):
        vao=self.ctx.vertex_array(self.program,[(self.ctx.buffer(np.asarray(data,'f4').tobytes()),
                 '3f 3f 2f','position','normal','uv')],self.ctx.buffer(np.asarray(faces,'i4').tobytes()))
        self.parts.append((vao,color,mode,tex))

    def render(self, x, y, z, heading=0, pitch=0, pallet_transform=None):
        a,p=map(math.radians,(heading,pitch))
        right=np.array([math.cos(a),0,-math.sin(a)])
        forward=np.array([math.sin(a)*math.cos(p),-math.sin(p),math.cos(a)*math.cos(p)])
        up=np.cross(forward,right)
        view=np.eye(4); view[:3,:3]=np.array([right,up,-forward]);view[:3,3]=-view[:3,:3]@np.array([x,y,z])
        near,far=.015,50
        proj=np.zeros((4,4));proj[0,0]=self.fx/320;proj[1,1]=self.fx/240
        proj[2,2]=-(far+near)/(far-near);proj[2,3]=-2*far*near/(far-near);proj[3,2]=-1
        self.program['mvp'].write(np.asarray((proj@view).T,'f4').tobytes())
        self.fbo.use();self.fbo.clear(.73,.78,.81,1,depth=1);self.ctx.enable(moderngl.DEPTH_TEST)
        for vao,color,mode,tex in self.parts:
            transform = pallet_transform if pallet_transform is not None and mode != 2 else np.eye(4)
            self.program['mvp'].write(np.asarray((proj@view@transform).T,'f4').tobytes())
            self.program['color'].value=color;self.program['textured'].value=mode
            if tex: tex.use()
            vao.render()
        self.ctx.copy_framebuffer(self.resolved,self.fbo)
        rgb=np.frombuffer(self.resolved.read(components=3,alignment=1),np.uint8).reshape(480,640,3)[::-1]
        return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)

    def render_relative(self, truth, vertical_offset=0):
        # y_down(front centre) = camera_height - .075; yaw = -camera heading.
        a=math.radians(-truth['yaw']);c,s=math.cos(a),math.sin(a)
        return self.render(-c*truth['x']-s*truth['z'], .075+vertical_offset,
                           2+s*truth['x']-c*truth['z'], -truth['yaw'])


def trajectory(t, duration, height):
    q=t/duration*3
    u=q%1; ease=(1-math.cos(math.pi*u))/2
    if q<1: return (0,height,-1+2*ease,0,0,'APPROACH: 3.0 m -> 1.0 m')
    if q<2: return (.55*math.sin(2*math.pi*u),height,1,0,0,'LATERAL: fixed camera heading')
    a=math.radians(50*math.sin(2*math.pi*u))
    return (1.55*math.sin(a),height,2.55-1.55*math.cos(a),-math.degrees(a),0,'ORBIT: camera turns toward pallet')


def relative_truth(pose):
    x,y,z,heading,*_=pose
    a=math.radians(heading);c,s=math.cos(a),math.sin(a)
    return dict(x=-c*x-s*(2-z),z=-s*x+c*(2-z),yaw=-heading)


def draw_face(im, corners, color, thickness=1, points=False):
    if corners is None or any(p is None for p in corners): return
    a=np.asarray(corners)
    if not np.isfinite(a).all() or np.abs(a).max()>1e6: return
    pts=np.rint(a).astype(np.int32)
    cv2.polylines(im,[pts],True,color,thickness,cv2.LINE_AA)
    if points:
        for j,(u,v) in enumerate(pts):
            if 0<=u<640 and 0<=v<480:
                cv2.circle(im,(int(u),int(v)),4,color,-1)
                label(im,str(j),int(u)+5,int(v)-5,color,.4)


def label(im,text,x,y,color=(220,230,240),scale=.55):
    cv2.putText(im,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=18)
    ap.add_argument('--fps',type=int,default=30);ap.add_argument('--height',type=float,default=.65)
    ap.add_argument('--options',type=Path,help='Existing simulator Gaussian options JSON')
    ap.add_argument('--show-truth',action='store_true',help='Optional cyan ground-truth diagnostic outline')
    ap.add_argument('--output',type=Path,default=ROOT/'camera_preview_output');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.seconds<=0 or args.fps<=0: ap.error('seconds and fps must be positive')
    camera=PalletCamera()
    overrides=json.loads(args.options.read_text(encoding='utf-8-sig')) if args.options else {}
    options=scenario_options(dict(x_noise=.01,y_noise=.01,z_noise=.02,yaw_noise=1,
        model_hz=10,latency=.12,dropout=.05))
    options.update(overrides)
    options['vertical_offset']=args.height-.075
    options=scenario_options(options)
    camera.fx=320/math.tan(math.radians(options['camera_hfov_deg']/2))
    sampler=GaussianResultModel(options)
    next_result=0.;sequence=0;packet=None;diagnostic=None
    zero_sample=dict(error=dict(x=0,y=0,z=0,yaw=0),interval=1/options['model_hz'],missed=False)
    writer=imageio_ffmpeg.write_frames(str(args.output/'pallet_camera_demo.mp4'),(1280,660),
                fps=args.fps,codec='libx264',quality=8,macro_block_size=2)
    raw_writer=imageio_ffmpeg.write_frames(str(args.output/'camera_raw.mp4'),(640,480),
                fps=args.fps,codec='libx264',quality=8,macro_block_size=2)
    writer.send(None);raw_writer.send(None);records=[]
    count=round(args.seconds*args.fps)
    try:
        for i in range(count):
            t=i/args.fps;x,y,z,heading,pitch,stage=trajectory(t,args.seconds,args.height)
            start=time.perf_counter();raw=camera.render(x,y,z,heading,pitch);render_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter()
            while next_result<=t+1e-9:
                sample=sampler.sample();measured=max(0,next_result-sample['latency'])
                captured=relative_truth(trajectory(measured,args.seconds,args.height))
                sequence+=1
                packet,diagnostic=observation(captured,measured,next_result,sequence,options,sample)
                next_result+=sample['interval']
            label_ms=(time.perf_counter()-start)*1000
            det=packet['step']['det_ok'];corners=packet['step']['vision_meta']['face_corners_px']
            truth=relative_truth((x,y,z,heading))
            gt,_=observation(truth,t,t,sequence,{**options,'dropout':0,'perception':'oracle','loss_duration':0},zero_sample)
            overlay=raw.copy()
            if args.show_truth:
                draw_face(overlay,gt['step']['vision_meta']['face_corners_px'],(255,220,30),1)
            draw_packet_overlay(overlay,packet,options)
            composed=np.full((660,1280,3),(25,23,20),np.uint8)
            label(composed,'SYNTHETIC CAMERA / FIXED 3D PALLET',20,30,scale=.7)
            label(composed,'RAW RGB - 640 x 480',20,63)
            label(composed,'LIVE FSM OVERLAY RULES / SYNTHETIC POSE',660,63)
            composed[80:560,:640]=raw;composed[80:560,640:]=overlay
            label(composed,f'{t:05.2f}s | {stage}',20,590)
            label(composed,f'Camera world XYZ: {x:+.2f}, {y:.2f}, {z:+.2f} m | heading {heading:+.1f} deg',20,620)
            label(composed,f'{"SYNTHETIC DETECT" if det else "SYNTHETIC MISS"} | #{sequence} | age {(t-diagnostic["measured"])*1000:.0f} ms',660,590,
                  (80,230,130) if det else (100,140,255))
            label(composed,f'Gaussian std XYZ: {options["x_noise"]:.2f}/{options["y_noise"]:.2f}/{options["z_noise"]:.2f} m, yaw {options["yaw_noise"]:.1f} deg',660,620,scale=.48)
            label(composed,f'NO NEURAL MODEL | {options["model_hz"]:g} Hz labels / {args.fps} fps RGB | dropout {options["dropout"]:.0%} | prescribed camera path, not FSM execution',20,648,scale=.45)
            writer.send(cv2.cvtColor(composed,cv2.COLOR_BGR2RGB));raw_writer.send(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB))
            records.append(dict(frame=i,t=t,camera_xyz=[x,y,z],heading=heading,pitch=pitch,stage=stage,
                detected=bool(det),truth=truth,packet=packet,diagnostic=diagnostic,
                render_ms=render_ms,label_ms=label_ms))
            if i in (count//6,count//2,5*count//6):
                cv2.imencode('.jpg',composed)[1].tofile(args.output/f'preview_{i:04d}.jpg')
            if i%args.fps==0: print(f'frame {i}/{count} synthetic_detect={det} render={render_ms:.1f}ms',flush=True)
    finally:
        writer.close();raw_writer.close()
    stats=dict(frames=count,fps=args.fps,duration=args.seconds,camera_height=args.height,hfov=options['camera_hfov_deg'],
        inference_executed=False,overlay='shared calib.camera_overlay',show_truth=args.show_truth,options=options,
        obj_bounds=camera.bounds,gpu=camera.ctx.info['GL_RENDERER'],detections=sum(r['detected'] for r in records),
        render_ms_median=float(np.median([r['render_ms'] for r in records])),
        label_ms_median=float(np.median([r['label_ms'] for r in records])),
        total_ms_p95=float(np.percentile([r['render_ms']+r['label_ms'] for r in records],95)))
    (args.output/'measurements.json').write_text(json.dumps(dict(summary=stats,frames=records),indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))


if __name__=='__main__': main()
