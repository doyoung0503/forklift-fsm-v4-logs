"""Render logged or live world telemetry. No FSM execution or random sampling."""
import math
import time

import cv2
import numpy as np

from camera_preview import PalletCamera, label
from runtime_camera_overlay import draw_packet_overlay
from protocol_world import PROFILE


def world_geometry(truth, config):
    a = math.radians(truth['heading']); c, s = math.cos(a), math.sin(a)
    rotation = np.array([[c, s], [-s, c]])
    pivot = np.array([truth['world_x'], truth['world_z']])
    offset = np.array([config.get('CAMERA_TO_ROT_CENTER_X_M', 0),
                       config.get('CAMERA_TO_ROT_CENTER_Z_M', -.68)])
    camera = pivot - rotation @ offset
    pallet = camera + rotation @ np.array([truth['x'], truth['z']])
    return dict(pivot=pivot, camera=camera, pallet=pallet, rotation=rotation,
                pallet_angle=truth['heading']+truth['yaw'])


class SimulationCamera:
    def __init__(self):
        self.camera = PalletCamera()

    def render(self, frame, options, config=None):
        config = config or PROFILE['parameters']
        g = world_geometry(frame['truth'], config)
        a = math.radians(g['pallet_angle']); c, s = math.cos(a), math.sin(a)
        transform = np.eye(4)
        transform[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
        transform[:3,3] = np.array([g['pallet'][0],0,g['pallet'][1]]) - transform[:3,:3] @ [0,0,2]
        self.camera.fx = 320/math.tan(math.radians(options.get('camera_hfov_deg',55)/2))
        start = time.perf_counter()
        raw = self.camera.render(g['camera'][0], .075+options.get('vertical_offset',0),
                                 g['camera'][1],frame['truth']['heading'],pallet_transform=transform)
        image = raw.copy()
        packet = frame.get('model_packet')
        blind = frame.get('vision_independent',frame.get('input_mode')=='clock')
        if packet is not None:
            draw_packet_overlay(image, packet, options, vision_independent=blind,
                                skip_detection=frame.get('skip_detection',False))
        else:
            label(image,'NO MODEL PACKET IN LOG',12,28,(80,120,255))
        return image, dict(t=frame['t'],model_sequence=packet['sequence'] if packet else None,
                           vision_independent=bool(blind),render_ms=(time.perf_counter()-start)*1000,
                           camera_world=g['camera'].tolist(),pallet_world=g['pallet'].tolist(),
                           gpu=self.camera.ctx.info['GL_RENDERER'],inference_executed=False)


def top_view(frame, options, config, trail=()):
    im=np.full((480,640,3), (243,246,248), np.uint8)
    g=world_geometry(frame['truth'],config)
    # Fixed six-metre square per run, equal scale on both axes.
    centre=np.array([options.get('pallet_x',0),options.get('pallet_z',2)-1.5])
    def P(p):
        q=(np.asarray(p)-centre)*72
        return (int(round(320+q[0])),int(round(240-q[1])))
    for i in range(-10,11):
        cv2.line(im,P([i,centre[1]-3.3]),P([i,centre[1]+3.3]),(219,225,230),1)
        cv2.line(im,P([centre[0]-4.4,i]),P([centre[0]+4.4,i]),(219,225,230),1)
        label(im,f'{i}',P([i,centre[1]-2.9])[0],463,(110,115,125),.35)
        label(im,f'{i}',8,P([centre[0]-4,i])[1],(110,115,125),.35)
    def poly(points,fill,edge):
        pts=np.array([P(p) for p in points],np.int32)
        cv2.fillPoly(im,[pts],fill);cv2.polylines(im,[pts],True,edge,2,cv2.LINE_AA)
    camera=g['camera'];R=g['rotation']
    half=math.radians(options.get('camera_hfov_deg',55)/2)
    rays=[camera,camera+R@[-4*math.tan(half),4],camera+R@[4*math.tan(half),4]]
    layer=im.copy();poly(rays,(245,219,194),(231,180,120));im=cv2.addWeighted(im,.32,layer,.68,0)
    a=math.radians(g['pallet_angle']);Rp=np.array([[math.cos(a),math.sin(a)],[-math.sin(a),math.cos(a)]])
    w=config.get('PALLET_FRONT_VISIBILITY_WIDTH_M',1.1)/2;d=options.get('pallet_depth',1.1)
    pp=lambda x,z:g['pallet']+Rp@[x,z]
    poly([pp(-w,0),pp(w,0),pp(w,d),pp(-w,d)],(97,151,111),(55,94,57))
    cv2.line(im,P(pp(-w,0)),P(pp(w,0)),(155,220,15),4)
    if len(trail)>1:
        cv2.polylines(im,[np.asarray([P(p) for p in trail],np.int32)],False,(185,100,40),2)
    vp=lambda x,z:camera+R@[x,z]
    poly([vp(-.45,-1.6),vp(.45,-1.6),vp(.45,-.12),vp(-.45,-.12)],(207,215,224),(98,107,117))
    span=config.get('FORK_OUTER_SPAN_M',.6)/2;fw=options.get('fork_width',.1)
    tip=config.get('CAMERA_TO_FORK_TIP_Z_M',1.18)
    for l,r in [(-span,-span+fw),(span-fw,span)]:
        poly([vp(l,0),vp(r,0),vp(r,tip),vp(l,tip)],(35,55,230) if frame.get('collision') else (35,190,235),(70,115,155))
    cv2.circle(im,P(g['pivot']),5,(40,40,40),-1)
    cv2.circle(im,P(camera),5,(240,115,35),-1)
    cv2.arrowedLine(im,P(camera),P(vp(0,.45)),(225,105,30),2,tipLength=.3)
    label(im,'X right / Z up | 1 m grid',15,22,(80,85,95),.45)
    label(im,'Blue: camera | black: rotation centre | yellow: forks',15,43,(80,85,95),.4)
    return im


def composite(frame, camera_image, options, config, trail, name, speed=1):
    im=np.full((720,1280,3),(25,23,20),np.uint8)
    im[80:560,:640]=top_view(frame,options,config,trail)
    im[80:560,640:]=camera_image
    label(im,f'{name} | ACTUAL FSM / VIRTUAL CAN / SYNTHETIC VISION',20,30,scale=.65)
    label(im,f'WORLD X-Z | t = {frame["t"]:.3f} s',20,63)
    label(im,f'CAMERA RGB + LIVE OVERLAY | t = {frame["t"]:.3f} s',660,63)
    label(im,f'{frame["state"]} / {frame["command"]}',20,593,scale=.65)
    t=frame['truth'];obs=frame['observation']
    label(im,f'Relative X {t["x"]:+.3f} m / Z {t["z"]:.3f} m / yaw {t["yaw"]:+.2f} deg',20,624)
    packet=frame.get('model_packet');seq=packet['sequence'] if packet else '?'
    blind=frame.get('vision_independent',frame.get('input_mode')=='clock')
    label(im,f'Model #{seq} | {"VISION OFF" if blind else "DETECTED" if obs.get("detected") else "MISS"}',660,593)
    label(im,f'CAN frames {frame["can_frame_count"]} | {"COLLISION" if frame.get("collision") else "FORK CLEAR"}',660,624)
    target=frame.get('target') or {}
    label(im,str(target.get('description') or '')[:90],20,654,scale=.45)
    label(im,f'Camera height {(.075+options.get("vertical_offset",0)):.2f} m | Replay {speed:g}x | RGB + labels from the same log | no neural inference',20,686,scale=.46)
    if frame.get('terminal'):
        reason=frame.get('failure_reason') or frame.get('lifecycle',{}).get('reason') or frame.get('lifecycle',{}).get('outcome','')
        label(im,str(reason)[:80],660,654,(100,150,255),.43)
    return im
