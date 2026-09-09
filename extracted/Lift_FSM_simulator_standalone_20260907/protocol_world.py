"""Independent simulated vehicle + model protocol. No FSM/calib imports."""
from __future__ import annotations
import hashlib,json,math,random
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from model_result_model import GaussianResultModel

HERE=Path(__file__).resolve().parent
EPOCH=1000.0
PROFILE=json.loads((HERE/"vehicle_profile.json").read_text(encoding="utf-8"))
SPEC=SimpleNamespace(**PROFILE["parameters"])
MODEL_PROTOCOL="lift.model.v1"
CAN_PROTOCOL="lift.can.v1"


def can_frame(can_id, data, t=0., extended=False, remote=False):
    if type(can_id) is not int or not 0 <= can_id <= (0x1fffffff if extended else 0x7ff):
        raise ValueError("Invalid CAN identifier")
    if not isinstance(data,(list,tuple,bytes)) or len(data)>8 or any(type(v) is not int or not 0<=v<=255 for v in data):
        raise ValueError("CAN data must contain 0..8 integer bytes")
    if not math.isfinite(float(t)):
        raise ValueError("Invalid CAN timestamp")
    return dict(protocol=CAN_PROTOCOL, t=float(t), id=can_id, data=list(data),
                dlc=len(data), extended=bool(extended), remote=bool(remote))

DEFAULTS = dict(x=.0, z=2.0, yaw=0., hz=30., max_seconds=185., seed=20260907,
                perception="fov", min_visible_fraction=.5,
                latency=.12, dropout=0., loss_start=0., loss_duration=0.,
                yaw_noise=0., position_noise=0., yaw_bias=0., x_bias=0., z_bias=0.,
                drive_scale=1., rotation_scale=1., drive_delay=0.,
                imu_scale=1., imu_bias_deg_s=0., imu_noise_deg_s=0., imu_angle_sd_deg=5.,
                drive_coast_sec=0., rotation_coast_fraction=0., rotation_coast_sec=.6,
                drive_scale_sd=0.0607956831911769, rotation_scale_sd=0.0607956831911769, vertical_offset=0.,
                fork_width=.1, pallet_depth=1.1, opening="continuous", completion_tolerance=.05,
                face_selection="largest",x_noise=0.,y_noise=0.,z_noise=0.,y_bias=0.,
                model_hz=30.,model_interval_sd=0.,latency_sd=0.,camera_hfov_deg=55.,
                placement_mode="relative",pallet_x=0.,pallet_z=2.,pallet_heading=180.,
                forklift_x=0.,forklift_z=0.,forklift_heading=0.,inference_mode="continuous")


def scenario_options(options):
    # Saved runs may still contain the retired range setting. It no longer
    # participates in visibility, and new normalized options omit it.
    options={key:value for key,value in options.items() if key!='camera_range'}
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown scenario settings: {sorted(unknown)}")
    out = {**DEFAULTS, **options}
    for key, default in DEFAULTS.items():
        if isinstance(default, (int, float)):
            out[key] = float(out[key])
            if not math.isfinite(out[key]):
                raise ValueError(f"{key} must be finite")
    bounds = dict(x=(-10, 10), z=(.05, 15), yaw=(-180, 180), hz=(5, 120),
                  max_seconds=(1, 600), seed=(0, 2**32-1),
                  min_visible_fraction=(0, 1), latency=(0, 3), dropout=(0, 1),
                  loss_start=(0, 600), loss_duration=(0, 600), yaw_noise=(0, 30),
                  position_noise=(0, 1), yaw_bias=(-90, 90), x_bias=(-2, 2), z_bias=(-2, 2),
                  drive_scale=(0, 3), rotation_scale=(0, 3), drive_delay=(0, 5),
                  imu_scale=(.5, 1.5), imu_bias_deg_s=(-5, 5), imu_noise_deg_s=(0, 5), imu_angle_sd_deg=(0, 90),
                  drive_coast_sec=(0, 3), rotation_coast_fraction=(0, 1),
                  rotation_coast_sec=(.01, 3), drive_scale_sd=(0, 1),
                  rotation_scale_sd=(0, 1), vertical_offset=(-3, 3),
                  fork_width=(.01, .29), pallet_depth=(.1, 3), completion_tolerance=(0, .5),
                  x_noise=(0,2),y_noise=(0,2),z_noise=(0,2),y_bias=(-2,2),
                  model_hz=(1,120),model_interval_sd=(0,1),latency_sd=(0,1),camera_hfov_deg=(10,150),
                  pallet_x=(-100,100),pallet_z=(-100,100),pallet_heading=(-180,180),
                  forklift_x=(-100,100),forklift_z=(-100,100),forklift_heading=(-180,180))
    for key, (lo, hi) in bounds.items():
        if not lo <= out[key] <= hi:
            raise ValueError(f"{key} must be in [{lo}, {hi}]")
    # Retired biased gain/rate settings cannot reintroduce nonzero means via saved presets.
    out.update(drive_scale=1.,rotation_scale=1.,imu_scale=1.,imu_bias_deg_s=0.,imu_noise_deg_s=0.)
    out["seed"] = int(out["seed"])
    if out["perception"] not in ("fov", "oracle") or out["opening"] not in ("continuous", "split"):
        raise ValueError("Invalid perception or opening model")
    if out["face_selection"] not in ("largest","fixed"):
        raise ValueError("Invalid face selection")
    if out["placement_mode"] not in ("relative","world"):
        raise ValueError("Invalid placement mode")
    if out["inference_mode"] not in ("continuous", "requested"):
        raise ValueError("Invalid inference mode")
    return out


def wrap(deg):
    return (deg + 180) % 360 - 180


def clip(poly, axis, boundary, greater):
    result = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        ai = a[axis] >= boundary if greater else a[axis] <= boundary
        bi = b[axis] >= boundary if greater else b[axis] <= boundary
        if ai:
            result.append(a)
        if ai != bi:
            f = (boundary - a[axis]) / (b[axis] - a[axis])
            result.append(tuple(a[j] + f * (b[j] - a[j]) for j in (0, 1)))
    return result


class Plant:
    """Independent planar vehicle, pinhole visibility and fork/slot geometry.

    Rotation endpoint regression does not specify the intra-command path.
    The default is delayed constant rate; an optional fraction is deferred
    to post-STOP coast while preserving the commanded endpoint total.
    """
    def __init__(self, options):
        self.o = options
        self.x, self.z, self.yaw = options["x"], options["z"], options["yaw"]
        if options["placement_mode"]=="world":
            angle=math.radians(options["forklift_heading"])
            c,s=math.cos(angle),math.sin(angle)
            dx=options["pallet_x"]-options["forklift_x"]
            dz=options["pallet_z"]-options["forklift_z"]
            # Vehicle position is its rotation centre; model pose is camera-relative.
            self.x=c*dx-s*dz+SPEC.CAMERA_TO_ROT_CENTER_X_M
            self.z=s*dx+c*dz+SPEC.CAMERA_TO_ROT_CENTER_Z_M
            self.yaw=wrap(options["pallet_heading"]-180-options["forklift_heading"])
        self.steer, self.drive, self.elapsed, self.coasts = 0., 0., 0., []
        self.movement=[127]*8
        self.last_hb=self.last_mov=self.last_ctrl=-1e9
        self.driving_mode=False
        self.emergency=False
        self.bus_status="awaiting CAN"
        self.received=0
        self.random = random.Random(options["seed"])
        self.drive_gain = self.rotation_gain = 1.
        self.heading=options["forklift_heading"]
        self.world_x,self.world_z=options["forklift_x"],options["forklift_z"]
        # Legacy report fields remain null; collision acceptance belongs to FSM.
        self.collision = None
        self.minimum_clearance = None

    def drive_distance(self, sec):
        t = max(0., sec - SPEC.FWD_T0_SEC - self.o["drive_delay"])
        a = min(t, SPEC.FWD_ACCEL_DURATION_SEC)
        raw = .5 * SPEC.FWD_ACCEL_M_S2 * a*a + max(0., t-a) * SPEC.FWD_ACCEL_M_S2*a
        return raw * self.drive_gain / SPEC.FWD_DISTANCE_SCALE

    def rotation_angle(self, sec):
        return SPEC.ROTATION_SLOPE_DEG_S * max(0., sec-SPEC.ROTATION_DELAY_SEC) * self.rotation_gain

    def receive_can(self, frame):
        """The ONLY control input. No state names or controller targets accepted."""
        self.received+=1
        if frame["extended"] or frame["remote"]:
            return
        data=frame["data"]
        if frame["id"]==0x764 and data==[0]:
            self.last_hb=frame["t"]
        elif frame["id"]==0x2e3 and len(data)==8:
            self.last_ctrl=frame["t"]
            self.emergency=bool(data[0]&0x80)
            self.driving_mode=data[0]==0x42 and data[3]==0x0a
        elif frame["id"]==0x1e3 and len(data)==8:
            self.last_mov=frame["t"]
            self.movement=list(data)
        self.update_bus(frame["t"])

    def update_bus(self, now):
        if self.emergency:
            reason="emergency"
        elif now-self.last_hb>SPEC.HEARTBEAT_TIMEOUT_SEC:
            reason="heartbeat timeout"
        elif now-self.last_ctrl>SPEC.CONTROL_TIMEOUT_SEC:
            reason="control timeout"
        elif not self.driving_mode:
            reason="non-driving control mode"
        elif now-self.last_mov>SPEC.MOVEMENT_TIMEOUT_SEC:
            reason="movement timeout"
        else:
            reason="ready"
        self.bus_status=reason
        self._set_axes(127-self.movement[1] if reason=="ready" else 0,
                       127-self.movement[2] if reason=="ready" else 0)

    def _set_axes(self, steer, drive):
        if (steer,drive)==(self.steer,self.drive):
            return
        if self.steer:
            coast = self.rotation_angle(self.elapsed) * self.o["rotation_coast_fraction"]
            self.coasts.append(["turn", coast * math.copysign(1,self.steer),
                                self.o["rotation_coast_sec"], 0.])
        if self.drive:
            dt = .0001
            speed = (self.drive_distance(self.elapsed+dt)-self.drive_distance(self.elapsed))/dt
            duration = self.o["drive_coast_sec"]
            self.coasts.append(["drive", speed * duration / 2 * math.copysign(1,self.drive),
                                max(.0001, duration), 0.])
        self.steer,self.drive,self.elapsed=steer,drive,0.
        if steer:
            self.rotation_gain = abs(steer)/SPEC.ROTATE_REFERENCE_DEFLECTION * (1. + self.random.gauss(0., self.o["rotation_scale_sd"]))
        if drive:
            self.drive_gain = abs(drive)/SPEC.DRIVE_REFERENCE_DEFLECTION * (1. + self.random.gauss(0., self.o["drive_scale_sd"]))

    def move(self, turn, forward):
        a = math.radians(turn)
        c, s = math.cos(a), math.sin(a)
        dx, dz = self.x-SPEC.CAMERA_TO_ROT_CENTER_X_M, self.z-SPEC.CAMERA_TO_ROT_CENTER_Z_M
        self.x = c*dx-s*dz+SPEC.CAMERA_TO_ROT_CENTER_X_M
        self.z = s*dx+c*dz+SPEC.CAMERA_TO_ROT_CENTER_Z_M-forward
        self.yaw = wrap(self.yaw-turn)
        self.heading += turn
        self.world_x += math.sin(math.radians(self.heading))*forward
        self.world_z += math.cos(math.radians(self.heading))*forward

    def advance(self, dt, now):
        # Preserve integration cadence for bus timing and combined motion/coast.
        steps = max(1, math.ceil(dt/.005))
        ds = dt/steps
        for i in range(steps):
            self.update_bus(now+i*ds)
            t0, t1 = self.elapsed, self.elapsed+ds
            turn = forward = 0.
            if self.steer:
                turn = (self.rotation_angle(t1)-self.rotation_angle(t0)) * (1-self.o["rotation_coast_fraction"])
                turn *= math.copysign(1,self.steer)
            if self.drive:
                forward = (self.drive_distance(t1)-self.drive_distance(t0)) * math.copysign(1,self.drive)
            for coast in self.coasts:
                kind, total, duration, elapsed = coast
                f0, f1 = min(1., elapsed/duration), min(1., (elapsed+ds)/duration)
                delta = total*((2*f1-f1*f1)-(2*f0-f0*f0))
                if kind == "turn":
                    turn += delta
                else:
                    forward += delta
                coast[3] += ds
            self.coasts = [coast for coast in self.coasts if coast[3] < coast[2]]
            self.move(turn, forward)
            self.elapsed = t1

    def truth(self):
        return dict(x=self.x, z=self.z, yaw=self.yaw, world_x=self.world_x,
                    world_z=self.world_z, heading=self.heading)


def projected_faces(x,z,yaw,y0,depth,fx):
    """Pinhole equivalent of the launcher's largest vertical cuboid-face selection."""
    a=math.radians(yaw);c,s=math.cos(a),math.sin(a)
    half=SPEC.PALLET_FRONT_VISIBILITY_WIDTH_M/2
    transform=lambda q,h,d:[x+c*q+s*d,y0+h,z-s*q+c*d]
    points=[transform(q,h,d) for d in (0,depth)
            for q,h in ((-half,-.075),(half,-.075),(half,.075),(-half,.075))]
    project=lambda p:[fx*p[0]/p[2]+320,fx*p[1]/p[2]+240] if p[2]>.001 else None
    pixels=[project(p) for p in points]
    specs=[("z_min",[0,1,2,3],(0,0),0.,2*half),
           ("z_max",[4,5,6,7],(0,depth),180.,2*half),
           ("x_min",[0,3,7,4],(-half,depth/2),90.,depth),
           ("x_max",[1,5,6,2],(half,depth/2),-90.,depth)]
    faces=[]
    for name,indices,(q,d),turn,width in specs:
        center=transform(q,0,d)
        selected_yaw=wrap(yaw+turn)
        normal=math.radians(selected_yaw)
        facing=center[0]*math.sin(normal)+center[2]*math.cos(normal)>0
        poly=[pixels[i] for i in indices]
        area=abs(sum(p[0]*r[1]-r[0]*p[1] for p,r in zip(poly,poly[1:]+poly[:1])))/2 if all(p is not None for p in poly) else 0.
        faces.append(dict(name=name,corners=[points[i] for i in indices],pixels=poly,
                          area=area,facing=facing,center=center,yaw=selected_yaw,width=width))
    positive=[p for p in pixels if p is not None]
    margin=min(min(p[0]/640,1-p[0]/640,p[1]/480,1-p[1]/480) for p in positive) if len(positive)==8 else -1.
    return faces,margin,transform(0,0,depth/2)


def observation(truth, measured, now, sequence, options, sample):
    fx=320/math.tan(math.radians(options["camera_hfov_deg"]/2))
    faces,_margin,body=projected_faces(truth["x"],truth["z"],truth["yaw"],options["vertical_offset"],options["pallet_depth"],fx)
    candidates=[f for f in faces if f["facing"] and f["area"]>0] or [f for f in faces if f["area"]>0]
    face=max(candidates,key=lambda f:f["area"]) if candidates else faces[0]
    if options["face_selection"]=="fixed":
        face=faces[0]
    true_poly=face["pixels"]
    if all(p is not None for p in true_poly):
        u0,u1=min(p[0] for p in true_poly),max(p[0] for p in true_poly)
        v0,v1=min(p[1] for p in true_poly),max(p[1] for p in true_poly)
        fraction=(max(0.,min(640.,u1)-max(0.,u0))/(u1-u0) *
                  max(0.,min(480.,v1)-max(0.,v0))/(v1-v0)) if u1>u0 and v1>v0 else 0.
    else:
        fraction=0.
    true_x,true_y,true_z=face["center"]
    error=sample["error"]
    px,py,pz=true_x+error["x"],true_y+error["y"],true_z+error["z"]
    yaw=wrap(face["yaw"]+error["yaw"])
    observed_faces,margin,observed_body=projected_faces(px,pz,yaw,py,options["pallet_depth"],fx)
    observed_face=observed_faces[0]
    detected=fraction>=options["min_visible_fraction"] and fraction>0 and face["facing"] and true_z>0
    if options["perception"]=="oracle":
        detected=true_z>0
    if sample["missed"] or options["loss_start"]<=measured<options["loss_start"]+options["loss_duration"]:
        detected=False
    face_bearing=math.degrees(math.atan2(px,pz))
    model_bearing=math.degrees(math.atan2(observed_body[0],observed_body[2]))
    latency=max(0.,now-measured)
    meta=dict(center_bearing_deg=face_bearing if detected else None,
              front_face_center_bearing_deg=face_bearing if detected else None,
              model_center_bearing_deg=model_bearing if detected else None,
              bbox_margin_norm=margin if detected else None,
              frame_width=640,frame_height=480,fx=fx,fy=fx,ppx=320.,ppy=240.,
              face_corners_px=observed_face["pixels"] if detected else None,
              face_corners_camera_m=observed_face["corners"] if detected else None,
              selected_front_face=face["name"] if detected else None,
              selected_face_area_px2=observed_face["area"] if detected else None,
              projected_face_areas_px2={f["name"]:f["area"] for f in observed_faces} if detected else None,
              yaw_raw_deg=yaw if detected else None,
              pos_x_m=px if detected else None,pos_z_m=pz if detected else None,
              measurement_mono=EPOCH+measured,fsm_step_mono=EPOCH+now,
              inference_latency_sec=latency,inference_start_mono=EPOCH+measured,
              inference_end_mono=EPOCH+now,model_inference_latency_sec=latency,
              pose_result_mono=EPOCH+now,camera_frame_number=sequence,
              sensor_timestamp_ms=measured*1000,sensor_timestamp_domain="simulation_clock")
    step=dict(det_ok=detected,detected_length=face["width"] if detected else None,
              dist_z=pz if detected else None,yaw_smooth=yaw if detected else None,
              offset_smooth=[px,py,pz] if detected else None,
              target_bearing_deg=model_bearing if detected else None,vision_meta=meta)
    packet=dict(protocol=MODEL_PROTOCOL,sequence=sequence,sim_time_s=now,
                clock=dict(domain="simulation_monotonic",now=EPOCH+now),step=step,
                result_interval_s=sample["interval"],result_fps=1/sample["interval"])
    diagnostic=dict(fraction=fraction,margin=margin,detected=detected,x=px,y=py,z=pz,
                    yaw=yaw,measured=measured,selected_face=face["name"],
                    sampled_error=error,ground_truth_pose=dict(x=true_x,y=true_y,z=true_z,yaw=face["yaw"]),
                    sequence=sequence,result_fps=packet["result_fps"],latency=latency)
    return packet,diagnostic


def model_step(packet):
    if packet.get("protocol")!=MODEL_PROTOCOL:
        raise ValueError("Unsupported model protocol")
    step=packet["step"]
    required={"det_ok","detected_length","dist_z","yaw_smooth","offset_smooth","target_bearing_deg","vision_meta"}
    if set(step)!=required:
        raise ValueError("Model packet has missing or unknown step fields")
    if type(step["det_ok"]) is not bool:
        raise ValueError("det_ok must be boolean")
    if step["offset_smooth"] is not None and len(step["offset_smooth"])!=3:
        raise ValueError("offset_smooth must be null or [x,y,z]")
    return step


class World:
    """A controller-independent environment. Inputs: CAN frames. Output: model packet."""
    def __init__(self, options=None):
        self.options=scenario_options(options or {})
        self.plant=Plant(self.options)
        self.now=0.
        self.sequence=0
        self.result_model=GaussianResultModel(self.options)
        self.history=[(0.,self.plant.truth())]
        self.last_packet=None
        self.last_diagnostic={}
        self.can_log=[]
        self.next_model_time=0.
        self.model_count=0
        self.model_observer=None

    def observe(self):
        if self.last_packet is None:
            self._emit_model()
        return self.last_packet

    def _emit_model(self):
        sample=self.result_model.sample()
        wanted=self.now-sample["latency"]
        # Retain history for variable-latency results, including unusually late
        # samples. A later output can refer to an earlier camera capture.
        index=0
        for i in range(len(self.history)-1,-1,-1):
            if self.history[i][0]<=wanted:
                index=i
                break
        measured,truth=self.history[index]
        if index+1<len(self.history) and measured<=wanted:
            after,other=self.history[index+1]
            f=max(0.,min(1.,(wanted-measured)/max(1e-12,after-measured)))
            truth={k:truth[k]+(wrap(other[k]-truth[k]) if k=="yaw" else other[k]-truth[k])*f for k in truth}
            measured=wanted
        if wanted<0:
            measured=wanted
        self.sequence=self.model_count
        interval=None if self.last_packet is None else self.now-self.last_packet["sim_time_s"]
        self.last_packet,self.last_diagnostic=observation(truth,measured,self.now,self.sequence,self.options,sample)
        self.last_packet["result_interval_s"]=interval
        self.last_packet["result_fps"]=None if not interval else 1/interval
        self.last_diagnostic["result_fps"]=self.last_packet["result_fps"]
        self.model_count+=1
        self.next_model_time=self.now+sample["interval"]
        if self.model_observer is not None:
            self.model_observer(self.last_packet,self.last_diagnostic)

    def _integrate_to(self,t):
        self.plant.advance(max(0.,t-self.now),self.now)
        if t>self.now+1e-12:
            self.now=t
            self.history.append((self.now,self.plant.truth()))

    def advance(self, dt, frames):
        self.observe()
        if not math.isfinite(float(dt)) or not 0<=dt<=1:
            raise ValueError("dt must be in [0,1] seconds")
        if not isinstance(frames,list) or len(frames)>1000:
            raise ValueError("CAN frames must be a list of at most 1000 frames")
        end=self.now+float(dt)
        valid=[]
        previous=self.now
        for f in frames:
            if f.get("protocol")!=CAN_PROTOCOL:
                raise ValueError("Unsupported CAN protocol")
            frame=can_frame(f["id"],f["data"],f.get("t",self.now),f.get("extended",False),f.get("remote",False))
            if f.get("dlc",frame["dlc"])!=frame["dlc"] or not previous-1e-8<=frame["t"]<=end+1e-8:
                raise ValueError("DLC mismatch or unordered/out-of-interval CAN timestamp")
            valid.append(frame)
            previous=frame["t"]
        # Validate the entire transaction before mutating the vehicle.
        cursor=0
        while True:
            next_can=valid[cursor]["t"] if cursor<len(valid) else float("inf")
            at=min(next_can,self.next_model_time,end)
            self._integrate_to(max(self.now,at))
            while cursor<len(valid) and valid[cursor]["t"]<=self.now+1e-9:
                f=valid[cursor]
                self.plant.receive_can(f)
                self.can_log.append(f)
                cursor+=1
            if self.next_model_time<=self.now+1e-9:
                self._emit_model()
            if self.now>=end-1e-10:
                break
        self.now=end
        return self.observe()

    def diagnostics(self):
        return dict(sim_time_s=self.now,next_model_time_s=self.next_model_time,model_count=self.model_count,
                    options=self.options,
                    truth=self.plant.truth(),observation=self.last_diagnostic,model_packet=self.last_packet,
                    collision=self.plant.collision,clearance=self.plant.minimum_clearance,
                    can=dict(id=0x1e3,data=self.plant.movement,status=self.plant.bus_status,
                             received_frames=self.plant.received),vehicle_profile=PROFILE)


class RequestedWorld(World):
    """Capture GT on a controller request; publish only after its sampled delay."""
    def __init__(self, options=None):
        super().__init__(options)
        self.pending = None
        self.next_request_time = 0.
        self.next_model_time = float('inf')
        # A non-result placeholder keeps camera intrinsics available while waiting.
        empty = dict(error={k: 0. for k in ('x', 'y', 'z', 'yaw')},
                     interval=1/self.options['model_hz'], latency=0., missed=True)
        self.last_packet, self.last_diagnostic = observation(
            self.plant.truth(), 0., 0., -1, self.options, empty)
        self.last_packet['pending'] = True

    def request_model(self):
        if self.pending is not None or self.now < self.next_request_time-1e-9:
            return False
        sample = self.result_model.sample()
        self.pending = (self.now, dict(self.plant.truth()), sample)
        self.next_model_time = self.now + sample['latency']
        self.next_request_time = self.now + sample['interval']
        if self.next_model_time <= self.now:
            self._emit_model()
        return True

    def _emit_model(self):
        captured, truth, sample = self.pending
        previous = None if self.model_count == 0 else self.last_packet['sim_time_s']
        self.last_packet, self.last_diagnostic = observation(
            truth, captured, self.now, self.model_count, self.options, sample)
        interval = None if previous is None else self.now-previous
        self.last_packet.update(request_time_s=captured, result_interval_s=interval,
                                result_fps=1/interval if interval else None)
        self.last_diagnostic['result_fps'] = self.last_packet['result_fps']
        self.model_count += 1
        self.pending = None
        self.next_model_time = float('inf')
        if self.model_observer is not None:
            self.model_observer(self.last_packet, self.last_diagnostic)


