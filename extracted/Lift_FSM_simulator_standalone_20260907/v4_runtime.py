"""Run the workspace's unmodified Python FSM with a virtual clock and CAN channel.

Only the environment is simulated. No FSM subclass, skipped settle state,
synthetic target arrival, or replacement planner is used.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import sys
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
DEPTH_CAM = HERE.parent / "depth_cam"
sys.path.insert(0, str(DEPTH_CAM))
from calib import control, command_status
from calib.fsm_v4 import config as cfg, top
from protocol_world import World, Plant, SPEC, PROFILE, DEFAULTS, scenario_options, model_step
from current_can_adapter import CurrentCanTransport

control.configure_can_enabled(False)

LOCK = threading.RLock()
EPOCH = 1000.0
OVERRIDES = {
    "COARSE_IMU_ENABLED",
    "ROT_TX_TIMED_STOP_ENABLED",
    "FWD_PREDICTIVE_MAX_ADVANCE_M",
    "COARSE_LATERAL_BASE_M", "COARSE_LATERAL_GAIN",
    "FINAL_YAW_TOL_DEG", "FINAL_LATERAL_TOL_M", "FINAL_DISTANCE_TOL_M",
    "IMAGE_EDGE_MARGIN_NORM", "STABLE_POSE_FRAMES", "MAX_CORRECTION_CYCLES",
}
SOURCE_PATHS = sorted(set(
    list((DEPTH_CAM / "calib/fsm_v4").glob("*.py"))
    + list((DEPTH_CAM / "calib/fsm").glob("*.py"))
    + [DEPTH_CAM / "calib/config.py", DEPTH_CAM / "calib/control.py",
       DEPTH_CAM / "main_rec_v4.py", DEPTH_CAM / "simulation_runtime.py",
       DEPTH_CAM / "calib/command_status.py", Path(cfg.ROT_GENERATED_ARTIFACT_PATH)]
))
SOURCE_HASHES = {str(p.relative_to(DEPTH_CAM)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in SOURCE_PATHS}
SOURCE_ID = hashlib.sha256(json.dumps(SOURCE_HASHES, sort_keys=True).encode()).hexdigest()
SIMULATOR_PATHS = [HERE/name for name in ("v4_runtime.py", "serve_v4.py", "v4_sim.js",
                                        "protocol_world.py", "current_can_adapter.py", "vehicle_profile.json")]
SIMULATOR_PATHS.extend(HERE/name for name in ("model_result_model.py","serve_world.py","protocol_client.py","view_geometry.js"))
SIMULATOR_HASHES = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SIMULATOR_PATHS}
SIMULATOR_ID = hashlib.sha256(json.dumps(SIMULATOR_HASHES,sort_keys=True).encode()).hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def fingerprint():
    return {
        "source_id": SOURCE_ID, "files": SOURCE_HASHES,
        "simulator_id": SIMULATOR_ID, "simulator_files": SIMULATOR_HASHES,
        "disk_matches_loaded": all(hashlib.sha256(p.read_bytes()).hexdigest() ==
                                   SOURCE_HASHES[str(p.relative_to(DEPTH_CAM))]
                                   for p in SOURCE_PATHS),
        "controller": "calib.fsm_v4.top.CalibrationFSMV4",
        "config": {k: v for k, v in vars(cfg).items() if k.isupper()
                   and isinstance(v, (str, int, float, bool))},
        "rotation_response": clean(vars(cfg.ROTATION_RESPONSE)),
        "vehicle_profile": PROFILE,
        "protocols": {"model":"lift.model.v1","can":"lift.can.v1"},
        "grid_fields": sorted(OVERRIDES),
        "boundary": "CalibrationFSMV4 via model packets and virtual CAN writes; physical CAN OFF. "
                    "Independent world accepts only CAN frames. Camera/YOLO/PnP and vehicle physics simulated. main_rec_v4 keyboard "
                    "debug checkpoints and hydraulic initialisation are outside this boundary.",
    }


@contextlib.contextmanager
def environment(now, overrides=None):
    """Serialise process-global config/CAN state; never patch the real time module."""
    with LOCK:
        overrides = overrides or {}
        if set(overrides) - OVERRIDES:
            raise ValueError("Unsupported FSM override")
        old = {k: getattr(cfg, k) for k in overrides}
        old_top_time, old_status_time = top.time, command_status.time
        try:
            for key, value in overrides.items():
                if isinstance(old[key], bool):
                    if not isinstance(value, bool):
                        raise ValueError(f"{key} must be a boolean")
                    setattr(cfg, key, value)
                    continue
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Invalid {key}")
                if isinstance(old[key], int) and (isinstance(value, bool) or int(value) != value):
                    raise ValueError(f"{key} must be an integer")
                if key == "IMAGE_EDGE_MARGIN_NORM":
                    if not 0 <= value < .5:
                        raise ValueError("IMAGE_EDGE_MARGIN_NORM must be in [0, .5)")
                elif key == "FWD_PREDICTIVE_MAX_ADVANCE_M":
                    if value < 0:
                        raise ValueError(f"{key} must be nonnegative")
                elif value <= 0:
                    raise ValueError(f"{key} must be positive")
                setattr(cfg, key, int(value) if isinstance(old[key], int) else value)
            cfg.validate()
            top.time = SimpleNamespace(monotonic=lambda: EPOCH + now)
            command_status.time = SimpleNamespace(time=lambda: EPOCH + now)
            if control._CAN_ENABLED or control._CAN_THREAD is not None:
                raise RuntimeError("Simulation requires CAN OFF and no CAN worker")
            yield
        finally:
            top.time, command_status.time = old_top_time, old_status_time
            for key, value in old.items():
                setattr(cfg, key, value)


class Session:
    def __init__(self, options=None, overrides=None, capture=True):
        self.options = scenario_options(options or {})
        self.overrides = dict(overrides or {})
        self.now, self.tick = 0., 0
        self.capture = capture
        self.frames, self.inputs, self.events = [], [], []
        self.digest = hashlib.sha256()
        self.done = False
        self.last_observation = {}
        self.last_lines = []
        self.last_command = "STOP"
        self.transport=CurrentCanTransport(control)
        self.last_model_sequence=None
        self.input_id = fingerprint()["source_id"]
        self.config = {k: v for k,v in vars(cfg).items() if k.isupper() and isinstance(v,(str,int,float,bool))}
        self.config.update(self.overrides)
        with environment(self.now, self.overrides), self.transport.bind(self.now), contextlib.redirect_stdout(io.StringIO()):
            self.fsm = top.CalibrationFSMV4()
        self.world=World(self.options)
        self.plant=self.world.plant
        self.world.observe()
        self.last_observation=self.world.last_diagnostic
        self.initial = self.snapshot()

    def snapshot(self):
        fsm, p = self.fsm, self.plant
        return clean(dict(t=self.now, state=fsm.state, command=self.last_command,
                         truth=p.truth(), observation=self.last_observation,
                         failure_reason=fsm.failure_reason, failure_state=fsm.failure_state,
                         collision=p.collision, clearance=p.minimum_clearance,
                         remaining=fsm._insert_remaining_m, stable_frames=len(fsm._samples),
                         target=fsm.motion_target_status, lines=self.last_lines,
                         can=dict(id=0x1e3, data=list(p.movement)),
                         bus_status=p.bus_status,can_frame_count=len(self.world.can_log),
                         terminal=self.done))

    def step(self):
        if self.done:
            return self.snapshot()
        dt = 1/self.options["hz"]
        before = (self.fsm.state, self.last_command)
        with environment(self.now, self.overrides), self.transport.bind(self.now):
            packet=self.world.observe()
            inputs=model_step(packet)
            self.last_observation=self.world.last_diagnostic
            fresh=packet["sequence"]!=self.last_model_sequence
            blind=bool(getattr(self.fsm,'vision_independent',False))
            if blind:
                inputs=dict(det_ok=False,detected_length=None,dist_z=None,yaw_smooth=None,
                            offset_smooth=None,target_bearing_deg=None,vision_meta=None)
            if fresh or blind:
                self.last_lines = [line[0] for line in self.fsm.step(**inputs)]
            if fresh:
                self.last_model_sequence=packet["sequence"]
            self.last_command = self.fsm.execu.last_cmd or "STOP"
            current_can=self.transport.pump(self.now)
            self.world.advance(0.,current_can)
            self.done = self.fsm.state in ("DONE","FAILED") or self.now >= self.options["max_seconds"]
            frame = self.snapshot()
            if before != (self.fsm.state,self.last_command):
                self.events.append(dict(t=self.now,state=self.fsm.state,command=self.last_command,
                                        lines=self.last_lines,can=frame["can"]))
            trace = dict(t=self.now, inputs=inputs, state=self.fsm.state,command=self.last_command,
                         failure_reason=self.fsm.failure_reason, failure_state=self.fsm.failure_state,
                         can=frame["can"],model_packet=packet,model_updated=fresh,fsm_updated=fresh or blind)
            if not self.done:
                until=min(self.now+dt,self.world.next_model_time)
                if getattr(self.fsm,'vision_independent',False):
                    until=min(until,self.now+.01)
                future_can=self.transport.pump(until)
                self.world.advance(until-self.now,future_can)
            else:
                until=self.now
                future_can=[]
            trace.update(can_frames=current_can+future_can,advance_until=until)
            self.digest.update(json.dumps(trace,sort_keys=True,allow_nan=False).encode())
            if self.capture:
                self.frames.append(frame)
                self.inputs.append(trace)
            if not self.done:
                self.tick += 1
                self.now=until
            return frame

    def run(self):
        while not self.done:
            self.step()
        return self.report()

    def report(self):
        fsm,p = self.fsm,self.plant
        reasons = []
        if fsm.state == "FAILED":
            reasons.append(fsm.failure_reason)
        true_remaining = max(0.,p.z-SPEC.INSERT_CAMERA_Z_REMAINDER_M)
        if fsm.state == "DONE" and true_remaining > self.options["completion_tolerance"]:
            reasons.append("FSM DONE with insertion distance remaining")
        if fsm.state not in ("DONE","FAILED"):
            reasons.append("simulation time limit" if self.done else "run in progress")
        return clean(dict(options=self.options,overrides=self.overrides,source_id=self.input_id,
                          simulator_id=SIMULATOR_ID,
                          trace_hash=self.digest.hexdigest(),state=fsm.state,elapsed=self.now,
                          finished=self.done, success=fsm.state=="DONE" and not reasons,reasons=reasons,
                          failure_state=fsm.failure_state,failure_reason=fsm.failure_reason,
                          collision=p.collision,minimum_clearance=p.minimum_clearance,
                          fsm_remaining=fsm._insert_remaining_m,true_remaining=true_remaining,
                          final=p.truth(),events=self.events,config=self.config,vehicle_profile=PROFILE,
                          can_frames=self.world.can_log if self.capture else [],
                          can_frame_count=len(self.world.can_log),
                          frames=self.frames,trace=self.inputs))


def replay_trace(trace, overrides=None):
    """Feed recorded sensor inputs into a fresh original FSM, compare every frame."""
    mismatches = []
    transport=CurrentCanTransport(control)
    movement=[127]*8
    with environment(0.,overrides),transport.bind(0.),contextlib.redirect_stdout(io.StringIO()):
        fsm = top.CalibrationFSMV4()
    for index,row in enumerate(trace):
        with environment(row["t"],overrides),transport.bind(row["t"]):
            if row.get('fsm_updated',row['model_updated']):
                fsm.step(**row['inputs'])
            command = fsm.execu.last_cmd or "STOP"
            current=transport.pump(row["t"])
            for f in current:
                if f["id"]==0x1e3:
                    movement=f["data"]
            actual = dict(state=fsm.state,command=command,failure_reason=fsm.failure_reason,
                          failure_state=fsm.failure_state,
                          can=dict(id=0x1e3,data=list(movement)))
            future=transport.pump(row["advance_until"])
            actual["can_frames"]=current+future
            for f in future:
                if f["id"]==0x1e3:
                    movement=f["data"]
            if any(actual[key]!=row[key] for key in actual):
                mismatches.append(dict(frame=index,t=row["t"],actual=actual,
                                       expected={key:row[key] for key in actual}))
    return dict(frames=len(trace),mismatches=mismatches,passed=not mismatches)


def summarize(reports):
    return dict(total=len(reports),success=sum(r["success"] for r in reports),
                fsm_done=sum(r["state"]=="DONE" for r in reports),
                collisions=sum(bool(r["collision"]) for r in reports),
                reasons=dict(Counter(reason for r in reports for reason in r["reasons"])))
