"""Offline pallet-centred feasibility map; never constructs a CAN executor.

Reuses real FSM decision/settle branches and planner. An ideal event adapter
replaces perception and actuation, not decision rules. See generated README.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'extracted/depth_cam'))
from calib.fsm_v4 import config as cfg, planner as p, top
from calib.fsm_v4.controllers import RotationController
from calib.fsm_v4.motion import forward_seconds
from calib.fsm_v4.pose import VisualPoseFilter

OUT = ROOT / '_docs/analysis/fsm_v4_feasibility_20260906'
HALF = cfg.PALLET_FRONT_VISIBILITY_WIDTH_M / 2
SETTLE = cfg.STOP_MIN_SETTLE_SEC + 1.0  # 10 stable frames at assumed 10 Hz


@lru_cache(maxsize=16384)
def sample_arrays(turn, forward):
    samples = np.asarray(list(p._action_samples(turn, forward)))
    angles = np.radians(samples[:, 0])
    return np.cos(angles), np.sin(angles), samples[:, 1]


@lru_cache(maxsize=65536)
def fast_margin(pose, turn_deg, forward_m, half_angle_deg=None):
    """Identical samples/equations to the original, vectorized for grid sweeps."""
    limit = p.configured_visible_half_angle_deg() if half_angle_deg is None else half_angle_deg
    c, s, forward = sample_arrays(float(turn_deg), float(forward_m))
    points = np.asarray(p._front_edge_points(pose))
    dx = points[:, 0] - cfg.CAMERA_TO_ROT_CENTER_X_M
    dz = points[:, 1] - cfg.CAMERA_TO_ROT_CENTER_Z_M
    x = c[:, None]*dx - s[:, None]*dz + cfg.CAMERA_TO_ROT_CENTER_X_M
    z = s[:, None]*dx + c[:, None]*dz + cfg.CAMERA_TO_ROT_CENTER_Z_M - forward[:, None]
    if np.any(z <= cfg.VISIBILITY_MIN_CORNER_DEPTH_M):
        return -math.inf
    return float(limit - np.max(np.abs(np.degrees(np.arctan2(x, z)))))


def initial_pose(radius, beta):
    # Rotation centre (r*sin(beta), -r*cos(beta)) about PALLET BODY centre.
    # Selected face is at body Z=-HALF. Vehicle points toward body centre.
    a = math.radians(beta)
    x = -HALF * math.sin(a) + cfg.CAMERA_TO_ROT_CENTER_X_M
    z = radius - HALF * math.cos(a) + cfg.CAMERA_TO_ROT_CENTER_Z_M
    return VisualPoseFilter().seed_vehicle(beta, x, z, 100.)


def meta_for(pose):
    points = p._front_edge_points(pose)
    if min(z for _, z in points) <= 0:
        margin = -1.
    else:
        ratio = max(abs(x/z) for x, z in points)
        margin = .5 - ratio/(2*math.tan(math.radians(cfg.CAMERA_HORIZONTAL_FOV_DEG/2)))
    return {'bbox_margin_norm': margin}


class NullIO:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class IdealFSM(top.CalibrationFSMV4):
    def __init__(self, pose):
        self.now = 100.
        top.time = SimpleNamespace(monotonic=lambda: self.now)
        self.tracer = None
        self.execu = NullIO()
        self.status = NullIO()
        self._pose_filter = VisualPoseFilter()
        self._rotation = RotationController()
        self.reset()
        self.pose = pose
        self.state = 'ACQUIRE_VERIFY'
        self._initial_visibility_complete = True
        self.actions = []
        self.states = []
        self.min_margin = p.action_visibility_margin_deg(pose, 0., 0.)
        self.pending = None

    def _observe_pose(self, *args):
        return self.pose, 'ideal stopped pose'

    def _stable_observation(self, pose, center, margin, required_frames=None):
        return pose, center, margin

    def _settle_observation(self, pose, center, margin, lines):
        return pose, center, margin

    def _update_rotation_adaptation(self, *args):
        pass  # No measured overshoot or session history in a geometry map.

    def _rotation_trace_result(self):
        return {}

    def _fail(self, reason, lines):
        self._failure_reason = reason
        self._failure_state = self.state
        self._set_state('FAILED')

    def _begin_rotation(self, state, mode, error_deg, pose, target_yaw_deg=None):
        ok = super()._begin_rotation(state, mode, error_deg, pose, target_yaw_deg)
        if ok:
            self.pending = ('rotate', state, error_deg, self._rotation.plan.hold_sec)
        return ok

    def _begin_translation(self, state, command, pose, target_m, kind):
        super()._begin_translation(state, command, pose, target_m, kind)
        self.pending = ('drive', state, target_m if command == 'FWD' else -target_m, 0.)

    def advance_action(self):
        kind, state, amount, duration = self.pending
        self.pending = None
        before = self.pose
        if kind == 'rotate':
            self.pose = p._pose_after_action(before, amount, 0.)
            self.min_margin = min(self.min_margin, p.action_visibility_margin_deg(before, amount, 0.))
            self._rotation_stop_error_deg = 0.
            self._rotation_coast_tracking = False
            target = {
                'FACE_ROTATE': 'FACE_SETTLE', 'RECENTER_ROTATE': 'RECENTER_SETTLE',
                'WAYPOINT_TURN': 'WAYPOINT_TURN_SETTLE', 'FINAL_ROTATE': 'FINAL_SETTLE',
            }.get(state)
            if target is None:
                self._fail('initial visibility recovery outside analysis scope', [])
                return
        else:
            if state == 'STANDOFF_MOVE':
                # Optimistic maximum fitted displacement within the real 15 s
                # command cap; reach near edge of 4.0 +/-0.1 m band.
                amount = max(0., before.pallet_z_m - (cfg.SAFETY_STANDOFF_Z_M + cfg.SAFETY_STANDOFF_BAND_M))
                cap = cfg.FWD_MAX_COMMAND_SEC
                active = max(0., cap - cfg.FWD_T0_SEC)
                accel_t = min(active, cfg.FWD_ACCEL_DURATION_SEC)
                max_distance = .5*cfg.FWD_ACCEL_M_S2*accel_t**2 + max(0., active-accel_t)*cfg.FWD_ACCEL_M_S2*accel_t
                # Predictive STOP may trigger up to the configured 0.12 m early.
                if amount > max_distance + cfg.FWD_PREDICTIVE_MAX_ADVANCE_M:
                    self._fail('standoff translation timeout (fitted 15 s envelope)', [])
                    return
                target = 'STANDOFF_SETTLE'
                amount = min(amount, max_distance)
            elif state == 'WAYPOINT_DRIVE':
                amount = min(amount, max(0., before.pallet_z_m-cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M))
                lat, longitudinal = p.staging_position_errors(before)
                c = math.cos(math.radians(before.yaw_deg))
                if c > 0 and longitudinal < 0:
                    amount = min(amount, -longitudinal/c)
                target = 'WAYPOINT_DRIVE_SETTLE'
            else:
                target = 'INSERT_SETTLE'
            duration = forward_seconds(abs(amount))
            self.pose = p._pose_after_action(before, 0., amount)
            if state != 'INSERT_DRIVE':
                self.min_margin = min(self.min_margin, p.action_visibility_margin_deg(before, 0., amount))
        self.actions.append({'kind': kind, 'state': state, 'amount': amount, 'seconds': duration})
        self.now += duration + SETTLE
        self._set_state(target)

    def run(self):
        for _ in range(100):
            self.states.append(self.state)
            if self.state in ('READY_TO_INSERT', 'FAILED'):
                break
            self.step(True, 1.1, self.pose.pallet_z_m, self.pose.yaw_deg,
                      self.pose.pallet_x_m, vision_meta=meta_for(self.pose))
            if self.pending is not None:
                self.advance_action()
        else:
            self._fail('analysis event limit / non-progress loop', [])
        success = self.state == 'READY_TO_INSERT'
        clearance = 0.
        if success:
            _, hits = p.fork_opening_alignment(self.pose, enforce_entry_distance=False)
            clearance = min(cfg.INSERT_OPENING_SPAN_M/2-abs(v) for v in hits)
        turning = sum(abs(a['amount']) for a in self.actions if a['kind'] == 'rotate')
        elapsed = self.now - 100.
        result = dict(success=success, reason=self._failure_reason or 'insertion accepted',
                      terminal=self.state, cycles=self._correction_cycles,
                      turn_deg=turning, elapsed_sec=elapsed,
                      forward_m=self._forward_used_m, min_fov_deg=self.min_margin,
                      clearance_m=clearance, final_yaw_deg=self.pose.yaw_deg,
                      final_camera_z_m=self.pose.pallet_z_m,
                      actions=self.actions, states=self.states)
        return result


def rotation_only_impossible(pose):
    """Sufficient analytic impossibility bounds under CURRENT yaw/ray gates.

For alpha in +/-20 degrees, fork rays require:
 abs(u-d*tan(alpha)) + w/(2*cos(alpha)) <= opening/2,
 d >= L*cos(alpha) + w/2*abs(sin(alpha)).
These necessary bounds prove rejection; no numerical search claims a proof.
"""
    a = math.radians(cfg.FINAL_YAW_TOL_DEG)
    d = -pose.rot_z_pallet_m
    u = abs(pose.rot_x_pallet_m)
    minimum_d = min(cfg.ROT_CENTER_TO_FORK_TIP_M,
                    cfg.ROT_CENTER_TO_FORK_TIP_M*math.cos(a)+cfg.FORK_OUTER_SPAN_M/2*math.sin(a))
    # Looser but universally necessary lateral bound (cos(alpha) <= 1).
    max_u = max(0., d)*math.tan(a) + (cfg.INSERT_OPENING_SPAN_M-cfg.FORK_OUTER_SPAN_M)/2
    return d < minimum_d - 1e-9 or u > max_u + 1e-9


def classify(result):
    if not result['success']:
        return 1
    if (result['turn_deg'] <= 10 and result['cycles'] <= 3
            and result['clearance_m'] >= .02 and result['min_fov_deg'] >= 3
            and result['elapsed_sec'] <= 60):
        return 4
    if (result['turn_deg'] <= 35 and result['cycles'] <= 5
            and result['clearance_m'] >= .01 and result['min_fov_deg'] >= 1
            and result['elapsed_sec'] <= 90):
        return 3
    return 2


def evaluate(radius, beta):
    pose = initial_pose(radius, beta)
    common = dict(radius_m=radius, beta_deg=beta,
                  x_m=radius*math.sin(math.radians(beta)),
                  y_m=radius*math.cos(math.radians(beta)),
                  camera_z_m=pose.pallet_z_m)
    if pose.pallet_z_m <= cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M and rotation_only_impossible(pose):
        return {**common, 'category': 0, 'success': False,
                'reason': 'rotation-only geometry impossible under current insertion gates'}
    if pose.pallet_z_m <= 0. or not p.action_keeps_front_visible(pose, 0., 0., meta_for(pose)):
        return {**common, 'category': 5, 'success': False,
                'reason': 'initial full-front observation unavailable; search not simulated'}
    runtime_time = top.time
    try:
        f = IdealFSM(pose)
        result = f.run()
    finally:
        top.time = runtime_time
    return {**common, **result, 'category': classify(result)}


def validate_fast_margin():
    rng = np.random.default_rng(20260906)
    original = p.action_visibility_margin_deg
    worst = 0.
    for _ in range(300):
        pose = initial_pose(rng.uniform(1.2, 10.), rng.uniform(-45., 45.))
        turn, forward = rng.uniform(-20., 20.), rng.uniform(0., .8)
        a, b = original(pose, turn, forward), fast_margin(pose, turn, forward)
        if math.isfinite(a):
            worst = max(worst, abs(a-b))
            assert abs(a-b) < 1e-10, (a, b)
        else:
            assert a == b
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radial-step', type=float, default=.1)
    parser.add_argument('--angle-step', type=float, default=2.5)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    error = validate_fast_margin()
    p.action_visibility_margin_deg = fast_margin
    if args.probe:
        for r, a in [(2., 0.), (2.5, 0.), (3., 0.), (4., 0.), (6., 0.), (8., 0.),
                     (10., 0.), (3., 15.), (4., 15.), (6., 30.), (4., 45.)]:
            started = time.perf_counter()
            result = evaluate(r, a)
            print(json.dumps(result), 'cpu_sec', round(time.perf_counter()-started, 3), flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    radii = np.round(np.arange(.8, 10.+1e-6, args.radial_step), 6)
    angles = np.round(np.arange(-45., 45.+1e-6, args.angle_step), 6)
    results = []
    started = time.perf_counter()
    for i, r in enumerate(radii):
        results.extend(evaluate(float(r), float(a)) for a in angles)
        if i % 5 == 0:
            print(f'{i+1}/{len(radii)} radial rows, {len(results)} points, {time.perf_counter()-started:.1f}s', flush=True)
    keys = sorted(set().union(*(row.keys() for row in results))-{'actions', 'states'})
    with (OUT/'samples.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    payload = dict(radii=radii.tolist(), angles=angles.tolist(), results=results,
                   vectorization_max_error=error,
                   source_sha256={name: hashlib.sha256((ROOT/'extracted/depth_cam/calib/fsm_v4'/name).read_bytes()).hexdigest()
                                  for name in ('config.py', 'planner.py', 'top.py', 'pose.py')},
                   config={k:v for k,v in vars(cfg).items() if k.isupper() and isinstance(v,(int,float,str,bool))},
                   category_counts=dict(Counter(r['category'] for r in results)),
                   reason_counts=dict(Counter(r['reason'] for r in results)))
    (OUT/'results.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:payload[k] for k in ('category_counts', 'reason_counts')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
