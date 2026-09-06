"""User-selected endpoint regression. Not an instantaneous motion/coast model."""
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path

from .rotation_artifact import (
    ArtifactValidationError, RotationResponseSelection, canonical_model_sha256,
    _no_duplicate_object, _reject_json_constant, _finite_number,
)
from .rotation_model import RotationPlan, bearing_gain


@dataclass(frozen=True)
class DelayedLinearEndpointResponse:
    slope_deg_s: float
    startup_delay_sec: float
    fitted_min_hold_sec: float
    fitted_max_hold_sec: float
    residual_sd_deg: float
    measured_settle_p90_sec: float = 3.0  # Guard policy, NOT a fitted percentile.
    domain: str = 'heading'
    source: str = 'User-selected Cleanlabel 48-point delayed linear endpoint fit'
    command_strength: int = 30
    endpoint_only: bool = True
    uses_embedded_inertia: bool = True  # Total angle already includes coast.
    stop_delay_sec: float = 0.0
    accel_deg_s2: float = 0.0          # Unknown; never used to predict movement.
    accel_duration_sec: float = 0.0
    fitted_min_angle_deg: float = 0.0

    @property
    def max_rate_deg_s(self):
        # Compatibility metadata: endpoint slope, NOT measured peak speed.
        return self.slope_deg_s

    @property
    def fitted_max_angle_deg(self):
        return self.total_angle(self.fitted_max_hold_sec)

    @property
    def min_total_deg(self):
        return 0.0

    def total_angle(self, hold_sec):
        value = float(hold_sec)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Command duration must be finite and nonnegative')
        return self.slope_deg_s * max(0., value - self.startup_delay_sec)

    def in_bearing_domain(self, range_m, rot_centre_offset_m):
        if self.domain == 'bearing':
            return self
        gain = bearing_gain(range_m, rot_centre_offset_m)
        return replace(self, slope_deg_s=self.slope_deg_s * gain,
                       residual_sd_deg=self.residual_sd_deg * gain, domain='bearing',
                       source=self.source + f' [bearing x{gain:.3f}]')

    def command_seconds(self, target_deg, max_hold_sec=2.5, coast_reduction_deg=0.):
        target, cap = abs(float(target_deg)), min(float(max_hold_sec), self.fitted_max_hold_sec)
        if not math.isfinite(target) or not math.isfinite(cap) or cap < 0:
            raise ValueError('Invalid target or time limit')
        if target == 0 or cap <= self.startup_delay_sec:
            return 0.
        return min(cap, self.startup_delay_sec + target / self.slope_deg_s)

    def plan(self, error_deg, min_angle_deg=2.5, max_angle_deg=20.,
             timeout_margin_sec=1., max_hold_sec=2.5, coast_reduction_deg=0.):
        signed = float(error_deg)
        if not math.isfinite(signed):
            raise ValueError('Nonfinite rotation error')
        sign = 1. if signed > 0 else -1.
        want = abs(signed)
        target = min(want, float(max_angle_deg))
        hold = self.command_seconds(target, max_hold_sec)
        if want < min_angle_deg:
            hold = 0.
        total = self.total_angle(hold)
        feasible = hold > 0 and total >= target - 1e-8
        note = 'endpoint-only; STOP state/coast split unknown; no extra inertia correction'
        if not feasible:
            note += '; below minimum or capped: settle and replan'
        return RotationPlan(sign * target, 'ROT_RIGHT' if signed > 0 else 'ROT_LEFT',
                            sign, hold, hold + timeout_margin_sec,
                            0., 0., 0., total, self.measured_settle_p90_sec,
                            self.residual_sd_deg, feasible, note)

    def stop_now_margin_deg(self, *args, **kwargs):
        return 0.  # Unknown coast; do not invent a live predictive coast margin.

    def should_stop_now(self, *args, **kwargs):
        return False  # Controller retains actual target crossing + hold + safety stops.

    def settle_sec(self, hold_sec):
        return self.measured_settle_p90_sec

    settle_wait_sec = settle_sec

    def angle_uncertainty_deg(self, hold_sec):
        return self.residual_sd_deg


def validate_endpoint_artifact(payload, expected_model_sha256_id):
    data = json.loads(payload, object_pairs_hook=_no_duplicate_object,
                      parse_constant=_reject_json_constant)
    if data.get('schema_version') != 1 or data.get('model_type') != 'selected_delayed_linear_endpoint':
        raise ArtifactValidationError('Unsupported selected endpoint artifact')
    if data.get('operator_selected') is not True:
        raise ArtifactValidationError('Endpoint response was not explicitly selected')
    if data.get('model_sha256') != expected_model_sha256_id:
        raise ArtifactValidationError('Selected endpoint and inference model hashes differ')
    if data.get('command_strength') != 30 or data.get('angle_domain') != 'heading':
        raise ArtifactValidationError('Selected endpoint requires strength 30 and heading')
    contract = data.get('inference_contract')
    if contract != dict(padding_px=100, padding_mode='BORDER_REFLECT_101', imgsz=640,
                        confidence=0.4, instance_selection='max_box_conf', input_width=640, input_height=480):
        raise ArtifactValidationError('Unexpected endpoint inference contract')
    p = data['parameters']
    k = _finite_number(p['slope_deg_s'], 'slope')
    d = _finite_number(p['effective_delay_s'], 'delay')
    lo, hi = [_finite_number(v, 'duration range') for v in data['measured_duration_range_s']]
    validation = data['validation']
    rmse = _finite_number(validation['equal_recording_loo_rmse_deg'], 'CV RMSE')
    if not (0 < k <= 60 and 0 <= lo < hi < 5 and 0 <= d < min(3., hi) and 0 <= rmse <= 3):
        raise ArtifactValidationError('Endpoint parameters or validation outside limits')
    if (type(validation['points']) is not int or validation['points'] < 20
            or type(validation['recordings']) is not int or validation['recordings'] < 4
            or validation['complete_loo_coverage'] is not True):
        raise ArtifactValidationError('Insufficient complete endpoint validation')
    guard = _finite_number(data['settle_guard_sec'], 'settle guard')
    if guard < 3:
        raise ArtifactValidationError('Endpoint requires >=3s post-STOP guard plus live stability')
    return DelayedLinearEndpointResponse(k, d, lo, hi, rmse, guard)


def load_selected_endpoint(path, inference_model_path):
    """Fail closed: never silently substitute the old physical model."""
    payload = Path(path).read_bytes()
    model_id = canonical_model_sha256(inference_model_path)
    response = validate_endpoint_artifact(payload, model_id)
    return RotationResponseSelection(response, True, str(path),
                                     'sha256:' + hashlib.sha256(payload).hexdigest(), model_id, '')


def validate_runtime_inference_contract(config):
    expected = dict(CONF_THR=.4, POSE_REFLECT_PADDING_PX=100, POSE_INFERENCE_IMGSZ=640,
                    POSE_INSTANCE_SELECTION='max_box_conf', STREAM_W=640, STREAM_H=480,
                    POSE_KPT_VIS_THR=.5, PALLET_FACE_W=1.1, PALLET_FACE_H=.15, PALLET_BODY_D=1.1)
    for key, value in expected.items():
        if getattr(config, key, None) != value:
            raise ArtifactValidationError(f'Live inference differs from selected endpoint calibration: {key}')
