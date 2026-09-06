import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

DEPTH_CAM = Path(__file__).resolve().parents[1]
if str(DEPTH_CAM) not in sys.path:
    sys.path.insert(0, str(DEPTH_CAM))
from calib.fsm_v4 import config as cfg
from calib import config as pose_cfg
from calib.fsm_v4.controllers import RotationController
from calib.fsm_v4.endpoint_response import validate_endpoint_artifact, validate_runtime_inference_contract
from calib.fsm_v4.rotation_artifact import ArtifactValidationError
import main_rec_v4 as launcher


class SelectedEndpointTests(unittest.TestCase):
    def test_config_and_forward_inverse_are_exact(self):
        cfg.validate()
        self.assertEqual(launcher.RUNTIME_MODEL_FILENAME, pose_cfg.MODEL_FILENAME)
        self.assertTrue(cfg.ROT_GENERATED_ARTIFACT_ACTIVE)
        self.assertTrue(cfg.ROTATION_RESPONSE.endpoint_only)
        for angle in (2.5, 5, 10, 14):
            plan = cfg.ROTATION_RESPONSE.plan(angle)
            self.assertAlmostEqual(plan.hold_sec, 1.276363332869698 + angle/12.10246966536945)
            self.assertAlmostEqual(plan.predicted_total_deg, angle)
        self.assertEqual(cfg.ROT_ACTIVE_COAST_HEURISTIC_REDUCTION_DEG, 0)

    def test_hold_cap_small_targets_and_bearing(self):
        response = cfg.ROTATION_RESPONSE
        self.assertEqual(response.plan(0).hold_sec, 0)
        self.assertEqual(response.plan(1).hold_sec, 0)
        self.assertEqual(response.plan(50).hold_sec, 2.5)
        self.assertFalse(response.plan(50).feasible)
        b = response.in_bearing_domain(2., .68)
        self.assertAlmostEqual(b.total_angle(2.), response.total_angle(2.)*1.34)
        with self.assertRaises(ValueError): response.plan(float('nan'))

    def test_endpoint_does_not_mix_adaptive_time_scaling(self):
        with patch.object(cfg, 'ROT_ADAPTIVE_OVERSHOOT_ENABLED', True):
            c = RotationController()
            c.start(10, 'ROT_RIGHT', 1., 1., domain='heading', adaptive_time_scale=.5)
        self.assertEqual(c.adaptive_time_scale, 1.)
        self.assertAlmostEqual(c.plan.hold_sec, cfg.ROTATION_RESPONSE.command_seconds(10))

    def test_controller_does_not_apply_old_coast_or_stop_on_noisy_rate(self):
        c = RotationController()
        c.start(10, 'ROT_RIGHT', 100., 100., domain='heading')
        u = c.update(8, 101.8, 101.8, .1)
        self.assertFalse(u.should_stop)
        self.assertEqual(u.coast_margin_deg, 0)
        u = c.update(7, 100+c.plan.hold_sec+.001, 100+c.plan.hold_sec+.001, .1)
        self.assertTrue(u.should_stop)
        self.assertEqual(u.stop_reason, 'fitted_hold_elapsed')
        c.start(10, 'ROT_RIGHT', 200., 200., domain='heading')
        self.assertEqual(c.update(-.1, 201., 201., .1).stop_reason, 'target_crossed')

    def test_reject_mismatched_model_and_failed_validation(self):
        data = json.loads(cfg.ROT_GENERATED_ARTIFACT_PATH.read_text(encoding='utf-8'))
        with self.assertRaises(ArtifactValidationError):
            validate_endpoint_artifact(json.dumps(data), 'sha256:'+'0'*64)
        bad = copy.deepcopy(data)
        bad['validation']['equal_recording_loo_rmse_deg'] = 4.
        with self.assertRaises(ArtifactValidationError):
            validate_endpoint_artifact(json.dumps(bad), data['model_sha256'])
        bad = copy.deepcopy(data); bad['operator_selected'] = False
        with self.assertRaises(ArtifactValidationError):
            validate_endpoint_artifact(json.dumps(bad), data['model_sha256'])
        with patch.object(pose_cfg, 'POSE_REFLECT_PADDING_PX', 0):
            with self.assertRaises(ArtifactValidationError): validate_runtime_inference_contract(pose_cfg)

    def test_old_calibrate_cannot_overwrite_selected_function(self):
        with patch.object(launcher, 'run_rotation_calibration_pipeline') as pipeline:
            self.assertEqual(launcher.main(require_calibration=True), 2)
            pipeline.assert_not_called()

    def test_artifact_identity_before_hardware(self):
        ready = launcher.RotationCalibrationReady(Path(pose_cfg.MODEL_PATH), cfg.ROT_GENERATED_ARTIFACT_MODEL_SHA256,
                                                  cfg.ROT_GENERATED_ARTIFACT_PATH, cfg.ROT_GENERATED_ARTIFACT_SHA256)
        launcher._assert_runtime_artifact_active(cfg, ready)


if __name__ == '__main__': unittest.main()
