from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


DEPTH_CAM_DIR = Path(__file__).resolve().parents[1]
if str(DEPTH_CAM_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_CAM_DIR))

from calib.fsm_v4.rotation_artifact import (  # noqa: E402
    ArtifactValidationError,
    PiecewiseRotationResponse,
    canonical_model_sha256,
    load_validated_rotation_artifact,
    select_rotation_response,
    validate_rotation_artifact_bytes,
)
from calib.fsm_v4.rotation_model import RotationResponse  # noqa: E402
from calib.fsm_v4 import config as cfg  # noqa: E402
from calib.fsm_v4.controllers import RotationController  # noqa: E402


def ready_artifact(model_id: str) -> dict:
    return {
        "schema_version": 1,
        "model_type": "log_rotation_piecewise",
        "parameters": {
            "command_strength": 30,
            "dead_time_sec": 1.0,
            "accel_duration_sec": 2.0,
            "accel_deg_s2": 4.0,
            "max_rate_deg_s": 8.0,
            "inertia_intercept_deg": 0.5,
            "inertia_slope_sec": 0.25,
        },
        "fit_metadata": {
            "fitted_min_hold_sec": 1.5,
            "fitted_max_hold_sec": 4.0,
            "fitted_min_angle_deg": 1.0,
            "fitted_max_angle_deg": 30.0,
        },
        "fit_report": {
            "status": "READY",
            "safe_for_control": True,
            "deployment_blockers": [],
            "model_type": (
                "deadtime_constant_acceleration_cruise_linear_inertia"
            ),
            "command_strength": 30,
            "angle_column": "yaw_deg",
            "angle_domain": "heading",
            "angle_period_deg": 360.0,
            "settings": {
                "command_strength": 30,
                "angle_period_deg": 360.0,
                "inertia_horizon_sec": 2.0,
                "min_loo_predictions": 4,
                "max_loo_rmse_deg": 3.0,
                "require_actual_can_timing": True,
                "can_time_point": "return",
            },
            "source": {
                "angle_column": "yaw_deg",
                "angle_domain": "heading",
                "model_ids": [model_id],
                "mixed_model_ids_detected": False,
                "recording_errors": {},
                "unidentified_model_recordings": [],
                "direction_sign_audit": {
                    "ambiguous_directions": [],
                    "left_right_modal_signs_are_opposite": True,
                },
            },
            "identifiability": {
                "dead_time": True,
                "acceleration_and_cruise": True,
                "linear_inertia": True,
            },
            "validation": {
                "leave_one_recording_out_folds": {
                    "attempted": 4,
                    "successful": 4,
                    "equal_recording_weight_rmse_deg": 1.25,
                },
            },
        },
    }


def encoded(artifact: dict) -> bytes:
    return json.dumps(artifact, allow_nan=False).encode("utf-8")


class RotationArtifactValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_id = "sha256:" + "1" * 64

    def test_ready_yaw_heading_artifact_loads(self):
        response = validate_rotation_artifact_bytes(
            encoded(ready_artifact(self.model_id)),
            expected_model_sha256_id=self.model_id,
        )
        self.assertIsInstance(response, PiecewiseRotationResponse)
        self.assertEqual(response.domain, "heading")
        self.assertEqual(response.command_strength, 30)
        self.assertAlmostEqual(response.residual_sd_deg, 1.25)

    def test_rejects_every_deployment_safety_contract_violation(self):
        mutations = {
            "not ready": lambda a: a["fit_report"].update(status="DIAGNOSTIC_ONLY"),
            "unsafe": lambda a: a["fit_report"].update(safe_for_control=False),
            "blocker": lambda a: a["fit_report"].update(deployment_blockers=["bad"]),
            "bearing": lambda a: a["fit_report"].update(angle_domain="bearing"),
            "wrong column": lambda a: a["fit_report"].update(angle_column="center_bearing_deg"),
            "mixed ids": lambda a: a["fit_report"]["source"].update(
                mixed_model_ids_detected=True
            ),
            "recording error": lambda a: a["fit_report"]["source"].update(
                recording_errors={"run": "missing"}
            ),
            "logical CAN time": lambda a: a["fit_report"]["settings"].update(
                require_actual_can_timing=False
            ),
            "failed LOO fold": lambda a: a["fit_report"]["validation"][
                "leave_one_recording_out_folds"
            ].update(successful=3),
            "high LOO error": lambda a: a["fit_report"]["validation"][
                "leave_one_recording_out_folds"
            ].update(equal_recording_weight_rmse_deg=3.01),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                artifact = copy.deepcopy(ready_artifact(self.model_id))
                mutate(artifact)
                with self.assertRaises(ArtifactValidationError):
                    validate_rotation_artifact_bytes(
                        encoded(artifact),
                        expected_model_sha256_id=self.model_id,
                    )

    def test_rejects_artifact_fitted_with_another_checkpoint(self):
        artifact = ready_artifact("sha256:" + "2" * 64)
        with self.assertRaisesRegex(
            ArtifactValidationError, "does not match",
        ):
            validate_rotation_artifact_bytes(
                encoded(artifact),
                expected_model_sha256_id=self.model_id,
            )

    def test_rejects_duplicate_json_keys_and_nonfinite_values(self):
        payload = encoded(ready_artifact(self.model_id))
        duplicate = payload.replace(
            b'"schema_version": 1,',
            b'"schema_version": 1, "schema_version": 1,',
            1,
        )
        with self.assertRaisesRegex(ArtifactValidationError, "duplicate"):
            validate_rotation_artifact_bytes(
                duplicate, expected_model_sha256_id=self.model_id,
            )
        nonfinite = payload.replace(b'"dead_time_sec": 1.0', b'"dead_time_sec": NaN')
        with self.assertRaises(ArtifactValidationError):
            validate_rotation_artifact_bytes(
                nonfinite, expected_model_sha256_id=self.model_id,
            )

    def test_file_loader_returns_content_hash(self):
        payload = encoded(ready_artifact(self.model_id))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotation_model.generated.json"
            path.write_bytes(payload)
            response, artifact_id = load_validated_rotation_artifact(
                path, expected_model_sha256_id=self.model_id,
            )
        self.assertIsInstance(response, PiecewiseRotationResponse)
        self.assertRegex(artifact_id, r"^sha256:[0-9a-f]{64}$")


class RotationArtifactSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fallback = RotationResponse(
            startup_delay_sec=1.0,
            stop_delay_sec=0.2,
            accel_deg_s2=4.0,
            decel_deg_s2=20.0,
            max_rate_deg_s=8.0,
        )

    def test_current_pt_hash_must_match_before_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model_path = directory / "new.pt"
            model_path.write_bytes(b"new inference checkpoint")
            model_id = canonical_model_sha256(model_path)
            artifact_path = directory / "rotation_model.generated.json"
            artifact_path.write_bytes(encoded(ready_artifact(model_id)))
            selected = select_rotation_response(
                artifact_path,
                self.fallback,
                inference_model_path=model_path,
            )
        self.assertTrue(selected.artifact_active)
        self.assertIsInstance(selected.response, PiecewiseRotationResponse)
        self.assertEqual(selected.inference_model_sha256_id, model_id)

    def test_missing_or_hash_mismatched_artifact_uses_exact_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model_path = directory / "runtime.pt"
            model_path.write_bytes(b"runtime checkpoint")
            artifact_path = directory / "rotation_model.generated.json"
            artifact_path.write_bytes(encoded(ready_artifact("sha256:" + "a" * 64)))
            selected = select_rotation_response(
                artifact_path,
                self.fallback,
                inference_model_path=model_path,
            )
            runtime_model_id = canonical_model_sha256(model_path)
            missing = select_rotation_response(
                directory / "missing.json",
                self.fallback,
                inference_model_path=model_path,
            )
        self.assertFalse(selected.artifact_active)
        self.assertIs(selected.response, self.fallback)
        self.assertIn("does not match", selected.fallback_reason)
        self.assertEqual(selected.inference_model_sha256_id, runtime_model_id)
        self.assertEqual(selected.artifact_sha256_id, "")
        self.assertFalse(missing.artifact_active)
        self.assertIs(missing.response, self.fallback)


class PiecewiseRuntimeResponseTests(unittest.TestCase):
    def setUp(self) -> None:
        model_id = "sha256:" + "3" * 64
        self.response = validate_rotation_artifact_bytes(
            encoded(ready_artifact(model_id)),
            expected_model_sha256_id=model_id,
        )

    def test_piecewise_drive_and_linear_inertia_are_inverted(self):
        # At hold=2: one driven second, rate=4, command angle=2,
        # inertia=0.5+0.25*4=1.5, total=3.5 degrees.
        self.assertAlmostEqual(self.response.rate_at(2.0), 4.0)
        self.assertAlmostEqual(self.response.angle_at(2.0), 2.0)
        self.assertAlmostEqual(self.response.coast_after_hold(2.0), 1.5)
        self.assertAlmostEqual(self.response.total_angle(2.0), 3.5)
        plan = self.response.plan(3.5, min_angle_deg=1.0)
        self.assertTrue(plan.feasible)
        self.assertAlmostEqual(plan.hold_sec, 2.0, places=8)
        self.assertAlmostEqual(plan.predicted_total_deg, 3.5, places=8)

    def test_legacy_coast_adjustment_is_not_applied_twice(self):
        self.assertAlmostEqual(
            self.response.adjusted_coast_after_hold(2.0, 999.0), 1.5
        )
        self.assertAlmostEqual(
            self.response.adjusted_total_angle(2.0, 999.0), 3.5
        )
        self.assertAlmostEqual(
            self.response.stop_now_margin_deg(
                2.0, measured_rate_deg_s=4.0,
                measurement_age_sec=0.0, coast_reduction_deg=999.0,
            ),
            1.5,
        )

    def test_heading_response_scales_to_bearing_using_camera_offset(self):
        bearing = self.response.in_bearing_domain(2.0, 0.68)
        gain = (2.0 + 0.68) / 2.0
        self.assertEqual(bearing.domain, "bearing")
        self.assertAlmostEqual(bearing.max_rate_deg_s, 8.0 * gain)
        self.assertAlmostEqual(bearing.inertia_intercept_deg, 0.5 * gain)
        self.assertAlmostEqual(bearing.inertia_slope_sec, 0.25)
        self.assertAlmostEqual(bearing.total_angle(2.0), 3.5 * gain)

    def test_does_not_extrapolate_below_observed_angle_range(self):
        plan = self.response.plan(0.5, min_angle_deg=0.1)
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.hold_sec, 0.0)

    def test_existing_controller_uses_generated_response_in_both_domains(self):
        with (
            patch.object(cfg, "ROTATION_RESPONSE", self.response),
            patch.object(cfg, "ROT_MIN_COMMANDABLE_ANGLE_DEG", 1.0),
            patch.object(cfg, "ROT_ACTIVE_COAST_HEURISTIC_REDUCTION_DEG", 0.0),
        ):
            heading = RotationController()
            heading.start(3.5, "ROT_RIGHT", 10.0, 10.0, domain="heading")
            bearing = RotationController()
            bearing.start(
                3.5, "ROT_RIGHT", 10.0, 10.0,
                domain="bearing", range_m=2.0,
            )
        self.assertIsNotNone(heading.plan)
        self.assertAlmostEqual(heading.plan.predicted_total_deg, 3.5, places=8)
        self.assertEqual(bearing.response.domain, "bearing")
        self.assertAlmostEqual(
            bearing.response.max_rate_deg_s,
            self.response.max_rate_deg_s * (2.0 + 0.68) / 2.0,
        )


if __name__ == "__main__":
    unittest.main()
