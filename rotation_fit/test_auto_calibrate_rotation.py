"""Tests for the transactional automatic rotation-calibration orchestrator."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from . import auto_calibrate_rotation as auto
    from .piecewise_rotation_model import LogRotationModel
except ImportError:  # unittest discovery from rotation_fit/
    import auto_calibrate_rotation as auto
    from piecewise_rotation_model import LogRotationModel


RECORDINGS = tuple(f"recording_{index}" for index in range(4))


def _ready_artifact(model_id: str, recordings=RECORDINGS) -> bytes:
    model = LogRotationModel(
        command_strength=30,
        dead_time_sec=0.8,
        accel_duration_sec=1.5,
        accel_deg_s2=8.0,
        max_rate_deg_s=12.0,
        inertia_intercept_deg=0.0,
        inertia_slope_sec=0.2,
        fitted_min_hold_sec=1.0,
        fitted_max_hold_sec=2.4,
        fitted_min_angle_deg=1.0,
        fitted_max_angle_deg=18.0,
    )
    artifact = model.to_dict()
    artifact["fit_report"] = {
        "status": "READY",
        "safe_for_control": True,
        "deployment_blockers": [],
        "model_type": "deadtime_constant_acceleration_cruise_linear_inertia",
        "command_strength": 30,
        "angle_column": "yaw_deg",
        "angle_domain": "heading",
        "angle_period_deg": 360.0,
        "settings": {
            "command_strength": 30,
            "angle_period_deg": 360.0,
            "require_actual_can_timing": True,
            "can_time_point": "return",
            "inertia_horizon_sec": 2.0,
            "min_loo_predictions": 4,
            "max_loo_rmse_deg": 3.0,
        },
        "source": {
            "angle_column": "yaw_deg",
            "angle_domain": "heading",
            "model_ids": [model_id],
            "mixed_model_ids_detected": False,
            "recording_errors": {},
            "unidentified_model_recordings": [],
            "recordings_selected": list(recordings),
            "recordings_loaded": len(recordings),
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
                "equal_recording_weight_rmse_deg": 1.0,
            }
        },
    }
    return (json.dumps(artifact, allow_nan=False) + "\n").encode("utf-8")


class PipelineFixture:
    def __init__(self, root: Path) -> None:
        self.runtime_root = root / "runtime"
        self.runtime_root.mkdir()
        self.model_path = self.runtime_root / "candidate.pt"
        self.model_path.write_bytes(b"candidate-model-weights")
        self.model_id = auto.sha256_file(self.model_path)
        self.recordings_dir = root / "recordings"
        self.recordings_dir.mkdir()
        self.runs_dir = root / "runs"
        self.deploy_path = root / "deploy" / "rotation_model.generated.json"
        self.fit_arguments: list[str] = []

    def batch_runner(self, **kwargs):
        self.assert_batch_arguments(kwargs)
        results_dir = Path(kwargs["results_dir"])
        suffix = kwargs["results_suffix"]
        results_dir.mkdir(parents=True, exist_ok=True)
        for prefix in RECORDINGS:
            path = results_dir / f"{prefix}{suffix}"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "frame_i", "yaw_deg", "pose_ok", "valid",
                        "confidence", "model_hash", "angle_domain",
                        "angle_source",
                    ),
                )
                writer.writeheader()
                for frame_i in range(36, 41):
                    writer.writerow(
                        {
                            "frame_i": frame_i,
                            "yaw_deg": frame_i / 10.0,
                            "pose_ok": 1,
                            "valid": 1,
                            "confidence": 0.9,
                            "model_hash": self.model_id,
                            "angle_domain": "heading",
                            "angle_source": "pnp_face_orientation",
                        }
                    )
        return {
            "schema_version": 1,
            "status": "complete",
            "model_hash": self.model_id,
            "angle_column": "yaw_deg",
            "angle_domain": "heading",
            "valid_column": "pose_ok",
            "model_id_column": "model_hash",
            "results_suffix": suffix,
            "results_dir": str(results_dir.resolve()),
            "recording_count": len(RECORDINGS),
            "successful_recording_prefixes": list(RECORDINGS),
            "recording_audit": [],
            "recordings": [{"prefix": name} for name in RECORDINGS],
        }

    def assert_batch_arguments(self, kwargs) -> None:
        assert Path(kwargs["model_path"]).resolve() == self.model_path.resolve()
        assert Path(kwargs["recordings_dir"]).resolve() == self.recordings_dir.resolve()
        assert kwargs["overwrite"] is False

    def fit_runner(self, arguments):
        self.fit_arguments = list(arguments)
        output = Path(arguments[arguments.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        output.joinpath("rotation_model.json").write_bytes(
            _ready_artifact(self.model_id)
        )
        return 0

    def run(self, **changes):
        values = {
            "runtime_root": self.runtime_root,
            "recordings_dir": self.recordings_dir,
            "runs_dir": self.runs_dir,
            "deploy_path": self.deploy_path,
            "batch_runner": self.batch_runner,
            "fit_runner": self.fit_runner,
            "bootstrap_runs": 0,
            "force": True,
        }
        values.update(changes)
        return auto.run_auto_calibration(**values)


class AutoCalibrationTests(unittest.TestCase):
    def test_success_runs_fixed_yaw_heading_fit_and_atomically_applies(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PipelineFixture(Path(temporary))
            fixture.deploy_path.parent.mkdir()
            old_payload = b"previous-runtime-artifact\n"
            fixture.deploy_path.write_bytes(old_payload)

            result = fixture.run()

            self.assertEqual(result["status"], "APPLIED")
            active = fixture.deploy_path.read_bytes()
            self.assertNotEqual(active, old_payload)
            auto.validate_deployable_artifact(
                active,
                expected_model_id=fixture.model_id,
                expected_recordings=RECORDINGS,
            )
            previous = fixture.deploy_path.with_name(
                "rotation_model.generated.previous.json"
            )
            self.assertEqual(previous.read_bytes(), old_payload)
            self.assertIn("--angle-column", fixture.fit_arguments)
            self.assertEqual(
                fixture.fit_arguments[
                    fixture.fit_arguments.index("--angle-column") + 1
                ],
                "yaw_deg",
            )
            self.assertEqual(
                fixture.fit_arguments[
                    fixture.fit_arguments.index("--angle-domain") + 1
                ],
                "heading",
            )
            self.assertEqual(
                fixture.fit_arguments[
                    fixture.fit_arguments.index("--angle-period") + 1
                ],
                "360.0",
            )
            self.assertNotIn("--allow-mixed-model-ids", fixture.fit_arguments)
            self.assertNotIn("--allow-logical-command-time", fixture.fit_arguments)
            selected = [
                fixture.fit_arguments[index + 1]
                for index, value in enumerate(fixture.fit_arguments[:-1])
                if value == "--recording-name"
            ]
            self.assertEqual(selected, list(RECORDINGS))
            saved = json.loads(
                Path(result["run_dir"], auto.RUN_MANIFEST_NAME).read_text("utf-8")
            )
            self.assertEqual(saved["status"], "APPLIED")
            self.assertEqual(saved["model_hash"], fixture.model_id)

    def test_fit_rejection_preserves_active_and_previous_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PipelineFixture(Path(temporary))
            fixture.deploy_path.parent.mkdir()
            active = b"known-active-artifact\n"
            previous_path = fixture.deploy_path.with_name(
                "rotation_model.generated.previous.json"
            )
            fixture.deploy_path.write_bytes(active)
            previous_path.write_bytes(b"older-backup\n")

            with self.assertRaisesRegex(auto.AutoCalibrationError, "exit code 2"):
                fixture.run(fit_runner=lambda _arguments: 2)

            self.assertEqual(fixture.deploy_path.read_bytes(), active)
            self.assertEqual(previous_path.read_bytes(), b"older-backup\n")
            manifests = list(fixture.runs_dir.glob(f"*/{auto.RUN_MANIFEST_NAME}"))
            self.assertEqual(len(manifests), 1)
            report = json.loads(manifests[0].read_text("utf-8"))
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["fit_exit_code"], 2)

    def test_hash_mismatched_fit_artifact_never_replaces_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PipelineFixture(Path(temporary))
            fixture.deploy_path.parent.mkdir()
            active = b"known-active-artifact\n"
            fixture.deploy_path.write_bytes(active)

            def wrong_fit(arguments):
                output = Path(arguments[arguments.index("--output-dir") + 1])
                output.mkdir(parents=True, exist_ok=True)
                wrong_id = "sha256:" + "0" * 64
                output.joinpath("rotation_model.json").write_bytes(
                    _ready_artifact(wrong_id)
                )
                return 0

            with self.assertRaisesRegex(
                auto.AutoCalibrationError, "currently loaded .pt model"
            ):
                fixture.run(fit_runner=wrong_fit)
            self.assertEqual(fixture.deploy_path.read_bytes(), active)

    def test_fit_input_csv_mutation_is_detected_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PipelineFixture(Path(temporary))
            fixture.deploy_path.parent.mkdir()
            active = b"known-active-artifact\n"
            previous_path = fixture.deploy_path.with_name(
                "rotation_model.generated.previous.json"
            )
            previous = b"older-backup\n"
            fixture.deploy_path.write_bytes(active)
            previous_path.write_bytes(previous)

            def mutating_fit(arguments):
                code = fixture.fit_runner(arguments)
                results_dir = Path(
                    arguments[arguments.index("--results-dir") + 1]
                )
                suffix = arguments[arguments.index("--results-suffix") + 1]
                result_path = results_dir / f"{RECORDINGS[0]}{suffix}"
                with result_path.open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = list(reader.fieldnames or ())
                    rows = list(reader)
                rows[0]["yaw_deg"] = "123.456"
                with result_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                return code

            with self.assertRaisesRegex(
                auto.AutoCalibrationError, "changed during fitting"
            ):
                fixture.run(fit_runner=mutating_fit)

            self.assertEqual(fixture.deploy_path.read_bytes(), active)
            self.assertEqual(previous_path.read_bytes(), previous)
            manifests = list(fixture.runs_dir.glob(f"*/{auto.RUN_MANIFEST_NAME}"))
            self.assertEqual(len(manifests), 1)
            report = json.loads(manifests[0].read_text("utf-8"))
            self.assertEqual(report["status"], "FAILED")
            self.assertFalse(report["log_validation_match"])
            self.assertNotEqual(
                report["log_validation"],
                report["predeployment_log_validation"],
            )

    def test_publish_failure_rolls_back_active_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PipelineFixture(Path(temporary))
            fixture.deploy_path.parent.mkdir()
            active = b"known-active-artifact\n"
            fixture.deploy_path.write_bytes(active)
            real_replace = auto._atomic_replace_bytes

            def fail_only_deploy(path: Path, payload: bytes):
                if Path(path).resolve() == fixture.deploy_path.resolve():
                    raise OSError("injected publish failure")
                return real_replace(path, payload)

            with mock.patch.object(
                auto, "_atomic_replace_bytes", side_effect=fail_only_deploy
            ):
                with self.assertRaises(auto.AutoCalibrationError):
                    fixture.run()
            self.assertEqual(fixture.deploy_path.read_bytes(), active)

    def test_matching_valid_artifact_skips_expensive_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PipelineFixture(Path(temporary))
            fixture.deploy_path.parent.mkdir()
            fixture.deploy_path.write_bytes(_ready_artifact(fixture.model_id))
            batch = mock.Mock(side_effect=AssertionError("must not run"))

            result = fixture.run(force=False, batch_runner=batch)

            self.assertEqual(result["status"], "ALREADY_APPLIED")
            batch.assert_not_called()

    def test_default_requires_one_model_but_requested_path_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "one.pt"
            first.write_bytes(b"one")
            self.assertEqual(auto.resolve_runtime_model(root), first.resolve())
            second = root / "two.pt"
            second.write_bytes(b"two")
            with self.assertRaisesRegex(auto.AutoCalibrationError, "exactly one"):
                auto.resolve_runtime_model(root)
            self.assertEqual(
                auto.resolve_runtime_model(root, second), second.resolve()
            )

    def test_runtime_validator_import_is_independent_of_process_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_id = auto.sha256_file(Path(__file__))
            artifact_path = root / "rotation_model.json"
            artifact_path.write_bytes(_ready_artifact(model_id))
            program = "\n".join(
                (
                    "import sys",
                    "from pathlib import Path",
                    "sys.path.insert(0, sys.argv[1])",
                    "import auto_calibrate_rotation as target",
                    "target.validate_deployable_artifact(",
                    "    Path(sys.argv[2]).read_bytes(),",
                    "    expected_model_id=sys.argv[3],",
                    ")",
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    program,
                    str(auto.HERE),
                    str(artifact_path),
                    model_id,
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}",
            )


if __name__ == "__main__":
    unittest.main()
