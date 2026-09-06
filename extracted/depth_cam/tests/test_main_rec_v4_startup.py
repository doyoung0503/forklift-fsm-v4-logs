from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


DEPTH_CAM_DIR = Path(__file__).resolve().parents[1]
if str(DEPTH_CAM_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_CAM_DIR))

import main_rec_v4 as launcher  # noqa: E402


class RotationStartupTests(unittest.TestCase):
    def test_default_runtime_model_is_pinned_even_with_other_weights_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configured = root / launcher.RUNTIME_MODEL_FILENAME
            configured.write_bytes(b"configured-v4-model")
            (root / "other_pose.pt").write_bytes(b"other-model")

            selected = launcher._configured_runtime_model(root)

            self.assertEqual(selected, configured.resolve())

    def test_pipeline_runs_before_returning_a_validated_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "extracted"
            recordings = runtime / "depth_cam" / "rec"
            runs = root / "rotation_fit" / "out" / "automatic"
            script = root / "rotation_fit" / "auto_calibrate_rotation.py"
            deploy = runtime / "depth_cam" / "calib" / "fsm_v4" / "rotation_model.generated.json"
            recordings.mkdir(parents=True)
            script.parent.mkdir(parents=True)
            script.write_text("# test entry point\n", encoding="utf-8")
            model = runtime / "pose.pt"
            model.write_bytes(b"model-v1")
            calls = []

            def run_process(command, **kwargs):
                calls.append((command, kwargs))
                deploy.parent.mkdir(parents=True)
                deploy.write_bytes(b"validated artifact")
                return SimpleNamespace(returncode=0)

            ready = launcher.run_rotation_calibration_pipeline(
                runtime_root=runtime,
                recordings_dir=recordings,
                runs_dir=runs,
                deploy_path=deploy,
                auto_script=script,
                model_filename="pose.pt",
                subprocess_runner=run_process,
                artifact_validator=lambda path, model_id: "sha256:" + "a" * 64,
            )

            self.assertEqual(len(calls), 1)
            command, options = calls[0]
            self.assertEqual(command[0], sys.executable)
            self.assertIn("--model", command)
            self.assertIn(str(model.resolve()), command)
            self.assertEqual(Path(options["cwd"]), root.resolve())
            self.assertFalse(options["check"])
            self.assertEqual(ready.model_path, model.resolve())
            self.assertTrue(ready.model_sha256_id.startswith("sha256:"))

    def test_zero_exit_without_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            recordings = root / "recordings"
            script = root / "rotation_fit" / "auto_calibrate_rotation.py"
            runtime.mkdir()
            recordings.mkdir()
            script.parent.mkdir()
            script.write_text("# test\n", encoding="utf-8")
            (runtime / "pose.pt").write_bytes(b"model")

            with self.assertRaisesRegex(
                launcher.RotationStartupError, "without a deployed"
            ):
                launcher.run_rotation_calibration_pipeline(
                    runtime_root=runtime,
                    recordings_dir=recordings,
                    runs_dir=root / "runs",
                    deploy_path=root / "missing.json",
                    auto_script=script,
                    model_filename="pose.pt",
                    subprocess_runner=lambda *args, **kwargs: SimpleNamespace(
                        returncode=0
                    ),
                )

    def test_failed_calibration_never_loads_runtime_components(self):
        calibration = mock.Mock(
            side_effect=launcher.RotationStartupError("fit validation failed")
        )
        components = mock.Mock(side_effect=AssertionError("must not load"))

        code = launcher.main(
            calibration_runner=calibration,
            component_loader=components,
        )

        self.assertEqual(code, 2)
        components.assert_not_called()

    def test_default_main_starts_runtime_without_calibration(self):
        runtime_main = mock.Mock()
        config = SimpleNamespace(
            MODEL_PATH=str(
                launcher.RUNTIME_ROOT / launcher.RUNTIME_MODEL_FILENAME
            ),
            ROT_GENERATED_ARTIFACT_ACTIVE=False,
            ROT_GENERATED_ARTIFACT_FALLBACK_REASON="artifact missing",
            CAN_ENABLED=False,
            CAMERA_DISPLAY_SCALE=1.0,
            DEBUG_STEP_MODE=False,
            INTERFACE_PANEL_WIDTH=550,
            FSM_PANEL_WIDTH=620,
        )
        components = (object(), config, runtime_main, object())
        model_path = launcher.RUNTIME_ROOT / launcher.RUNTIME_MODEL_FILENAME

        with mock.patch.object(
            launcher, "_configured_runtime_model", return_value=model_path
        ), mock.patch.object(
            launcher, "_sha256_file", return_value="sha256:" + "c" * 64
        ), mock.patch.object(
            launcher, "_load_runtime_components_direct", return_value=components
        ), mock.patch.object(
            launcher, "run_rotation_calibration_pipeline"
        ) as calibration:
            code = launcher.main()

        self.assertEqual(code, 0)
        calibration.assert_not_called()
        runtime_main.assert_called_once()

    def test_runtime_import_must_activate_the_same_artifact_and_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "pose.pt"
            artifact = root / "rotation_model.generated.json"
            model.write_bytes(b"model")
            artifact.write_bytes(b"artifact")
            ready = launcher.RotationCalibrationReady(
                model_path=model.resolve(),
                model_sha256_id=launcher._sha256_file(model),
                deploy_path=artifact.resolve(),
                artifact_sha256_id="sha256:" + "b" * 64,
            )
            good = SimpleNamespace(
                MODEL_PATH=str(model),
                ROT_GENERATED_ARTIFACT_PATH=artifact,
                ROT_GENERATED_ARTIFACT_ACTIVE=True,
                ROT_GENERATED_ARTIFACT_MODEL_SHA256=ready.model_sha256_id,
                ROT_GENERATED_ARTIFACT_SHA256=ready.artifact_sha256_id,
                ROT_GENERATED_ARTIFACT_FALLBACK_REASON="",
                ROT_USE_FITTED_RESPONSE=True,
                ROTATION_RESPONSE=SimpleNamespace(uses_embedded_inertia=True),
            )
            launcher._assert_runtime_artifact_active(good, ready)

            bad = SimpleNamespace(**vars(good))
            bad.ROT_GENERATED_ARTIFACT_ACTIVE = False
            bad.ROT_GENERATED_ARTIFACT_FALLBACK_REASON = "hash mismatch"
            with self.assertRaisesRegex(
                launcher.RotationStartupError, "hash mismatch"
            ):
                launcher._assert_runtime_artifact_active(bad, ready)

            disabled = SimpleNamespace(**vars(good))
            disabled.ROT_USE_FITTED_RESPONSE = False
            with self.assertRaisesRegex(
                launcher.RotationStartupError, "ROT_USE_FITTED_RESPONSE"
            ):
                launcher._assert_runtime_artifact_active(disabled, ready)


if __name__ == "__main__":
    unittest.main()
