"""Fast-start launcher for FSM v4 visual macro-action control.

Default/--mode real execution loads the pinned Cleanlabel pose model and selected 48-point
endpoint response after identity validation. Missing/mismatched selected
artifacts block startup. The legacy ``--calibrate`` physical fitter is blocked
for this selection. --mode simulation branches to the model-result/virtual-CAN
runtime before hardware imports. Importing this module performs no hardware work.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


DEPTH_CAM_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = DEPTH_CAM_DIR.parent
WORKSPACE_ROOT = RUNTIME_ROOT.parent
RUNTIME_MODEL_FILENAME = "pallet_yolo26n_pose_cleanlabel.pt"
AUTO_CALIBRATION_SCRIPT = (
    WORKSPACE_ROOT / "rotation_fit" / "auto_calibrate_rotation.py"
)
RECORDINGS_DIR = DEPTH_CAM_DIR / "rec"
CALIBRATION_RUNS_DIR = WORKSPACE_ROOT / "rotation_fit" / "out" / "automatic"
GENERATED_ARTIFACT_PATH = (
    DEPTH_CAM_DIR / "calib" / "fsm_v4" / "rotation_endpoint.selected.json"
)


class RotationStartupError(RuntimeError):
    """The rotation response is not safe to use for this FSM process."""


@dataclass(frozen=True)
class RotationCalibrationReady:
    model_path: Path
    model_sha256_id: str
    deploy_path: Path
    artifact_sha256_id: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _configured_runtime_model(
    runtime_root: Path,
    model_filename: str = RUNTIME_MODEL_FILENAME,
) -> Path:
    root = runtime_root.resolve()
    filename = str(model_filename).strip()
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".pt":
        raise RotationStartupError(
            f"invalid configured runtime model filename: {model_filename!r}"
        )
    selected = (root / filename).resolve()
    if not selected.is_file():
        raise RotationStartupError(f"configured runtime model not found: {selected}")
    if selected.stat().st_size <= 0:
        raise RotationStartupError(f"runtime model is empty: {selected}")
    return selected


def _calibration_environment(workspace_root: Path) -> dict[str, str]:
    """Build an environment that does not depend on the caller's sys.path."""

    environment = os.environ.copy()
    root_text = str(workspace_root.resolve())
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        root_text if not inherited else root_text + os.pathsep + inherited
    )
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _strictly_validate_artifact(
    artifact_path: Path,
    expected_model_sha256_id: str,
) -> str:
    """Use the same strict reader as FSM config, before hardware imports."""

    try:
        from calib.fsm_v4.rotation_artifact import (
            load_validated_rotation_artifact,
        )

        response, artifact_id = load_validated_rotation_artifact(
            artifact_path,
            expected_model_sha256_id=expected_model_sha256_id,
            expected_command_strength=30,
            max_rate_deg_s=60.0,
            startup_timeout_sec=3.0,
            max_command_hold_sec=2.5,
        )
    except Exception as exc:
        raise RotationStartupError(
            f"deployed rotation artifact failed runtime validation: {exc}"
        ) from exc
    if getattr(response, "uses_embedded_inertia", False) is not True:
        raise RotationStartupError(
            "deployed response does not contain the fitted inertia model"
        )
    return artifact_id


def run_rotation_calibration_pipeline(
    *,
    runtime_root: Path = RUNTIME_ROOT,
    recordings_dir: Path = RECORDINGS_DIR,
    runs_dir: Path = CALIBRATION_RUNS_DIR,
    deploy_path: Path = GENERATED_ARTIFACT_PATH,
    auto_script: Path = AUTO_CALIBRATION_SCRIPT,
    model_filename: str = RUNTIME_MODEL_FILENAME,
    subprocess_runner: Callable[..., object] = subprocess.run,
    artifact_validator: Optional[Callable[[Path, str], str]] = None,
) -> RotationCalibrationReady:
    """Run/fast-check calibration and return its immutable startup identity."""

    runtime_root = runtime_root.resolve()
    recordings_dir = recordings_dir.resolve()
    runs_dir = runs_dir.resolve()
    deploy_path = deploy_path.resolve()
    auto_script = auto_script.resolve()
    workspace_root = auto_script.parent.parent.resolve()

    if not auto_script.is_file():
        raise RotationStartupError(
            f"automatic calibration entry point not found: {auto_script}"
        )
    if not recordings_dir.is_dir():
        raise RotationStartupError(f"recordings directory not found: {recordings_dir}")

    model_path = _configured_runtime_model(runtime_root, model_filename)
    model_id_before = _sha256_file(model_path)
    command = [
        sys.executable,
        "-u",
        str(auto_script),
        "--runtime-root",
        str(runtime_root),
        "--model",
        str(model_path),
        "--recordings-dir",
        str(recordings_dir),
        "--runs-dir",
        str(runs_dir),
        "--deploy-path",
        str(deploy_path),
    ]
    print(
        "[ROTATION STARTUP] validating or rebuilding the yaw/heading response "
        f"for {model_path.name} ({model_id_before})",
        flush=True,
    )
    try:
        completed = subprocess_runner(
            command,
            cwd=str(workspace_root),
            env=_calibration_environment(workspace_root),
            check=False,
        )
    except OSError as exc:
        raise RotationStartupError(
            f"could not start automatic rotation calibration: {exc}"
        ) from exc
    return_code = getattr(completed, "returncode", None)
    if return_code != 0:
        raise RotationStartupError(
            "automatic rotation calibration failed "
            f"(exit code {return_code}); FSM v4 was not started"
        )

    # Do not trust a zero exit code by itself. Verify that the configured model
    # stayed byte-identical, then independently validate the file
    # that is visible at the final deployment pathname.
    model_after = _configured_runtime_model(runtime_root, model_filename)
    if model_after != model_path:
        raise RotationStartupError("runtime model selection changed during calibration")
    model_id_after = _sha256_file(model_after)
    if model_id_after != model_id_before:
        raise RotationStartupError("runtime .pt model changed during calibration")
    if not deploy_path.is_file():
        raise RotationStartupError(
            "calibration returned success without a deployed rotation artifact"
        )
    validate = artifact_validator or _strictly_validate_artifact
    artifact_id = validate(deploy_path, model_id_after)
    if _sha256_file(model_after) != model_id_after:
        raise RotationStartupError(
            "runtime .pt model changed while validating the deployed artifact"
        )
    return RotationCalibrationReady(
        model_path=model_after,
        model_sha256_id=model_id_after,
        deploy_path=deploy_path,
        artifact_sha256_id=artifact_id,
    )


def _assert_runtime_artifact_active(
    config: object,
    ready: RotationCalibrationReady,
) -> None:
    """Close the publish/import TOCTOU window before camera or CAN startup."""

    configured_model = Path(str(getattr(config, "MODEL_PATH", ""))).resolve()
    if configured_model != ready.model_path:
        raise RotationStartupError(
            "FSM config selected a different inference model after calibration"
        )
    configured_artifact = Path(
        str(getattr(config, "ROT_GENERATED_ARTIFACT_PATH", ""))
    ).resolve()
    if configured_artifact != ready.deploy_path:
        raise RotationStartupError(
            "FSM config selected a different generated rotation artifact"
        )
    if _sha256_file(ready.model_path) != ready.model_sha256_id:
        raise RotationStartupError(
            "runtime .pt model changed before the FSM runtime was imported"
        )
    if getattr(config, "ROT_GENERATED_ARTIFACT_ACTIVE", False) is not True:
        reason = getattr(
            config,
            "ROT_GENERATED_ARTIFACT_FALLBACK_REASON",
            "generated artifact was not activated",
        )
        raise RotationStartupError(
            "FSM rejected the generated rotation artifact: " + str(reason)
        )
    if getattr(config, "ROT_USE_FITTED_RESPONSE", False) is not True:
        raise RotationStartupError(
            "FSM rotation fitting is disabled by ROT_USE_FITTED_RESPONSE"
        )
    if (
        getattr(config, "ROT_GENERATED_ARTIFACT_MODEL_SHA256", "")
        != ready.model_sha256_id
    ):
        raise RotationStartupError(
            "active rotation artifact is bound to a different pose model"
        )
    if (
        getattr(config, "ROT_GENERATED_ARTIFACT_SHA256", "")
        != ready.artifact_sha256_id
    ):
        raise RotationStartupError(
            "rotation artifact changed between validation and runtime import"
        )
    response = getattr(config, "ROTATION_RESPONSE", None)
    if getattr(response, "uses_embedded_inertia", False) is not True:
        raise RotationStartupError(
            "FSM selected the legacy fallback instead of the generated response"
        )


def _load_runtime_components(ready: RotationCalibrationReady):
    """Import hardware-facing runtime code only after calibration succeeds."""

    try:
        from calib.fsm_v4 import config as v4_config

        _assert_runtime_artifact_active(v4_config, ready)

        # These imports can load camera, inference, UI and controller modules;
        # keep them strictly behind the artifact-active gate above.
        from calib.fsm_v4 import CalibrationFSMV4
        from main_rec import main as runtime_main
        from ui.diagram_v4 import draw_fsm_v4_diagram_panel
    except RotationStartupError:
        raise
    except Exception as exc:
        raise RotationStartupError(f"could not load FSM v4 runtime: {exc}") from exc
    return CalibrationFSMV4, v4_config, runtime_main, draw_fsm_v4_diagram_panel


def _load_runtime_components_direct(model_path: Path):
    """Load the runtime immediately, allowing config's built-in response."""

    try:
        from calib.fsm_v4 import config as v4_config

        configured_model = Path(v4_config.MODEL_PATH).resolve()
        if configured_model != model_path.resolve():
            raise RotationStartupError(
                "FSM config selected a different inference model: "
                f"configured={configured_model}, requested={model_path.resolve()}"
            )
        if getattr(v4_config, 'ROT_RESPONSE_MODE', '') == 'selected_delayed_linear':
            from calib.fsm_v4.endpoint_response import load_selected_endpoint
            fresh = load_selected_endpoint(v4_config.ROT_GENERATED_ARTIFACT_PATH, model_path)
            ready = RotationCalibrationReady(model_path.resolve(), fresh.inference_model_sha256_id,
                                             Path(fresh.artifact_path).resolve(), fresh.artifact_sha256_id)
            _assert_runtime_artifact_active(v4_config, ready)
        from calib.fsm_v4 import CalibrationFSMV4
        from main_rec import main as runtime_main
        from ui.diagram_v4 import draw_fsm_v4_diagram_panel
    except RotationStartupError:
        raise
    except Exception as exc:
        raise RotationStartupError(f"could not load FSM v4 runtime: {exc}") from exc
    return CalibrationFSMV4, v4_config, runtime_main, draw_fsm_v4_diagram_panel


def main(
    *,
    mode: str = "real",
    simulation_ipc: bool = False,
    simulation_window: bool = True,
    simulation_port: int = 8766,
    simulation_browser: bool = True,
    require_calibration: bool = False,
    calibration_runner: Optional[Callable[[], RotationCalibrationReady]] = None,
    component_loader: Optional[Callable[[RotationCalibrationReady], tuple]] = None,
) -> int:
    """Start immediately by default; opt into a full calibration when needed."""

    if mode not in ("real", "simulation"):
        raise ValueError("mode must be real or simulation")
    if mode == "simulation":
        if require_calibration or calibration_runner is not None or component_loader is not None:
            raise ValueError("Simulation mode does not run physical calibration or the real runtime loader")
        # Branch before main_rec, camera, inference and hardware startup imports.
        from simulation_runtime import run_simulation
        return run_simulation(ipc=simulation_ipc,show_window=simulation_window,
                              port=simulation_port,open_browser=simulation_browser)

    if require_calibration and calibration_runner is None:
        print('[STARTUP BLOCKED] Selected endpoint response is pinned; --calibrate uses the old physical fitter. Run offline fitting explicitly, then select a new endpoint artifact.', file=sys.stderr)
        return 2

    strict_start = bool(
        require_calibration
        or calibration_runner is not None
        or component_loader is not None
    )
    if strict_start:
        try:
            ready = (calibration_runner or run_rotation_calibration_pipeline)()
            loader = component_loader or _load_runtime_components
            CalibrationFSMV4, v4_config, runtime_main, diagram_drawer = loader(ready)
        except RotationStartupError as exc:
            print(f"[ROTATION STARTUP BLOCKED] {exc}", file=sys.stderr, flush=True)
            return 2

        print(
            "[ROTATION STARTUP] generated response is active; starting FSM v4 ",
            f"(model={ready.model_path.name}, "
            f"model_hash={ready.model_sha256_id}, "
            f"artifact={ready.artifact_sha256_id})",
            flush=True,
        )
    else:
        try:
            model_path = _configured_runtime_model(RUNTIME_ROOT)
            model_id = _sha256_file(model_path)
            (
                CalibrationFSMV4,
                v4_config,
                runtime_main,
                diagram_drawer,
            ) = _load_runtime_components_direct(model_path)
        except RotationStartupError as exc:
            print(f"[FSM v4 STARTUP BLOCKED] {exc}", file=sys.stderr, flush=True)
            return 2

        response_mode = (
            "generated artifact"
            if v4_config.ROT_GENERATED_ARTIFACT_ACTIVE
            else "built-in fitted fallback"
        )
        print(
            "[FSM v4 FAST START] starting without automatic recalibration "
            f"(model={model_path.name}, model_hash={model_id}, "
            f"rotation_response={response_mode})",
            flush=True,
        )
        if not v4_config.ROT_GENERATED_ARTIFACT_ACTIVE:
            print(
                "[FSM v4 FAST START] calibration artifact unavailable; "
                f"fallback reason: {v4_config.ROT_GENERATED_ARTIFACT_FALLBACK_REASON}",
                flush=True,
            )

    runtime_main(
        fsm_factory=CalibrationFSMV4,
        recording_prefix="forklift_v4_recording",
        window_title="[REAL] Forklift HUD + FSM v4 visual macro control",
        fsm_version="v4",
        diagram_drawer=diagram_drawer,
        can_enabled=v4_config.CAN_ENABLED,
        camera_enabled=getattr(v4_config, "CAMERA_ENABLED", None),
        camera_display_scale=v4_config.CAMERA_DISPLAY_SCALE,
        debug_step_mode=v4_config.DEBUG_STEP_MODE,
        interface_panel_width=v4_config.INTERFACE_PANEL_WIDTH,
        fsm_panel_width=v4_config.FSM_PANEL_WIDTH,
    )
    return 0


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="FSM v4 real/simulation launcher")
    parser.add_argument('--mode',choices=('real','simulation'),default='real')
    parser.add_argument('--calibrate',action='store_true')
    parser.add_argument('--ipc',action='store_true',help='Simulation worker: model/CAN JSON over stdin/stdout')
    parser.add_argument('--no-window',action='store_true',help='Simulation worker without its native result window')
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    if args.mode=='real' and (args.ipc or args.no_window or args.no_browser or args.port!=8766):
        parser.error('IPC/window/server options require --mode simulation')
    if args.mode=='simulation' and args.calibrate:
        parser.error('--calibrate is not available in simulation mode')
    raise SystemExit(main(mode=args.mode,require_calibration=args.calibrate,
        simulation_ipc=args.ipc,simulation_window=not args.no_window,
        simulation_port=args.port,simulation_browser=not args.no_browser))
