"""Read-only preflight check for the standalone FSM v4 bundle."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPTH_CAM = ROOT / "depth_cam"
ROTATION_FIT = ROOT.parent / "rotation_fit"
AUTO_CALIBRATION_SCRIPT = ROTATION_FIT / "auto_calibrate_rotation.py"
AUTO_CALIBRATION_FILES = (
    AUTO_CALIBRATION_SCRIPT,
    ROTATION_FIT / "batch_infer_rotation_logs.py",
    ROTATION_FIT / "fit_piecewise_rotation_model.py",
    ROTATION_FIT / "rotation_log_fit.py",
    ROTATION_FIT / "piecewise_rotation_model.py",
)

REQUIRED_FILES = (
    DEPTH_CAM / "main_rec.py",
    DEPTH_CAM / "main_rec_v4.py",
    DEPTH_CAM / "calib" / "config.py",
    DEPTH_CAM / "calib" / "control.py",
    DEPTH_CAM / "calib" / "fsm" / "commands.py",
    DEPTH_CAM / "calib" / "fsm_v4" / "config.py",
    DEPTH_CAM / "calib" / "fsm_v4" / "rotation_artifact.py",
    DEPTH_CAM / "calib" / "fsm_v4" / "top.py",
    DEPTH_CAM / "ui" / "diagram.py",
    DEPTH_CAM / "ui" / "diagram_v4.py",
)

REQUIRED_MODULES = (
    "numpy",
    "cv2",
    "pyrealsense2",
    "torch",
    "ultralytics",
)


def version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project: {ROOT}")
    failures: list[str] = []

    print("\n[Files]")
    model_candidates = sorted(ROOT.glob("*.pt"))
    if len(model_candidates) == 1 and model_candidates[0].stat().st_size > 0:
        expected_model = model_candidates[0].resolve()
        print(f"OK   model: {model_candidates[0].name}")
    else:
        expected_model = None
        names = ", ".join(path.name for path in model_candidates) or "(none)"
        print(f"FAIL expected exactly one .pt model, found: {names}")
        failures.append("project root must contain exactly one non-empty .pt model")
    for path in REQUIRED_FILES:
        ok = path.is_file() and path.stat().st_size > 0
        print(f"{'OK  ' if ok else 'FAIL'} {path.relative_to(ROOT)}")
        if not ok:
            failures.append(f"missing file: {path.relative_to(ROOT)}")
    for path in AUTO_CALIBRATION_FILES:
        ok = path.is_file() and path.stat().st_size > 0
        label = path.relative_to(ROOT.parent)
        print(f"{'OK  ' if ok else 'FAIL'} {label}")
        if not ok:
            failures.append(f"missing file: {label}")

    print("\n[Python modules]")
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
            print(f"OK   {name}: {version}")
            if name == "ultralytics":
                installed = importlib.metadata.version("ultralytics")
                if version_tuple(installed) < (8, 4, 60):
                    print("FAIL ultralytics must be >= 8.4.60 for YOLO26 Pose")
                    failures.append("ultralytics version is below 8.4.60")
        except Exception as exc:
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            failures.append(f"module import failed: {name}")

    if str(DEPTH_CAM) not in sys.path:
        sys.path.insert(0, str(DEPTH_CAM))

    print("\n[FSM imports]")
    try:
        from calib import config as common_config

        configured_model = Path(common_config.MODEL_PATH).resolve()
        if expected_model is None or configured_model != expected_model:
            raise RuntimeError(f"unexpected model path: {configured_model}")
        print(f"OK   common model path: {configured_model.name}")

    except Exception as exc:
        print(f"FAIL common config: {type(exc).__name__}: {exc}")
        failures.append("common config/model path check failed")

    try:
        # Package import is required because fsm_v4.config uses relative
        # imports (for example .rotation_model).
        v4_config = importlib.import_module("calib.fsm_v4.config")

        v4_config.validate()
        print("OK   calib.fsm_v4.config.validate()")
        if v4_config.CAN_ENABLED:
            try:
                can_module = importlib.import_module("canlib")
                version = getattr(can_module, "__version__", "installed")
                print(f"OK   CAN ON / canlib: {version}")
            except Exception as can_exc:
                print(f"FAIL CAN ON / canlib: {type(can_exc).__name__}: {can_exc}")
                failures.append("CAN is enabled but canlib import failed")
        else:
            print("OK   CAN OFF monitor mode (canlib/Kvaser not required)")
    except Exception as exc:
        print(f"FAIL v4 config: {type(exc).__name__}: {exc}")
        failures.append("v4 config validation failed")

    try:
        from calib.fsm_v4 import CalibrationFSMV4  # noqa: F401

        print("OK   CalibrationFSMV4 import")
    except Exception as exc:
        print(f"FAIL CalibrationFSMV4: {type(exc).__name__}: {exc}")
        failures.append("CalibrationFSMV4 import failed")

    print()
    if failures:
        print("Preflight FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Preflight OK. Hardware connection and calibration are not verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
