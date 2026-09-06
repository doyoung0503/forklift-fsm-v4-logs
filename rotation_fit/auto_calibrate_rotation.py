#!/usr/bin/env python3
"""End-to-end, fail-closed rotation calibration for a newly loaded pose model.

The automatic pipeline is intentionally a *startup* operation, not a hot
reload.  It runs inference into a private run directory, validates the derived
logs, invokes the timestamp-aligned yaw/heading fitter, validates the emitted
artifact again, and only then atomically replaces the runtime artifact.

The raw recordings and their timing/control logs are never modified.  A failed
stage leaves the currently deployed artifact byte-for-byte unchanged and keeps
the run directory as an audit record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import secrets
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

try:
    from .piecewise_rotation_model import LogRotationModel
except ImportError:  # direct script execution
    from piecewise_rotation_model import LogRotationModel


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parent
DEFAULT_RUNTIME_ROOT = WORKSPACE_ROOT / "extracted"
DEFAULT_RECORDINGS_DIR = DEFAULT_RUNTIME_ROOT / "depth_cam" / "rec"
DEFAULT_RUNS_DIR = HERE / "out" / "automatic"
DEFAULT_DEPLOY_PATH = (
    DEFAULT_RUNTIME_ROOT
    / "depth_cam"
    / "calib"
    / "fsm_v4"
    / "rotation_model.generated.json"
)

DEFAULT_RESULTS_SUFFIX = "_new_pose.csv"
RUN_MANIFEST_NAME = "auto_calibration_manifest.json"
BATCH_MANIFEST_NAME = "batch_inference_manifest.json"
MODEL_ID_PREFIX = "sha256:"
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024

BatchRunner = Callable[..., Mapping[str, Any]]
FitRunner = Callable[[Optional[list[str]]], int]


class AutoCalibrationError(RuntimeError):
    """A pipeline stage failed and the active artifact was not changed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return MODEL_ID_PREFIX + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return MODEL_ID_PREFIX + digest.hexdigest()


def resolve_runtime_model(
    runtime_root: Path,
    requested_model: Optional[Path] = None,
) -> Path:
    """Resolve the exact requested runtime model, or the sole model by default."""

    root = runtime_root.resolve()
    if not root.is_dir():
        raise AutoCalibrationError(f"runtime root not found: {root}")
    if requested_model is not None:
        selected = requested_model.resolve()
        if selected.parent != root or selected.suffix.lower() != ".pt":
            raise AutoCalibrationError(
                "requested model must be a .pt file directly inside the runtime "
                f"root: requested={selected}, runtime_root={root}"
            )
        if not selected.is_file():
            raise AutoCalibrationError(f"requested runtime model not found: {selected}")
        if selected.stat().st_size <= 0:
            raise AutoCalibrationError(f"runtime model is empty: {selected}")
        return selected

    candidates = sorted(path.resolve() for path in root.glob("*.pt"))
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "(none)"
        raise AutoCalibrationError(
            f"expected exactly one runtime .pt model in {root}, found "
            f"{len(candidates)}: {names}"
        )
    selected = candidates[0]
    if selected.stat().st_size <= 0:
        raise AutoCalibrationError(f"runtime model is empty: {selected}")
    return selected


def _plain_recording_prefix(value: object) -> str:
    prefix = str(value).strip()
    if (
        not prefix
        or Path(prefix).name != prefix
        or "/" in prefix
        or "\\" in prefix
        or prefix in {".", ".."}
    ):
        raise AutoCalibrationError(f"unsafe recording prefix: {value!r}")
    return prefix


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AutoCalibrationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise AutoCalibrationError(f"invalid JSON numeric constant: {value}")


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AutoCalibrationError(f"{location} must be a JSON object")
    return value


def _integer(value: object, location: str) -> int:
    if type(value) is not int:
        raise AutoCalibrationError(f"{location} must be a JSON integer")
    return value


def _finite(value: object, location: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise AutoCalibrationError(f"{location} must be a JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise AutoCalibrationError(f"{location} must be finite")
    return number


def validate_deployable_artifact(
    payload: bytes,
    *,
    expected_model_id: str,
    expected_recordings: Optional[Sequence[str]] = None,
    expected_command_strength: int = 30,
    maximum_rate_deg_s: float = 60.0,
    startup_timeout_sec: float = 3.0,
    maximum_hold_sec: float = 2.5,
) -> tuple[dict[str, Any], LogRotationModel, str]:
    """Validate both the numerical model and its control-safety evidence."""

    if not isinstance(payload, bytes):
        raise TypeError("artifact payload must be bytes")
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise AutoCalibrationError(
            f"rotation artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
        )
    if not (
        isinstance(expected_model_id, str)
        and len(expected_model_id) == len(MODEL_ID_PREFIX) + 64
        and expected_model_id.startswith(MODEL_ID_PREFIX)
        and all(ch in "0123456789abcdef" for ch in expected_model_id[7:])
    ):
        raise AutoCalibrationError(
            "expected model id must be sha256:<64 lowercase hex>"
        )

    # LogRotationModel performs the strict schema/parameter/range checks,
    # including duplicate-key and acceleration-boundary validation.
    try:
        model = LogRotationModel.from_json(payload)
    except (TypeError, ValueError) as exc:
        raise AutoCalibrationError(f"invalid fitted model: {exc}") from exc
    try:
        artifact = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AutoCalibrationError(f"invalid artifact JSON: {exc}") from exc
    artifact = dict(_mapping(artifact, "artifact"))
    report = _mapping(artifact.get("fit_report"), "fit_report")

    if report.get("status") != "READY":
        raise AutoCalibrationError("fit_report.status must be 'READY'")
    if report.get("safe_for_control") is not True:
        raise AutoCalibrationError("fit_report.safe_for_control must be true")
    blockers = report.get("deployment_blockers")
    if type(blockers) is not list or blockers:
        raise AutoCalibrationError(
            "fit_report.deployment_blockers must be an empty array"
        )
    if report.get("model_type") != (
        "deadtime_constant_acceleration_cruise_linear_inertia"
    ):
        raise AutoCalibrationError("unsupported fit_report.model_type")
    if _integer(report.get("command_strength"), "fit_report.command_strength") != (
        expected_command_strength
    ):
        raise AutoCalibrationError("fit command strength does not match runtime")
    if model.command_strength != expected_command_strength:
        raise AutoCalibrationError("model command strength does not match runtime")
    if report.get("angle_column") != "yaw_deg" or report.get(
        "angle_domain"
    ) != "heading":
        raise AutoCalibrationError("fit angle contract must be yaw_deg/heading")
    if _finite(report.get("angle_period_deg"), "fit_report.angle_period_deg") != 360.0:
        raise AutoCalibrationError("fit angle period must be 360 degrees")

    settings = _mapping(report.get("settings"), "fit_report.settings")
    if _integer(
        settings.get("command_strength"),
        "fit_report.settings.command_strength",
    ) != expected_command_strength:
        raise AutoCalibrationError(
            "fit settings command strength does not match runtime"
        )
    if _finite(
        settings.get("angle_period_deg"),
        "fit_report.settings.angle_period_deg",
    ) != 360.0:
        raise AutoCalibrationError("fit settings angle period must be 360 degrees")
    if settings.get("require_actual_can_timing") is not True:
        raise AutoCalibrationError("fit must require actual CAN timing")
    if settings.get("can_time_point") != "return":
        raise AutoCalibrationError("fit must use CAN write-return timestamps")
    inertia_horizon = _finite(
        settings.get("inertia_horizon_sec"),
        "fit_report.settings.inertia_horizon_sec",
    )
    if inertia_horizon <= 0.0:
        raise AutoCalibrationError("inertia horizon must be positive")

    source = _mapping(report.get("source"), "fit_report.source")
    if source.get("angle_column") != "yaw_deg" or source.get(
        "angle_domain"
    ) != "heading":
        raise AutoCalibrationError("source angle contract must be yaw_deg/heading")
    model_ids = source.get("model_ids")
    if model_ids != [expected_model_id]:
        raise AutoCalibrationError(
            "fit artifact is not bound to the currently loaded .pt model"
        )
    if source.get("mixed_model_ids_detected") is not False:
        raise AutoCalibrationError("mixed inference model ids are not allowed")
    if _mapping(source.get("recording_errors"), "source.recording_errors"):
        raise AutoCalibrationError("selected recording inputs contain errors")
    unidentified = source.get("unidentified_model_recordings")
    if type(unidentified) is not list or unidentified:
        raise AutoCalibrationError("recordings without model identity are not allowed")
    sign_audit = _mapping(
        source.get("direction_sign_audit"),
        "fit_report.source.direction_sign_audit",
    )
    ambiguous = sign_audit.get("ambiguous_directions")
    if type(ambiguous) is not list or ambiguous:
        raise AutoCalibrationError("rotation direction signs are ambiguous")
    if sign_audit.get("left_right_modal_signs_are_opposite") is not True:
        raise AutoCalibrationError("LEFT and RIGHT signs must be opposite")

    if expected_recordings is not None:
        expected = sorted({_plain_recording_prefix(name) for name in expected_recordings})
        selected_raw = source.get("recordings_selected")
        if type(selected_raw) is not list:
            raise AutoCalibrationError("fit report lacks recordings_selected")
        selected = sorted({_plain_recording_prefix(name) for name in selected_raw})
        if selected != expected:
            raise AutoCalibrationError(
                "fit report recording selection differs from validated batch"
            )
        loaded = _integer(source.get("recordings_loaded"), "source.recordings_loaded")
        if loaded != len(expected):
            raise AutoCalibrationError(
                "not every validated batch recording was loaded by the fitter"
            )

    identifiability = _mapping(
        report.get("identifiability"), "fit_report.identifiability"
    )
    for name in ("dead_time", "acceleration_and_cruise", "linear_inertia"):
        if identifiability.get(name) is not True:
            raise AutoCalibrationError(f"fit component {name!r} is not identifiable")

    validation = _mapping(report.get("validation"), "fit_report.validation")
    folds = _mapping(
        validation.get("leave_one_recording_out_folds"),
        "fit_report.validation.leave_one_recording_out_folds",
    )
    attempted = _integer(folds.get("attempted"), "LOO attempted folds")
    successful = _integer(folds.get("successful"), "LOO successful folds")
    minimum = _integer(settings.get("min_loo_predictions"), "minimum LOO folds")
    threshold = _finite(settings.get("max_loo_rmse_deg"), "LOO RMSE threshold")
    rmse = _finite(folds.get("equal_recording_weight_rmse_deg"), "LOO RMSE")
    if (
        minimum < 1
        or threshold <= 0.0
        or attempted != successful
        or successful < minimum
        or rmse > threshold
    ):
        raise AutoCalibrationError("leave-one-recording-out validation did not pass")

    if model.dead_time_sec < 0.0 or model.dead_time_sec >= startup_timeout_sec:
        raise AutoCalibrationError("fitted dead time is outside runtime limits")
    if model.max_rate_deg_s <= 0.0 or model.max_rate_deg_s > maximum_rate_deg_s:
        raise AutoCalibrationError("fitted maximum rate is outside runtime limits")
    if model.fitted_min_hold_sec > min(model.fitted_max_hold_sec, maximum_hold_sec):
        raise AutoCalibrationError("no fitted hold duration is inside runtime limits")

    # Use the exact validator that will accept/reject the file on the next FSM
    # process start as the final compatibility authority.  The lazy fsm_v4
    # package import has no config/model-selection side effect.
    workspace_import_root = str(WORKSPACE_ROOT)
    inserted_workspace_root = workspace_import_root not in sys.path
    if inserted_workspace_root:
        # Direct execution from ``extracted`` (the Windows launcher working
        # directory) otherwise exposes rotation_fit/, but not its sibling
        # ``extracted`` package, on sys.path.
        sys.path.insert(0, workspace_import_root)
    try:
        from extracted.depth_cam.calib.fsm_v4.rotation_artifact import (
            validate_rotation_artifact_bytes as runtime_validate,
        )

        runtime_validate(
            payload,
            expected_model_sha256_id=expected_model_id,
            expected_command_strength=expected_command_strength,
            max_rate_deg_s=maximum_rate_deg_s,
            startup_timeout_sec=startup_timeout_sec,
            max_command_hold_sec=maximum_hold_sec,
        )
    except ImportError as exc:
        raise AutoCalibrationError(
            "cannot import the FSM runtime artifact validator"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise AutoCalibrationError(
            f"FSM runtime rejected the fitted artifact: {exc}"
        ) from exc
    finally:
        if inserted_workspace_root:
            try:
                sys.path.remove(workspace_import_root)
            except ValueError:
                pass
    return artifact, model, _sha256_bytes(payload)


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    """Durably write a sibling temporary file and atomically replace ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_replace_bytes(path, payload)


def _previous_path(deploy_path: Path) -> Path:
    return deploy_path.with_name(
        f"{deploy_path.stem}.previous{deploy_path.suffix}"
    )


def promote_artifact(
    candidate_path: Path,
    deploy_path: Path,
    *,
    expected_model_id: str,
    expected_recordings: Sequence[str],
) -> dict[str, Any]:
    """Atomically promote one validated artifact, retaining one rollback copy."""

    candidate = candidate_path.read_bytes()
    _, _, candidate_id = validate_deployable_artifact(
        candidate,
        expected_model_id=expected_model_id,
        expected_recordings=expected_recordings,
    )
    old_payload = deploy_path.read_bytes() if deploy_path.is_file() else None
    previous = _previous_path(deploy_path)
    previous_existed = previous.is_file()
    prior_previous_payload = previous.read_bytes() if previous_existed else None
    previous_id = ""
    if old_payload is not None:
        previous_id = _sha256_bytes(old_payload)
        _atomic_replace_bytes(previous, old_payload)

    try:
        _atomic_replace_bytes(deploy_path, candidate)
        visible = deploy_path.read_bytes()
        if visible != candidate:
            raise AutoCalibrationError("deployed artifact bytes changed during publish")
        validate_deployable_artifact(
            visible,
            expected_model_id=expected_model_id,
            expected_recordings=expected_recordings,
        )
    except BaseException:
        # A replace failure itself leaves the old file intact.  If a later
        # verification failed, restore the captured previous bytes.
        if old_payload is not None:
            _atomic_replace_bytes(deploy_path, old_payload)
        elif deploy_path.exists():
            deploy_path.unlink()
        if prior_previous_payload is not None:
            _atomic_replace_bytes(previous, prior_previous_payload)
        elif not previous_existed and previous.exists():
            previous.unlink()
        raise

    return {
        "deploy_path": str(deploy_path.resolve()),
        "artifact_sha256": candidate_id,
        "previous_path": str(previous.resolve()) if old_payload is not None else "",
        "previous_sha256": previous_id,
    }


def _validate_batch_manifest(
    raw_manifest: Mapping[str, Any],
    *,
    expected_model_id: str,
    results_dir: Path,
    results_suffix: str,
) -> tuple[dict[str, Any], list[str]]:
    manifest = dict(raw_manifest)
    if str(manifest.get("status", "")).lower() not in {"complete", "ready"}:
        raise AutoCalibrationError("batch inference did not report completion")
    if manifest.get("model_hash") != expected_model_id:
        raise AutoCalibrationError("batch inference model hash mismatch")
    if manifest.get("angle_column") != "yaw_deg" or manifest.get(
        "angle_domain"
    ) != "heading":
        raise AutoCalibrationError("batch angle contract must be yaw_deg/heading")
    if manifest.get("model_id_column") != "model_hash":
        raise AutoCalibrationError("batch model-id column must be model_hash")
    if manifest.get("valid_column") not in {"pose_ok", "valid"}:
        raise AutoCalibrationError("batch validity column must be pose_ok or valid")
    if manifest.get("results_suffix") != results_suffix:
        raise AutoCalibrationError("batch result suffix differs from requested suffix")
    reported_results_dir = manifest.get("results_dir")
    if reported_results_dir and Path(str(reported_results_dir)).resolve() != results_dir:
        raise AutoCalibrationError("batch results directory mismatch")

    prefixes_raw = manifest.get("successful_recording_prefixes")
    if prefixes_raw is None:
        # Compatibility with an early batch API: completed recording reports
        # imply success.  The final batch module exposes the explicit field.
        records = manifest.get("recordings")
        if type(records) is not list:
            raise AutoCalibrationError(
                "batch manifest lacks successful_recording_prefixes"
            )
        prefixes_raw = [
            item.get("prefix")
            for item in records
            if isinstance(item, Mapping) and item.get("eligible", True) is not False
        ]
    if type(prefixes_raw) is not list:
        raise AutoCalibrationError("successful_recording_prefixes must be an array")
    prefixes = [_plain_recording_prefix(item) for item in prefixes_raw]
    if not prefixes or len(prefixes) != len(set(prefixes)):
        raise AutoCalibrationError(
            "batch must contain at least one unique successful recording"
        )
    if _integer(manifest.get("recording_count"), "batch recording_count") != len(
        prefixes
    ):
        raise AutoCalibrationError("batch recording_count does not match successes")
    reports = manifest.get("recordings")
    if type(reports) is not list:
        raise AutoCalibrationError("batch recordings must be an array")
    report_prefixes = sorted(
        _plain_recording_prefix(item.get("prefix"))
        for item in reports
        if isinstance(item, Mapping)
    )
    if report_prefixes != sorted(prefixes):
        raise AutoCalibrationError(
            "batch recording reports do not match successful prefixes"
        )
    return manifest, sorted(prefixes)


def validate_inference_results(
    results_dir: Path,
    *,
    prefixes: Sequence[str],
    results_suffix: str,
    expected_model_id: str,
) -> dict[str, Any]:
    """Independently validate the published per-frame inference logs."""

    required = {
        "frame_i", "yaw_deg", "pose_ok", "valid", "confidence",
        "model_hash", "angle_domain", "angle_source",
    }
    per_recording: dict[str, Any] = {}
    total_rows = 0
    total_valid = 0
    for prefix in prefixes:
        path = results_dir / f"{_plain_recording_prefix(prefix)}{results_suffix}"
        if not path.is_file():
            raise AutoCalibrationError(f"missing inference result: {path.name}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = sorted(required - fields)
            if missing:
                raise AutoCalibrationError(
                    f"{path.name} lacks columns: {', '.join(missing)}"
                )
            previous_frame: Optional[int] = None
            rows = 0
            valid = 0
            for line_number, row in enumerate(reader, start=2):
                rows += 1
                try:
                    frame_number = int(row["frame_i"])
                except (TypeError, ValueError) as exc:
                    raise AutoCalibrationError(
                        f"invalid frame_i at {path.name}:{line_number}"
                    ) from exc
                if previous_frame is not None and frame_number <= previous_frame:
                    raise AutoCalibrationError(
                        f"frame_i is not strictly increasing in {path.name}"
                    )
                previous_frame = frame_number
                if row.get("model_hash") != expected_model_id:
                    raise AutoCalibrationError(
                        f"model hash mismatch at {path.name}:{line_number}"
                    )
                if row.get("angle_domain") != "heading":
                    raise AutoCalibrationError(
                        f"non-heading inference row at {path.name}:{line_number}"
                    )
                if row.get("angle_source") != "pnp_face_orientation":
                    raise AutoCalibrationError(
                        f"non-PnP angle source at {path.name}:{line_number}"
                    )
                pose_ok = str(row.get("pose_ok", "")).strip().lower() in {
                    "1", "true", "yes", "ok"
                }
                valid_flag = str(row.get("valid", "")).strip().lower() in {
                    "1", "true", "yes", "ok"
                }
                if valid_flag != pose_ok:
                    raise AutoCalibrationError(
                        f"valid/pose_ok mismatch at {path.name}:{line_number}"
                    )
                if pose_ok:
                    try:
                        yaw = float(row.get("yaw_deg", ""))
                    except (TypeError, ValueError) as exc:
                        raise AutoCalibrationError(
                            f"valid pose lacks yaw_deg at {path.name}:{line_number}"
                        ) from exc
                    if not math.isfinite(yaw):
                        raise AutoCalibrationError(
                            f"non-finite yaw_deg at {path.name}:{line_number}"
                        )
                    valid += 1
            if rows < 3 or valid < 3:
                raise AutoCalibrationError(
                    f"{path.name} has insufficient rows/valid poses ({rows}/{valid})"
                )
        digest = sha256_file(path)
        per_recording[prefix] = {
            "path": str(path.resolve()),
            "rows": rows,
            "valid_poses": valid,
            "sha256": digest,
        }
        total_rows += rows
        total_valid += valid
    return {
        "recording_count": len(prefixes),
        "total_rows": total_rows,
        "total_valid_poses": total_valid,
        "recordings": per_recording,
    }


def _default_batch_runner() -> BatchRunner:
    try:
        from .batch_infer_rotation_logs import run_batch_inference
    except ImportError:
        from batch_infer_rotation_logs import run_batch_inference
    return run_batch_inference


def _default_fit_runner() -> FitRunner:
    try:
        from .fit_piecewise_rotation_model import main as fit_main
    except ImportError:
        from fit_piecewise_rotation_model import main as fit_main
    return fit_main


def _call_fit(runner: FitRunner, arguments: list[str]) -> int:
    try:
        result = runner(arguments)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1
    return int(result)


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise AutoCalibrationError(
            f"another calibration may be active; lock exists: {path}"
        ) from exc
    try:
        payload = json.dumps(
            {"pid": os.getpid(), "created_utc": _utc_now()},
            ensure_ascii=False,
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _record_stage(manifest: dict[str, Any], status: str, **details: Any) -> None:
    manifest["status"] = status
    manifest.setdefault("stages", []).append(
        {"status": status, "at_utc": _utc_now(), **details}
    )


def run_auto_calibration(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    model_path: Optional[Path] = None,
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    deploy_path: Path = DEFAULT_DEPLOY_PATH,
    results_suffix: str = DEFAULT_RESULTS_SUFFIX,
    device: str = "auto",
    confidence_threshold: float = 0.30,
    bootstrap_runs: int = 200,
    min_loo_recordings: int = 4,
    max_loo_rmse_deg: float = 3.0,
    force: bool = False,
    batch_runner: Optional[BatchRunner] = None,
    fit_runner: Optional[FitRunner] = None,
) -> dict[str, Any]:
    """Run inference -> log validation -> fit -> validation -> atomic apply."""

    runtime_model = resolve_runtime_model(runtime_root, model_path)
    model_id = sha256_file(runtime_model)
    recordings_dir = recordings_dir.resolve()
    runs_dir = runs_dir.resolve()
    deploy_path = deploy_path.resolve()
    if not recordings_dir.is_dir():
        raise AutoCalibrationError(
            f"recordings directory not found: {recordings_dir}"
        )
    if Path(results_suffix).name != results_suffix or not results_suffix.endswith(".csv"):
        raise AutoCalibrationError("results suffix must be a plain .csv suffix")

    runs_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runs_dir / ".auto_calibration.lock"
    with _exclusive_lock(lock_path):
        if deploy_path.is_file() and not force:
            try:
                _, _, artifact_id = validate_deployable_artifact(
                    deploy_path.read_bytes(), expected_model_id=model_id
                )
            except (OSError, AutoCalibrationError, TypeError, ValueError):
                pass
            else:
                return {
                    "schema_version": 1,
                    "status": "ALREADY_APPLIED",
                    "completed_utc": _utc_now(),
                    "model_path": str(runtime_model),
                    "model_hash": model_id,
                    "deploy_path": str(deploy_path),
                    "artifact_sha256": artifact_id,
                }

        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"-{model_id[7:19]}-{secrets.token_hex(3)}"
        )
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=False, exist_ok=False)
        manifest_path = run_dir / RUN_MANIFEST_NAME
        results_dir = run_dir / "inference"
        fit_dir = run_dir / "fit"
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "STARTED",
            "started_utc": _utc_now(),
            "model_path": str(runtime_model),
            "model_hash": model_id,
            "model_size_bytes": runtime_model.stat().st_size,
            "recordings_dir": str(recordings_dir),
            "run_dir": str(run_dir),
            "results_dir": str(results_dir),
            "fit_dir": str(fit_dir),
            "deploy_path": str(deploy_path),
            "settings": {
                "angle_column": "yaw_deg",
                "angle_domain": "heading",
                "command_strength": 30,
                "results_suffix": results_suffix,
                "device": device,
                "confidence_threshold": confidence_threshold,
                "bootstrap_runs": bootstrap_runs,
                "min_loo_recordings": min_loo_recordings,
                "max_loo_rmse_deg": max_loo_rmse_deg,
            },
            "stages": [],
        }
        _record_stage(manifest, "STARTED")
        _atomic_write_json(manifest_path, manifest)

        active_before = (
            deploy_path.read_bytes() if deploy_path.is_file() else None
        )
        try:
            runner = batch_runner or _default_batch_runner()
            batch = runner(
                model_path=runtime_model,
                recordings_dir=recordings_dir,
                results_dir=results_dir,
                results_suffix=results_suffix,
                manifest_name=BATCH_MANIFEST_NAME,
                device=device,
                confidence_threshold=confidence_threshold,
                overwrite=False,
            )
            batch_manifest, prefixes = _validate_batch_manifest(
                batch,
                expected_model_id=model_id,
                results_dir=results_dir,
                results_suffix=results_suffix,
            )
            manifest["batch_inference"] = batch_manifest
            manifest["successful_recording_prefixes"] = prefixes
            _record_stage(
                manifest, "INFERENCE_COMPLETE", recording_count=len(prefixes)
            )
            _atomic_write_json(manifest_path, manifest)

            log_validation = validate_inference_results(
                results_dir,
                prefixes=prefixes,
                results_suffix=results_suffix,
                expected_model_id=model_id,
            )
            manifest["log_validation"] = log_validation
            _record_stage(manifest, "LOGS_VALIDATED")
            _atomic_write_json(manifest_path, manifest)

            fit_arguments = [
                "--recordings-dir", str(recordings_dir),
                "--results-dir", str(results_dir),
                "--results-suffix", results_suffix,
                "--angle-column", "yaw_deg",
                "--angle-domain", "heading",
                "--angle-period", "360.0",
                "--valid-column", "pose_ok",
                "--confidence-column", "confidence",
                "--min-confidence", str(confidence_threshold),
                "--model-id-column", "model_hash",
                "--output-dir", str(fit_dir),
                "--bootstrap-runs", str(bootstrap_runs),
                "--min-loo-recordings", str(min_loo_recordings),
                "--max-loo-rmse-deg", str(max_loo_rmse_deg),
                "--can-time-point", "return",
            ]
            for prefix in prefixes:
                fit_arguments.extend(["--recording-name", prefix])
            manifest["fit_arguments"] = fit_arguments
            _record_stage(manifest, "FITTING")
            _atomic_write_json(manifest_path, manifest)

            code = _call_fit(fit_runner or _default_fit_runner(), fit_arguments)
            manifest["fit_exit_code"] = code
            if code != 0:
                raise AutoCalibrationError(
                    f"rotation fitter rejected the candidate (exit code {code})"
                )
            candidate_path = fit_dir / "rotation_model.json"
            if not candidate_path.is_file():
                raise AutoCalibrationError(
                    "fitter exited successfully without rotation_model.json"
                )
            candidate_payload = candidate_path.read_bytes()
            _, model, candidate_id = validate_deployable_artifact(
                candidate_payload,
                expected_model_id=model_id,
                expected_recordings=prefixes,
            )
            manifest["candidate_artifact"] = {
                "path": str(candidate_path),
                "sha256": candidate_id,
                "parameters": model.to_dict()["parameters"],
                "fit_metadata": model.to_dict()["fit_metadata"],
            }
            _record_stage(manifest, "FIT_VALIDATED")
            _atomic_write_json(manifest_path, manifest)

            # The fitter consumes files from the private inference directory,
            # but it is still an independently callable component.  Re-read
            # every input immediately before promotion so a buggy or replaced
            # fitter cannot fit one byte sequence and deploy as though it had
            # consumed the batch that passed the first validation.
            predeployment_log_validation = validate_inference_results(
                results_dir,
                prefixes=prefixes,
                results_suffix=results_suffix,
                expected_model_id=model_id,
            )
            manifest["predeployment_log_validation"] = (
                predeployment_log_validation
            )
            manifest["log_validation_match"] = (
                predeployment_log_validation == log_validation
            )
            if not manifest["log_validation_match"]:
                raise AutoCalibrationError(
                    "inference results changed during fitting; refusing deployment"
                )
            _record_stage(manifest, "INPUTS_REVALIDATED")
            _atomic_write_json(manifest_path, manifest)

            # Close the model-file time-of-check/time-of-use window.  The batch
            # also snapshots and rechecks it, but promotion repeats the check.
            if sha256_file(runtime_model) != model_id:
                raise AutoCalibrationError(
                    "runtime .pt model changed before artifact promotion"
                )
            deployment = promote_artifact(
                candidate_path,
                deploy_path,
                expected_model_id=model_id,
                expected_recordings=prefixes,
            )
            manifest["deployment"] = deployment
            manifest["completed_utc"] = _utc_now()
            _record_stage(manifest, "APPLIED")
            try:
                _atomic_write_json(manifest_path, manifest)
            except BaseException:
                # Keep the externally visible artifact and audit state aligned.
                if active_before is not None:
                    _atomic_replace_bytes(deploy_path, active_before)
                elif deploy_path.exists():
                    deploy_path.unlink()
                raise
            return manifest
        except BaseException as exc:
            # Promotion itself rolls back.  Assert the documented invariant for
            # every earlier failure too; never silently continue after drift.
            active_after = (
                deploy_path.read_bytes() if deploy_path.is_file() else None
            )
            if active_after != active_before:
                try:
                    if active_before is not None:
                        _atomic_replace_bytes(deploy_path, active_before)
                    elif deploy_path.exists():
                        deploy_path.unlink()
                except BaseException as rollback_exc:
                    manifest["rollback_error"] = (
                        f"{type(rollback_exc).__name__}: {rollback_exc}"
                    )
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            manifest["failed_utc"] = _utc_now()
            _record_stage(manifest, "FAILED")
            try:
                _atomic_write_json(manifest_path, manifest)
            except BaseException:
                pass
            if isinstance(exc, AutoCalibrationError):
                raise
            raise AutoCalibrationError(
                f"automatic rotation calibration failed: {type(exc).__name__}: {exc}"
            ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the runtime pose model, re-infer recordings, validate logs, "
            "fit yaw/heading dynamics, validate, and atomically apply."
        )
    )
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--deploy-path", type=Path, default=DEFAULT_DEPLOY_PATH)
    parser.add_argument("--results-suffix", default=DEFAULT_RESULTS_SUFFIX)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--bootstrap-runs", type=int, default=200)
    parser.add_argument("--min-loo-recordings", type=int, default=4)
    parser.add_argument("--max-loo-rmse-deg", type=float, default=3.0)
    parser.add_argument(
        "--force", action="store_true",
        help="rerun even when a valid artifact already matches the model hash",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_auto_calibration(
            runtime_root=args.runtime_root,
            model_path=args.model,
            recordings_dir=args.recordings_dir,
            runs_dir=args.runs_dir,
            deploy_path=args.deploy_path,
            results_suffix=args.results_suffix,
            device=args.device,
            confidence_threshold=args.confidence,
            bootstrap_runs=args.bootstrap_runs,
            min_loo_recordings=args.min_loo_recordings,
            max_loo_rmse_deg=args.max_loo_rmse_deg,
            force=args.force,
        )
    except (AutoCalibrationError, OSError, ValueError) as exc:
        print(f"automatic rotation calibration failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "model_hash": result["model_hash"],
                "deploy_path": result["deploy_path"],
                "artifact_sha256": (
                    result.get("deployment", {}).get("artifact_sha256")
                    or result.get("artifact_sha256", "")
                ),
                "run_dir": result.get("run_dir", ""),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
