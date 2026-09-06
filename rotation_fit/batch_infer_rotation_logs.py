#!/usr/bin/env python3
"""Run a pose model over recorded raw videos and publish fit-ready CSV files.

The legacy live recorder starts ``*_raw.mp4`` at ``frame_i == 36`` after its FPS
probe has completed.  It is important not to end-align the raw video to the
timing CSV: an old recovered video is known to have lost frames in its middle.
New recordings can provide an explicit ``raw_video_frame_index`` timing column
or ``raw_video_first_frame_i`` metadata and those mappings take precedence.

Publishing is transactional at the batch level: every result is first written
to a private staging directory.  Existing result files are replaced only after
all videos have decoded, inferred, passed PnP, and validated successfully.  The
manifest is committed last and is the completion marker for downstream fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

import cv2
import numpy as np


DEFAULT_RESULTS_SUFFIX = "_new_pose.csv"
DEFAULT_MANIFEST_NAME = "batch_inference_manifest.json"
DEFAULT_FRONT_CLASS_NAME = "item"
DEFAULT_CONFIDENCE = 0.30
EXPECTED_KEYPOINTS = 9
MINIMUM_ULTRALYTICS_VERSION = (8, 4, 60)

RESULT_COLUMNS = (
    "frame_i",
    "yaw_deg",
    "pose_ok",
    "valid",
    "confidence",
    "model_hash",
    "angle_domain",
    "angle_source",
    "model_det_ok",
    "pos_x_m",
    "pos_z_m",
    "pose_src",
    "pose_n_used",
    "pose_rms_px",
    "keypoint_count",
    "video_frame_i",
    "failure_reason",
)


class BatchInferenceError(RuntimeError):
    """The batch cannot safely be used as rotation-fit input."""


class RecordingEligibilityError(BatchInferenceError):
    """One recording is unusable, but other recordings may still be processed."""

    def __init__(self, message: str, *, audit: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.audit = audit or {}


@dataclass(frozen=True)
class RecordingInput:
    prefix: str
    video_path: Path
    timing_path: Path
    meta_path: Path
    control_path: Path


@dataclass(frozen=True)
class RecordingExclusion:
    prefix: str
    eligible: bool
    exclusion_stage: str
    exclusion_reason: str
    source_video: Optional[str] = None
    frame_mapping_method: Optional[str] = None
    video_frame_count: Optional[int] = None
    timing_row_count: Optional[int] = None
    joined_timing_frames: Optional[int] = None
    joined_ratio: Optional[float] = None
    timing_span_frames: Optional[int] = None
    timing_span_coverage_ratio: Optional[float] = None
    raw_first_frame_i: Optional[int] = None
    raw_last_frame_i: Optional[int] = None
    timing_last_frame_i: Optional[int] = None


@dataclass(frozen=True)
class FrameMapping:
    frame_ids: tuple[int, ...]
    joinable_frame_ids: frozenset[int]
    method: str
    timing_row_count: int
    timing_first_frame_i: int
    timing_last_frame_i: int
    joined_timing_frames: int
    joined_ratio: float
    timing_span_frames: int
    timing_span_coverage_ratio: float
    raw_first_frame_i: int
    raw_last_frame_i: int
    raw_minus_timing_tail_frames: int


@dataclass(frozen=True)
class PalletGeometry:
    width_m: float
    height_m: float
    length_m: float
    visibility_threshold: float


@dataclass(frozen=True)
class Detection:
    keypoints: Optional[np.ndarray]
    confidence: Optional[float]
    bbox_xyxy: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PoseEstimate:
    yaw_deg: float
    pos_x_m: float
    pos_z_m: float
    source: str = ""
    n_used: int = 0
    rms_px: Optional[float] = None


@dataclass(frozen=True)
class RecordingReport:
    prefix: str
    source_video: str
    result_file: str
    frame_count: int
    first_frame_i: int
    last_frame_i: int
    frame_mapping_method: str
    timing_row_count: int
    joined_timing_frames: int
    joined_ratio: float
    timing_span_frames: int
    timing_span_coverage_ratio: float
    raw_minus_timing_tail_frames: int
    decoded_frames: int
    detections: int
    valid_poses: int
    joined_valid_poses: int
    valid_fraction: float
    result_sha256: str
    elapsed_sec: float


class Predictor(Protocol):
    device: str

    def predict(self, image: np.ndarray) -> Detection:
        ...


PoseSolver = Callable[[np.ndarray, Any, PalletGeometry], Optional[PoseEstimate]]
CaptureFactory = Callable[[str], Any]
ProgressCallback = Callable[[str], None]


def _say(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_model(model_path: Path, destination: Path) -> tuple[str, int]:
    """Copy weights once while hashing, so inference and provenance cannot race."""
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("rb") as source, destination.open("xb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            target.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        target.flush()
        os.fsync(target.fileno())
    if size <= 0:
        raise BatchInferenceError(f"model file is empty: {model_path}")
    return f"sha256:{digest.hexdigest()}", size


def _safe_suffix(value: str) -> str:
    if not value or Path(value).name != value or not value.endswith(".csv"):
        raise ValueError("results suffix must be a plain .csv filename suffix")
    return value


def _safe_manifest_name(value: str) -> str:
    if not value or Path(value).name != value or not value.endswith(".json"):
        raise ValueError("manifest name must be a plain .json filename")
    return value


def discover_recordings_with_audit(
    recordings_dir: Path,
    *,
    only: Sequence[str] = (),
) -> tuple[list[RecordingInput], list[RecordingExclusion]]:
    """Find control-log recordings and explain every missing inference input."""
    recordings_dir = recordings_dir.resolve()
    if not recordings_dir.is_dir():
        raise FileNotFoundError(f"recordings directory not found: {recordings_dir}")

    filters = tuple(item for item in only if item)
    found: list[RecordingInput] = []
    excluded: list[RecordingExclusion] = []
    prefixes = {
        path.name[: -len("_control_seq.jsonl")]
        for path in recordings_dir.glob("*_control_seq.jsonl")
    }
    # Surface orphan raw videos too; an absent control log makes them unusable
    # for a command-response fit and should not disappear silently.
    prefixes.update(
        path.name[: -len("_raw.mp4")]
        for path in recordings_dir.glob("*_raw.mp4")
    )
    prefixes.update(
        path.name[: -len("_raw.avi")]
        for path in recordings_dir.glob("*_raw.avi")
    )
    for prefix in sorted(prefixes):
        if filters and not any(token in prefix for token in filters):
            continue
        video = recordings_dir / f"{prefix}_raw.mp4"
        lossless_video = recordings_dir / f"{prefix}_raw.avi"
        if video.is_file() and lossless_video.is_file():
            raise BatchInferenceError(f"ambiguous raw videos for {prefix}")
        if lossless_video.is_file():
            video = lossless_video
        timing = recordings_dir / f"{prefix}_inference_timing.csv"
        meta = recordings_dir / f"{prefix}_meta.json"
        control = recordings_dir / f"{prefix}_control_seq.jsonl"
        missing = [p.name for p in (video, timing, meta, control) if not p.is_file()]
        if missing:
            excluded.append(
                RecordingExclusion(
                    prefix=prefix,
                    eligible=False,
                    exclusion_stage="input_discovery",
                    exclusion_reason="missing " + ", ".join(missing),
                    source_video=str(video) if video.is_file() else None,
                )
            )
            continue
        found.append(RecordingInput(prefix, video, timing, meta, control))

    if not found and not excluded:
        detail = f" matching {filters!r}" if filters else ""
        raise BatchInferenceError(
            f"no recording inputs{detail} in {recordings_dir}"
        )
    return found, excluded


def discover_recordings(
    recordings_dir: Path,
    *,
    only: Sequence[str] = (),
) -> list[RecordingInput]:
    """Compatibility helper returning only complete recording inputs."""
    found, _ = discover_recordings_with_audit(recordings_dir, only=only)
    return found


def _read_timing_frame_rows(
    timing_path: Path,
) -> tuple[list[int], Optional[dict[int, int]]]:
    try:
        with timing_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "frame_i" not in reader.fieldnames:
                raise RecordingEligibilityError(
                    f"frame_i column missing from {timing_path.name}"
                )
            values: list[int] = []
            index_column = next(
                (
                    name
                    for name in ("raw_video_frame_index", "raw_video_frame_i")
                    if name in reader.fieldnames
                ),
                None,
            )
            explicit: Optional[dict[int, int]] = {} if index_column else None
            for line_number, row in enumerate(reader, start=2):
                raw = str(row.get("frame_i", "")).strip()
                try:
                    numeric = float(raw)
                except ValueError as exc:
                    raise RecordingEligibilityError(
                        f"invalid frame_i at {timing_path.name}:{line_number}: {raw!r}"
                    ) from exc
                if not math.isfinite(numeric) or not numeric.is_integer():
                    raise RecordingEligibilityError(
                        f"non-integral frame_i at {timing_path.name}:{line_number}"
                    )
                frame_i = int(numeric)
                values.append(frame_i)
                if explicit is not None:
                    raw_index = str(row.get(index_column or "", "")).strip()
                    if raw_index:
                        try:
                            index_numeric = float(raw_index)
                        except ValueError as exc:
                            raise RecordingEligibilityError(
                                f"invalid {index_column} at "
                                f"{timing_path.name}:{line_number}"
                            ) from exc
                        if (
                            not math.isfinite(index_numeric)
                            or not index_numeric.is_integer()
                            or index_numeric < 0
                        ):
                            raise RecordingEligibilityError(
                                f"invalid {index_column} at "
                                f"{timing_path.name}:{line_number}"
                            )
                        index = int(index_numeric)
                        if index in explicit:
                            raise RecordingEligibilityError(
                                f"duplicate {index_column}={index} in {timing_path.name}"
                            )
                        explicit[index] = frame_i
    except OSError as exc:
        raise RecordingEligibilityError(
            f"cannot read timing log {timing_path}: {exc}"
        ) from exc

    if not values:
        raise RecordingEligibilityError(f"no timing rows in {timing_path.name}")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise RecordingEligibilityError(
            f"frame_i is not strictly increasing in {timing_path.name}"
        )
    return values, explicit


def _metadata_raw_first_frame(meta_path: Path) -> Optional[int]:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    candidate = meta.get("raw_video_first_frame_i")
    raw_video = meta.get("raw_video")
    if candidate is None and isinstance(raw_video, dict):
        candidate = raw_video.get("first_frame_i")
    if candidate is None:
        return None
    try:
        numeric = float(candidate)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        return None
    return int(numeric)


def build_frame_mapping(
    timing_path: Path,
    meta_path: Path,
    video_frame_count: int,
) -> FrameMapping:
    """Build an auditable raw-index to ``frame_i`` mapping.

    Explicit per-frame indices are authoritative.  Current legacy recordings
    have no such column, so their source-code invariant ``raw[0] == frame 36``
    is used.  A missing final timing row is tolerated and reported; a video that
    ends materially before the timing log is rejected because that is the
    signature of a pause or an internally damaged/recovered video.
    """
    if video_frame_count <= 0:
        raise RecordingEligibilityError("video reports no frames")
    values, explicit = _read_timing_frame_rows(timing_path)
    timing_set = set(values)

    if explicit is not None:
        expected_indices = set(range(video_frame_count))
        actual_indices = set(explicit)
        if actual_indices != expected_indices:
            missing = len(expected_indices - actual_indices)
            extra = len(actual_indices - expected_indices)
            raise RecordingEligibilityError(
                "explicit raw-video mapping is incomplete: "
                f"missing={missing}, extra={extra}",
                audit={
                    "frame_mapping_method": "timing_raw_video_frame_index",
                    "video_frame_count": video_frame_count,
                    "timing_row_count": len(values),
                    "joined_timing_frames": len(actual_indices & expected_indices),
                },
            )
        mapped = tuple(explicit[index] for index in range(video_frame_count))
        if any(right <= left for left, right in zip(mapped, mapped[1:])):
            raise RecordingEligibilityError(
                "explicit raw-video frame mapping is not strictly increasing"
            )
        method = "timing_raw_video_frame_index"
    else:
        configured_first = _metadata_raw_first_frame(meta_path)
        first = configured_first if configured_first is not None else 36
        method = (
            "metadata_raw_video_first_frame_i"
            if configured_first is not None
            else "legacy_recorder_first_frame_i_36"
        )
        mapped = tuple(range(first, first + video_frame_count))

    joined = sum(frame_i in timing_set for frame_i in mapped)
    joined_ratio = joined / video_frame_count
    raw_first = mapped[0]
    raw_last = mapped[-1]
    tail_delta = raw_last - values[-1]
    timing_span_frames = sum(frame_i >= raw_first for frame_i in values)
    timing_span_coverage_ratio = (
        joined / timing_span_frames if timing_span_frames else 0.0
    )
    audit = {
        "frame_mapping_method": method,
        "video_frame_count": video_frame_count,
        "timing_row_count": len(values),
        "joined_timing_frames": joined,
        "joined_ratio": joined_ratio,
        "timing_span_frames": timing_span_frames,
        "timing_span_coverage_ratio": timing_span_coverage_ratio,
        "raw_first_frame_i": raw_first,
        "raw_last_frame_i": raw_last,
        "timing_last_frame_i": values[-1],
    }

    if explicit is None:
        missing_positions = [
            index for index, frame_i in enumerate(mapped) if frame_i not in timing_set
        ]
        allowed_missing = [] if not missing_positions else [video_frame_count - 1]
        if missing_positions != allowed_missing or tail_delta not in (0, 1):
            raise RecordingEligibilityError(
                "legacy raw/timing alignment is ambiguous: expected raw[0]="
                f"frame_i 36, raw_last={raw_last}, timing_last={values[-1]}, "
                f"joined={joined}/{video_frame_count}",
                audit=audit,
            )
    elif joined != video_frame_count:
        raise RecordingEligibilityError(
            "explicit raw-video mapping contains frame ids absent from timing log",
            audit=audit,
        )

    return FrameMapping(
        frame_ids=mapped,
        joinable_frame_ids=frozenset(frame_i for frame_i in mapped if frame_i in timing_set),
        method=method,
        timing_row_count=len(values),
        timing_first_frame_i=values[0],
        timing_last_frame_i=values[-1],
        joined_timing_frames=joined,
        joined_ratio=joined_ratio,
        timing_span_frames=timing_span_frames,
        timing_span_coverage_ratio=timing_span_coverage_ratio,
        raw_first_frame_i=raw_first,
        raw_last_frame_i=raw_last,
        raw_minus_timing_tail_frames=tail_delta,
    )


def load_timing_frame_ids(
    timing_path: Path,
    video_frame_count: int,
    *,
    meta_path: Optional[Path] = None,
) -> list[int]:
    """Compatibility wrapper around :func:`build_frame_mapping`."""
    if meta_path is None:
        meta_path = timing_path.with_name(
            timing_path.name.replace("_inference_timing.csv", "_meta.json")
        )
    return list(build_frame_mapping(timing_path, meta_path, video_frame_count).frame_ids)


def load_camera_metadata(meta_path: Path) -> tuple[Any, PalletGeometry]:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchInferenceError(f"cannot read metadata {meta_path}: {exc}") from exc

    raw_intrinsics = meta.get("intrinsics")
    if not isinstance(raw_intrinsics, dict):
        raise BatchInferenceError(f"intrinsics missing from {meta_path.name}")
    required = ("fx", "fy", "ppx", "ppy", "width", "height")
    try:
        values = {name: float(raw_intrinsics[name]) for name in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise BatchInferenceError(
            f"invalid intrinsics in {meta_path.name}: {exc}"
        ) from exc
    if (
        not all(math.isfinite(value) for value in values.values())
        or values["fx"] <= 0.0
        or values["fy"] <= 0.0
        or values["width"] <= 0.0
        or values["height"] <= 0.0
    ):
        raise BatchInferenceError(f"non-physical intrinsics in {meta_path.name}")

    try:
        coeffs = [float(value) for value in raw_intrinsics.get("coeffs", [0] * 5)]
    except (TypeError, ValueError) as exc:
        raise BatchInferenceError(
            f"invalid distortion coefficients in {meta_path.name}"
        ) from exc
    if not coeffs or not all(math.isfinite(value) for value in coeffs):
        raise BatchInferenceError(
            f"invalid distortion coefficients in {meta_path.name}"
        )

    intrinsics = SimpleNamespace(
        fx=values["fx"],
        fy=values["fy"],
        ppx=values["ppx"],
        ppy=values["ppy"],
        width=int(round(values["width"])),
        height=int(round(values["height"])),
        coeffs=coeffs,
        model=str(raw_intrinsics.get("model", "")),
    )

    raw_size = meta.get("pallet_size_m", {})
    try:
        geometry = PalletGeometry(
            width_m=float(raw_size.get("width", 1.1)),
            height_m=float(raw_size.get("height", 0.15)),
            length_m=float(raw_size.get("length", 1.1)),
            visibility_threshold=float(meta.get("pose_kpt_visibility_threshold", 0.5)),
        )
    except (TypeError, ValueError) as exc:
        raise BatchInferenceError(
            f"invalid pallet geometry in {meta_path.name}"
        ) from exc
    geometry_values = (
        geometry.width_m,
        geometry.height_m,
        geometry.length_m,
        geometry.visibility_threshold,
    )
    if not all(math.isfinite(value) for value in geometry_values):
        raise BatchInferenceError(f"non-finite pallet geometry in {meta_path.name}")
    if min(geometry.width_m, geometry.height_m, geometry.length_m) <= 0.0:
        raise BatchInferenceError(f"non-physical pallet geometry in {meta_path.name}")
    if not 0.0 <= geometry.visibility_threshold <= 1.0:
        raise BatchInferenceError(f"invalid visibility threshold in {meta_path.name}")
    return intrinsics, geometry


def select_detection_index(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    classes: np.ndarray,
    *,
    target_class: int,
    confidence_threshold: float,
    target_aspect_ratio: float,
) -> Optional[int]:
    """Match ``calib.perception.Perception`` detection-selection semantics."""
    candidates = [
        index
        for index, (class_id, confidence) in enumerate(zip(classes, confidences))
        if int(class_id) == int(target_class)
        and float(confidence) >= float(confidence_threshold)
    ]
    if not candidates:
        return None

    def score(index: int) -> tuple[float, float]:
        x1, y1, x2, y2 = map(float, boxes_xyxy[index])
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        return abs(width / height - target_aspect_ratio), -float(confidences[index])

    return min(candidates, key=score)


def resolve_device(requested: str) -> str:
    value = requested.strip().lower()
    if value == "cpu":
        return "cpu"
    if value.isdigit():
        value = f"cuda:{value}"
    if value == "cuda":
        value = "cuda:0"
    if value != "auto" and not value.startswith("cuda:"):
        raise ValueError("device must be auto, cpu, cuda, cuda:N, or N")

    try:
        import torch
    except ImportError as exc:
        if value == "auto":
            return "cpu"
        raise BatchInferenceError("PyTorch is required for CUDA inference") from exc

    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise BatchInferenceError(f"requested {value}, but CUDA is not available")
    try:
        index = int(value.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("CUDA device must have the form cuda:N") from exc
    if index < 0 or index >= int(torch.cuda.device_count()):
        raise BatchInferenceError(
            f"CUDA device index {index} is unavailable "
            f"(device_count={torch.cuda.device_count()})"
        )
    return f"cuda:{index}"


def _class_names_as_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


class YoloPosePredictor:
    """Small batch wrapper with the same target-selection rule as live code."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str,
        confidence_threshold: float,
        front_class_name: str,
        use_half: bool,
        target_aspect_ratio: float = 1.1 / 0.15,
    ) -> None:
        try:
            installed_version = importlib.metadata.version("ultralytics")
        except importlib.metadata.PackageNotFoundError as exc:
            raise BatchInferenceError(
                "cannot determine the installed ultralytics version"
            ) from exc
        if _version_tuple(installed_version) < MINIMUM_ULTRALYTICS_VERSION:
            required = ".".join(map(str, MINIMUM_ULTRALYTICS_VERSION))
            raise BatchInferenceError(
                "ultralytics>=" + required + " is required so batch and live "
                "inference use the same quantize precision option; installed="
                + installed_version
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BatchInferenceError(
                "ultralytics is required for .pt batch inference"
            ) from exc

        self.device = resolve_device(device)
        self.confidence_threshold = float(confidence_threshold)
        self.front_class_name = front_class_name
        self.use_half = bool(use_half and self.device.startswith("cuda:"))
        self.target_aspect_ratio = float(target_aspect_ratio)
        self.model = YOLO(str(model_path))

        task = getattr(self.model, "task", None)
        if task not in (None, "pose"):
            raise BatchInferenceError(
                f"model task is {task!r}; a pose model is required"
            )
        keypoint_shape = getattr(getattr(self.model, "model", None), "kpt_shape", None)
        if keypoint_shape and int(keypoint_shape[0]) < EXPECTED_KEYPOINTS:
            raise BatchInferenceError(
                f"model has {keypoint_shape[0]} keypoints; "
                f"at least {EXPECTED_KEYPOINTS} are required"
            )
        names = _class_names_as_dict(getattr(self.model, "names", None))
        matching = [index for index, name in names.items() if name == front_class_name]
        if matching:
            self.target_class = matching[0]
        elif len(names) == 1:
            self.target_class = next(iter(names))
        else:
            raise BatchInferenceError(
                f"cannot resolve class {front_class_name!r} from model names {names}"
            )

    def predict(self, image: np.ndarray) -> Detection:
        kwargs: dict[str, Any] = {
            "source": image,
            "device": self.device,
            "verbose": False,
            "conf": self.confidence_threshold,
            # Keep batch calibration numerically aligned with the live
            # Perception path.  The required Ultralytics version exposes the
            # unified precision option used by calib/perception.py.
            "quantize": 16 if self.use_half else None,
        }
        result = self.model.predict(**kwargs)[0]
        if (
            result.boxes is None
            or len(result.boxes) == 0
            or result.keypoints is None
            or result.keypoints.data is None
            or len(result.keypoints.data) == 0
        ):
            return Detection(None, None, None)

        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
        keypoints = result.keypoints.data.detach().cpu().numpy().astype(np.float32)
        index = select_detection_index(
            boxes,
            confidences,
            classes,
            target_class=self.target_class,
            confidence_threshold=self.confidence_threshold,
            target_aspect_ratio=self.target_aspect_ratio,
        )
        if index is None:
            return Detection(None, None, None)
        selected = np.asarray(keypoints[index], dtype=np.float32)
        if selected.ndim != 2 or selected.shape[1] < 2:
            raise BatchInferenceError(
                f"model returned malformed keypoints with shape {selected.shape}"
            )
        if selected.shape[0] < EXPECTED_KEYPOINTS:
            raise BatchInferenceError(
                f"model returned {selected.shape[0]} keypoints; "
                f"at least {EXPECTED_KEYPOINTS} are required"
            )
        return Detection(
            selected,
            float(confidences[index]),
            np.asarray(boxes[index], dtype=np.float32),
        )


def load_live_pose_solver(depth_cam_dir: Optional[Path] = None) -> PoseSolver:
    """Load the exact PnP implementation used by ``main_rec.py`` lazily."""
    if depth_cam_dir is None:
        depth_cam_dir = Path(__file__).resolve().parents[1] / "extracted" / "depth_cam"
    depth_cam_dir = depth_cam_dir.resolve()
    if not (depth_cam_dir / "calib" / "geometry.py").is_file():
        raise BatchInferenceError(f"depth_cam source not found: {depth_cam_dir}")

    path_text = str(depth_cam_dir)
    inserted = path_text not in sys.path
    if inserted:
        sys.path.insert(0, path_text)
    try:
        from calib.geometry import pose_from_visible_kpts_pnp
    except Exception as exc:
        raise BatchInferenceError(
            f"cannot load live PnP implementation from {depth_cam_dir}: {exc}"
        ) from exc
    finally:
        if inserted:
            try:
                sys.path.remove(path_text)
            except ValueError:
                pass

    def solve(
        keypoints: np.ndarray,
        intrinsics: Any,
        pallet: PalletGeometry,
    ) -> Optional[PoseEstimate]:
        result = pose_from_visible_kpts_pnp(
            kpts=keypoints,
            intrin=intrinsics,
            face_w=pallet.width_m,
            face_h=pallet.height_m,
            body_d=pallet.length_m,
            vis_thr=pallet.visibility_threshold,
        )
        ok, yaw, _pitch, _roll, center, _rvec, _tvec, info = result
        if not ok or yaw is None or center is None:
            return None
        center_array = np.asarray(center, dtype=float).reshape(3)
        yaw_value = float(yaw)
        if (
            not math.isfinite(yaw_value)
            or not np.all(np.isfinite(center_array))
            or float(center_array[2]) <= 0.0
        ):
            return None
        rms = info.get("rms")
        rms_value = None if rms is None else float(rms)
        if rms_value is not None and not math.isfinite(rms_value):
            rms_value = None
        return PoseEstimate(
            yaw_deg=yaw_value,
            pos_x_m=float(center_array[0]),
            pos_z_m=float(center_array[2]),
            source=str(info.get("src") or ""),
            n_used=int(info.get("n_used") or 0),
            rms_px=rms_value,
        )

    return solve


def _format_float(value: Optional[float], digits: int) -> str:
    if value is None:
        return ""
    numeric = float(value)
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else ""


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def infer_recording(
    recording: RecordingInput,
    output_path: Path,
    *,
    predictor: Predictor,
    pose_solver: PoseSolver,
    model_id: str,
    capture_factory: CaptureFactory = cv2.VideoCapture,
    progress: ProgressCallback = _say,
    progress_every: int = 250,
    allow_low_pose: bool = False,
    allow_unmapped_video: bool = False,
) -> RecordingReport:
    """Infer one recording into an output inside a private staging directory."""
    started = time.perf_counter()
    try:
        intrinsics, geometry = load_camera_metadata(recording.meta_path)
    except BatchInferenceError as exc:
        raise RecordingEligibilityError(str(exc)) from exc
    capture = capture_factory(str(recording.video_path))
    if not capture.isOpened():
        raise RecordingEligibilityError(f"cannot open video: {recording.video_path}")

    try:
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if frame_count <= 0:
            raise RecordingEligibilityError(
                f"video frame count is unavailable: {recording.video_path.name}"
            )
        if (width, height) != (intrinsics.width, intrinsics.height):
            raise RecordingEligibilityError(
                f"video size {width}x{height} does not match intrinsics "
                f"{intrinsics.width}x{intrinsics.height} for {recording.prefix}"
            )
        try:
            mapping = build_frame_mapping(
                recording.timing_path, recording.meta_path, frame_count
            )
        except RecordingEligibilityError:
            if not allow_unmapped_video:
                raise
            timing_values, _explicit = _read_timing_frame_rows(recording.timing_path)
            # Evaluation-only fallback.  These rows deliberately do not join to
            # control timing and therefore cannot enter a response calibration.
            mapping = FrameMapping(
                frame_ids=tuple(range(frame_count)),
                joinable_frame_ids=frozenset(),
                method="evaluation_video_index_only",
                timing_row_count=len(timing_values),
                timing_first_frame_i=timing_values[0],
                timing_last_frame_i=timing_values[-1],
                joined_timing_frames=0,
                joined_ratio=0.0,
                timing_span_frames=0,
                timing_span_coverage_ratio=0.0,
                raw_first_frame_i=0,
                raw_last_frame_i=frame_count - 1,
                raw_minus_timing_tail_frames=0,
            )
        frame_ids = list(mapping.frame_ids)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        detections = 0
        valid_poses = 0
        joined_valid_poses = 0
        with output_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for video_frame_i, frame_i in enumerate(frame_ids):
                decoded, image = capture.read()
                if not decoded or image is None:
                    raise RecordingEligibilityError(
                        f"video decode stopped at frame {video_frame_i}/{frame_count} "
                        f"for {recording.prefix}"
                    )
                detection = predictor.predict(image)
                pose: Optional[PoseEstimate] = None
                failure_reason = "no_detection"
                keypoint_count = ""
                if detection.keypoints is not None:
                    detections += 1
                    keypoint_count = int(len(detection.keypoints))
                    failure_reason = "pnp_failed"
                    pose = pose_solver(detection.keypoints, intrinsics, geometry)
                if pose is not None:
                    if not (
                        math.isfinite(float(pose.yaw_deg))
                        and math.isfinite(float(pose.pos_x_m))
                        and math.isfinite(float(pose.pos_z_m))
                        and float(pose.pos_z_m) > 0.0
                    ):
                        pose = None
                    else:
                        valid_poses += 1
                        if frame_i in mapping.joinable_frame_ids:
                            joined_valid_poses += 1
                        failure_reason = ""

                writer.writerow(
                    {
                        "frame_i": frame_i,
                        "yaw_deg": _format_float(
                            None if pose is None else pose.yaw_deg, 6
                        ),
                        "pose_ok": int(pose is not None),
                        "valid": int(pose is not None),
                        "confidence": _format_float(detection.confidence, 6),
                        "model_hash": model_id,
                        "angle_domain": "heading",
                        "angle_source": "pnp_face_orientation",
                        "model_det_ok": int(detection.keypoints is not None),
                        "pos_x_m": _format_float(
                            None if pose is None else pose.pos_x_m, 8
                        ),
                        "pos_z_m": _format_float(
                            None if pose is None else pose.pos_z_m, 8
                        ),
                        "pose_src": "" if pose is None else pose.source,
                        "pose_n_used": "" if pose is None else pose.n_used,
                        "pose_rms_px": _format_float(
                            None if pose is None else pose.rms_px, 6
                        ),
                        "keypoint_count": keypoint_count,
                        "video_frame_i": video_frame_i,
                        "failure_reason": failure_reason,
                    }
                )
                if progress_every > 0 and (video_frame_i + 1) % progress_every == 0:
                    progress(
                        f"[{recording.prefix}] {video_frame_i + 1}/{frame_count}"
                    )
            extra, _ = capture.read()
            if extra:
                raise RecordingEligibilityError(
                    f"video contains more frames than CAP_PROP_FRAME_COUNT for "
                    f"{recording.prefix}; frame alignment is ambiguous"
                )
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        capture.release()

    validate_result_csv(
        output_path,
        expected_frame_ids=frame_ids,
        expected_model_id=model_id,
        minimum_valid_poses=0,
    )
    if joined_valid_poses < 3 and not allow_low_pose:
        # This is a property of the newly loaded model, not of the recorded
        # source.  Silently dropping a hard recording would bias the fit toward
        # the few videos on which the model succeeds, so fail the whole batch.
        raise BatchInferenceError(
            f"{recording.prefix} has only {joined_valid_poses} valid poses that "
            "join to timing rows; at least 3 are required; refusing to "
            "selection-bias calibration by excluding this model failure",
        )
    elapsed = time.perf_counter() - started
    return RecordingReport(
        prefix=recording.prefix,
        source_video=str(recording.video_path.resolve()),
        result_file=output_path.name,
        frame_count=frame_count,
        first_frame_i=frame_ids[0],
        last_frame_i=frame_ids[-1],
        frame_mapping_method=mapping.method,
        timing_row_count=mapping.timing_row_count,
        joined_timing_frames=mapping.joined_timing_frames,
        joined_ratio=mapping.joined_ratio,
        timing_span_frames=mapping.timing_span_frames,
        timing_span_coverage_ratio=mapping.timing_span_coverage_ratio,
        raw_minus_timing_tail_frames=mapping.raw_minus_timing_tail_frames,
        decoded_frames=frame_count,
        detections=detections,
        valid_poses=valid_poses,
        joined_valid_poses=joined_valid_poses,
        valid_fraction=valid_poses / frame_count,
        result_sha256=sha256_file(output_path),
        elapsed_sec=elapsed,
    )


def validate_result_csv(
    path: Path,
    *,
    expected_frame_ids: Sequence[int],
    expected_model_id: str,
    minimum_valid_poses: int = 3,
) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RESULT_COLUMNS:
                raise BatchInferenceError(f"unexpected result schema in {path.name}")
            rows = list(reader)
    except OSError as exc:
        raise BatchInferenceError(f"cannot validate {path}: {exc}") from exc

    if len(rows) != len(expected_frame_ids):
        raise BatchInferenceError(
            f"result row count mismatch in {path.name}: "
            f"{len(rows)} != {len(expected_frame_ids)}"
        )
    valid_count = 0
    for index, (row, expected_frame_i) in enumerate(zip(rows, expected_frame_ids)):
        try:
            frame_i = int(row["frame_i"])
        except (TypeError, ValueError) as exc:
            raise BatchInferenceError(
                f"invalid result frame_i at row {index + 2} in {path.name}"
            ) from exc
        if frame_i != expected_frame_i:
            raise BatchInferenceError(
                f"result frame mismatch at row {index + 2} in {path.name}: "
                f"{frame_i} != {expected_frame_i}"
            )
        if row.get("model_hash") != expected_model_id:
            raise BatchInferenceError(
                f"model hash mismatch at row {index + 2} in {path.name}"
            )
        if row.get("angle_domain") != "heading":
            raise BatchInferenceError(f"non-heading result in {path.name}")
        pose_ok = row.get("pose_ok") == "1"
        if row.get("valid") != ("1" if pose_ok else "0"):
            raise BatchInferenceError(f"valid/pose_ok mismatch in {path.name}")
        if pose_ok:
            try:
                yaw = float(row["yaw_deg"])
            except (TypeError, ValueError) as exc:
                raise BatchInferenceError(
                    f"valid pose has no yaw_deg in {path.name}"
                ) from exc
            if not math.isfinite(yaw):
                raise BatchInferenceError(f"valid pose has non-finite yaw in {path.name}")
            valid_count += 1
        elif row.get("yaw_deg", "").strip():
            raise BatchInferenceError(f"invalid pose carries yaw_deg in {path.name}")
    if valid_count < minimum_valid_poses:
        raise BatchInferenceError(
            f"{path.name} has only {valid_count} valid poses; "
            f"at least {minimum_valid_poses} are required"
        )


def _publish_transaction(
    staged_and_targets: Sequence[tuple[Path, Path]],
    *,
    overwrite: bool,
) -> None:
    if not overwrite:
        existing = [str(target) for _, target in staged_and_targets if target.exists()]
        if existing:
            raise FileExistsError("result already exists: " + ", ".join(existing))

    backups: list[tuple[Path, Optional[Path]]] = []
    committed: list[tuple[Path, Path]] = []
    backup_dir = staged_and_targets[0][0].parent / ".previous"
    backup_dir.mkdir(exist_ok=True)
    try:
        for index, (staged, target) in enumerate(staged_and_targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            backup: Optional[Path] = None
            if target.exists():
                backup = backup_dir / f"{index:05d}-{target.name}"
                os.replace(target, backup)
            backups.append((target, backup))
            try:
                os.replace(staged, target)
            except BaseException:
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                raise
            committed.append((target, staged))
    except BaseException:
        # Restore all previously visible files.  Moving instead of unlinking
        # keeps both old and newly generated data recoverable during rollback.
        for target, staged in reversed(committed):
            if target.exists():
                os.replace(target, staged)
        for target, backup in reversed(backups):
            if backup is not None and backup.exists():
                os.replace(backup, target)
        raise


def run_batch_inference(
    *,
    model_path: Path,
    recordings_dir: Path,
    results_dir: Path,
    results_suffix: str = DEFAULT_RESULTS_SUFFIX,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    device: str = "auto",
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    front_class_name: str = DEFAULT_FRONT_CLASS_NAME,
    use_half: bool = True,
    only: Sequence[str] = (),
    overwrite: bool = True,
    recordings: Optional[Sequence[RecordingInput]] = None,
    predictor: Optional[Predictor] = None,
    pose_solver: Optional[PoseSolver] = None,
    capture_factory: CaptureFactory = cv2.VideoCapture,
    progress: ProgressCallback = _say,
    allow_low_pose: bool = False,
    allow_unmapped_video: bool = False,
) -> dict[str, Any]:
    """Generate and atomically publish a complete set of fit-ready CSVs."""
    model_path = model_path.resolve()
    recordings_dir = recordings_dir.resolve()
    results_dir = results_dir.resolve()
    suffix = _safe_suffix(results_suffix)
    manifest_filename = _safe_manifest_name(manifest_name)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise BatchInferenceError(f"expected a .pt model: {model_path}")
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence threshold must be between 0 and 1")

    if recordings is not None:
        selected = list(recordings)
        exclusions: list[RecordingExclusion] = []
    else:
        selected, exclusions = discover_recordings_with_audit(
            recordings_dir, only=only
        )
    prefixes = [recording.prefix for recording in selected]
    if len(prefixes) != len(set(prefixes)):
        raise BatchInferenceError("duplicate recording prefixes in batch")

    target_paths = [results_dir / f"{prefix}{suffix}" for prefix in prefixes]
    target_paths.append(results_dir / manifest_filename)
    if not overwrite:
        existing = [str(path) for path in target_paths if path.exists()]
        if existing:
            raise FileExistsError("result already exists: " + ", ".join(existing))

    results_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".batch-infer-", dir=results_dir))
    try:
        snapshot_path = staging_dir / ".model-snapshot.pt"
        model_id, model_size = snapshot_model(model_path, snapshot_path)
        if predictor is None and selected:
            predictor = YoloPosePredictor(
                snapshot_path,
                device=device,
                confidence_threshold=confidence_threshold,
                front_class_name=front_class_name,
                use_half=use_half,
            )
        if pose_solver is None and selected:
            pose_solver = load_live_pose_solver()
        predictor_device = getattr(predictor, "device", None)
        resolved_device = str(predictor_device or resolve_device(device))

        progress(
            f"model={model_path.name} id={model_id} device={resolved_device} "
            f"recordings={len(selected)}"
        )
        reports: list[RecordingReport] = []
        staged_pairs: list[tuple[Path, Path]] = []
        batch_started = time.perf_counter()
        for number, recording in enumerate(selected, start=1):
            progress(f"[{number}/{len(selected)}] {recording.prefix}")
            staged_path = staging_dir / f"{recording.prefix}{suffix}"
            try:
                if predictor is None or pose_solver is None:
                    raise BatchInferenceError("inference components were not initialised")
                report = infer_recording(
                    recording,
                    staged_path,
                    predictor=predictor,
                    pose_solver=pose_solver,
                    model_id=model_id,
                    capture_factory=capture_factory,
                    progress=progress,
                    allow_low_pose=allow_low_pose,
                    allow_unmapped_video=allow_unmapped_video,
                )
            except RecordingEligibilityError as exc:
                try:
                    staged_path.unlink()
                except FileNotFoundError:
                    pass
                details = dict(exc.audit)
                exclusions.append(
                    RecordingExclusion(
                        prefix=recording.prefix,
                        eligible=False,
                        exclusion_stage=(
                            "frame_mapping"
                            if details.get("frame_mapping_method")
                            else "recording_validation"
                        ),
                        exclusion_reason=str(exc),
                        source_video=str(recording.video_path.resolve()),
                        frame_mapping_method=details.get("frame_mapping_method"),
                        video_frame_count=details.get("video_frame_count"),
                        timing_row_count=details.get("timing_row_count"),
                        joined_timing_frames=details.get("joined_timing_frames"),
                        joined_ratio=details.get("joined_ratio"),
                        timing_span_frames=details.get("timing_span_frames"),
                        timing_span_coverage_ratio=details.get(
                            "timing_span_coverage_ratio"
                        ),
                        raw_first_frame_i=details.get("raw_first_frame_i"),
                        raw_last_frame_i=details.get("raw_last_frame_i"),
                        timing_last_frame_i=details.get("timing_last_frame_i"),
                    )
                )
                progress(f"[excluded] {recording.prefix}: {exc}")
                continue
            reports.append(report)
            staged_pairs.append((staged_path, results_dir / staged_path.name))

        # Re-read the source after inference.  The private snapshot remains the
        # actual weights used, but a changed source path must never be activated.
        if f"sha256:{sha256_file(model_path)}" != model_id:
            raise BatchInferenceError(
                f"source model changed while batch inference was running: {model_path}"
            )

        successful_prefixes = [report.prefix for report in reports]
        success_audit = [
            {
                **asdict(report),
                "eligible": True,
                "exclusion_stage": None,
                "exclusion_reason": None,
            }
            for report in reports
        ]
        exclusion_audit = [asdict(item) for item in exclusions]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_path),
            "model_name": model_path.name,
            "model_hash": model_id,
            "model_size_bytes": model_size,
            "device": resolved_device,
            "half_precision": bool(
                use_half and resolved_device.lower().startswith("cuda:")
            ),
            "confidence_threshold": float(confidence_threshold),
            "allow_low_pose": bool(allow_low_pose),
            "allow_unmapped_video": bool(allow_unmapped_video),
            "front_class_name": front_class_name,
            "angle_column": "yaw_deg",
            "angle_domain": "heading",
            "valid_column": "pose_ok",
            "confidence_column": "confidence",
            "model_id_column": "model_hash",
            "results_suffix": suffix,
            "recordings_dir": str(recordings_dir),
            "results_dir": str(results_dir),
            "recording_count": len(reports),
            "input_recording_count": len(selected) + len(
                [item for item in exclusions if item.exclusion_stage == "input_discovery"]
            ),
            "excluded_recording_count": len(exclusions),
            "successful_recording_prefixes": successful_prefixes,
            "total_frames": sum(report.frame_count for report in reports),
            "total_valid_poses": sum(report.valid_poses for report in reports),
            "elapsed_sec": time.perf_counter() - batch_started,
            "recordings": [asdict(report) for report in reports],
            "excluded_recordings": exclusion_audit,
            "recording_audit": success_audit + exclusion_audit,
        }
        staged_manifest = staging_dir / manifest_filename
        _atomic_write_json(staged_manifest, manifest)
        staged_pairs.append((staged_manifest, results_dir / manifest_filename))

        # Manifest is deliberately the last pair: its presence means the whole
        # generation was committed, not merely that inference started.
        _publish_transaction(staged_pairs, overwrite=overwrite)
        progress(
            f"published {len(reports)} result CSV(s) and {manifest_filename}"
        )
        return manifest
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a .pt pose model over raw recordings and atomically publish "
            "yaw_deg/heading CSV files for the rotation fitter."
        )
    )
    parser.add_argument("--model", type=Path, required=True, help="new .pt weights")
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("extracted/depth_cam/rec"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("rotation_fit/out/new_inference"),
    )
    parser.add_argument("--results-suffix", default=DEFAULT_RESULTS_SUFFIX)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:N, or N (default: auto)",
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--front-class-name", default=DEFAULT_FRONT_CLASS_NAME)
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use FP16 on CUDA (ignored on CPU)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="process prefixes containing this token; may be repeated",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="atomically replace an older result set after full success",
    )
    parser.add_argument(
        "--allow-low-pose",
        action="store_true",
        help=(
            "evaluation mode: publish recordings with fewer than three valid "
            "poses instead of rejecting the calibration batch"
        ),
    )
    parser.add_argument(
        "--allow-unmapped-video",
        action="store_true",
        help=(
            "evaluation mode: infer raw frames even when their timing mapping "
            "is unusable; output cannot be used for response calibration"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_batch_inference(
            model_path=args.model,
            recordings_dir=args.recordings_dir,
            results_dir=args.results_dir,
            results_suffix=args.results_suffix,
            manifest_name=args.manifest_name,
            device=args.device,
            confidence_threshold=args.confidence,
            front_class_name=args.front_class_name,
            use_half=args.half,
            only=args.only,
            overwrite=args.overwrite,
            allow_low_pose=args.allow_low_pose,
            allow_unmapped_video=args.allow_unmapped_video,
        )
    except (BatchInferenceError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"batch inference failed: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "model_hash": manifest["model_hash"],
                "recording_count": manifest["recording_count"],
                "total_frames": manifest["total_frames"],
                "total_valid_poses": manifest["total_valid_poses"],
                "results_dir": manifest["results_dir"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
