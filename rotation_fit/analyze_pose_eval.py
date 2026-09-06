"""Summarise batch pose inference quality for model/video evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterable, Optional


def _quantile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def _longest_failure_run(rows: Iterable[dict[str, str]]) -> dict[str, object]:
    best_reason = ""
    best_start = None
    best_end = None
    best_length = 0
    reason = ""
    start = None
    end = None
    length = 0
    for row in rows:
        current = row.get("failure_reason", "")
        frame = int(row["video_frame_i"])
        if current and current == reason and end is not None and frame == end + 1:
            end = frame
            length += 1
        elif current:
            reason, start, end, length = current, frame, frame, 1
        else:
            reason, start, end, length = "", None, None, 0
        if length > best_length:
            best_reason = reason
            best_start = start
            best_end = end
            best_length = length
    return {
        "reason": best_reason or None,
        "start_video_frame": best_start,
        "end_video_frame": best_end,
        "frames": best_length,
    }


def _analyse_csv(path: Path, prefix: str) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    detections = sum(row["model_det_ok"] == "1" for row in rows)
    valid = sum(row["pose_ok"] == "1" for row in rows)
    failures = Counter(row["failure_reason"] or "ok" for row in rows)
    confidence = [float(row["confidence"]) for row in rows if row["confidence"]]
    rms = [float(row["pose_rms_px"]) for row in rows if row["pose_rms_px"]]
    n_used = Counter(row["pose_n_used"] for row in rows if row["pose_n_used"])

    yaw_jumps: list[float] = []
    z_jumps: list[float] = []
    yaw_values: list[float] = []
    z_values: list[float] = []
    yaw_peak = {"from_video_frame": None, "to_video_frame": None, "value_deg": None}
    z_peak = {"from_video_frame": None, "to_video_frame": None, "value_m": None}
    previous: Optional[tuple[int, float, float]] = None
    for row in rows:
        if row["pose_ok"] != "1":
            previous = None
            continue
        frame = int(row["video_frame_i"])
        yaw = float(row["yaw_deg"])
        z_m = float(row["pos_z_m"])
        yaw_values.append(yaw)
        z_values.append(z_m)
        if previous is not None and frame == previous[0] + 1:
            yaw_delta = (yaw - previous[1] + 180.0) % 360.0 - 180.0
            yaw_jump = abs(yaw_delta)
            z_jump = abs(z_m - previous[2])
            yaw_jumps.append(yaw_jump)
            z_jumps.append(z_jump)
            if yaw_peak["value_deg"] is None or yaw_jump > yaw_peak["value_deg"]:
                yaw_peak = {
                    "from_video_frame": previous[0],
                    "to_video_frame": frame,
                    "value_deg": yaw_jump,
                }
            if z_peak["value_m"] is None or z_jump > z_peak["value_m"]:
                z_peak = {
                    "from_video_frame": previous[0],
                    "to_video_frame": frame,
                    "value_m": z_jump,
                }
        previous = (frame, yaw, z_m)

    frame_count = len(rows)
    date_match = re.search(r"_(20\d{6})_", prefix)
    return {
        "prefix": prefix,
        "date": date_match.group(1) if date_match else "unknown",
        "frames": frame_count,
        "detections": detections,
        "valid_poses": valid,
        "detection_rate": detections / frame_count if frame_count else 0.0,
        "valid_pose_rate": valid / frame_count if frame_count else 0.0,
        "pnp_success_per_detection": valid / detections if detections else 0.0,
        "failure_counts": dict(sorted(failures.items())),
        "confidence_median": median(confidence) if confidence else None,
        "confidence_p10": _quantile(confidence, 0.10),
        "reprojection_rms_median_px": median(rms) if rms else None,
        "reprojection_rms_p95_px": _quantile(rms, 0.95),
        "pose_points_used": dict(sorted(n_used.items())),
        "yaw_abs_jump_p95_deg": _quantile(yaw_jumps, 0.95),
        "yaw_abs_jump_max_deg": max(yaw_jumps) if yaw_jumps else None,
        "yaw_peak_jump": yaw_peak,
        "z_abs_jump_p95_m": _quantile(z_jumps, 0.95),
        "z_abs_jump_max_m": max(z_jumps) if z_jumps else None,
        "z_peak_jump": z_peak,
        "yaw_range_deg": (
            [min(yaw_values), max(yaw_values)] if yaw_values else None
        ),
        "z_range_m": [min(z_values), max(z_values)] if z_values else None,
        "longest_failure_run": _longest_failure_run(rows),
    }


def analyse(results_dir: Path) -> dict[str, object]:
    results_dir = results_dir.resolve()
    manifest_path = results_dir / "batch_inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = [
        _analyse_csv(results_dir / item["result_file"], item["prefix"])
        for item in manifest["recordings"]
    ]
    dates: dict[str, dict[str, object]] = {}
    for date in sorted({str(item["date"]) for item in videos}):
        group = [item for item in videos if item["date"] == date]
        frames = sum(int(item["frames"]) for item in group)
        detections = sum(int(item["detections"]) for item in group)
        valid = sum(int(item["valid_poses"]) for item in group)
        dates[date] = {
            "videos": len(group),
            "frames": frames,
            "detections": detections,
            "valid_poses": valid,
            "detection_rate": detections / frames if frames else 0.0,
            "valid_pose_rate": valid / frames if frames else 0.0,
            "pnp_success_per_detection": valid / detections if detections else 0.0,
        }
    return {
        "model_name": manifest["model_name"],
        "model_hash": manifest["model_hash"],
        "device": manifest["device"],
        "confidence_threshold": manifest["confidence_threshold"],
        "input_recording_count": manifest["input_recording_count"],
        "evaluated_recording_count": manifest["recording_count"],
        "excluded_recordings": manifest["excluded_recordings"],
        "total_frames": manifest["total_frames"],
        "total_valid_poses": manifest["total_valid_poses"],
        "overall_valid_pose_rate": (
            manifest["total_valid_poses"] / manifest["total_frames"]
        ),
        "by_date": dates,
        "videos": sorted(videos, key=lambda item: item["valid_pose_rate"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyse(args.results_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
