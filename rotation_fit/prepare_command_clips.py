#!/usr/bin/env python3
"""Prepare model-independent, lossless CAN-window recordings for batch inference.

Video playback time is NOT measurement time. Native frame_i, sensor timestamps,
host timestamps and complete CAN history remain available in the sidecar logs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    from .batch_infer_rotation_logs import (
        RecordingEligibilityError, build_frame_mapping,
        discover_recordings_with_audit, sha256_file,
    )
    from .rotation_log_fit import FitSettings, load_command_windows, _frame_analysis_timeline
except ImportError:
    from batch_infer_rotation_logs import (
        RecordingEligibilityError, build_frame_mapping,
        discover_recordings_with_audit, sha256_file,
    )
    from rotation_log_fit import FitSettings, load_command_windows, _frame_analysis_timeline

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "out" / "command_clips"


def select_frames(frame_ids, timing_rows, windows, *, margin_sec=0.5):
    """Union of full drive + baseline + coast windows, using host AND sensor clocks.

No model confidence, yaw, detection flags or measured motion affect selection.
Extra margin covers clock-fit and delivery differences; raw timing stays intact.
"""
    if not math.isfinite(margin_sec) or margin_sec < 0.5:
        raise ValueError("margin_sec must be finite and at least 0.5")
    settings = FitSettings()
    eligible = [w for w in windows if w.strength == settings.command_strength
                and w.start.source == "can_tx" and w.stop.source == "can_tx"]
    intervals = [(w.start.write_return_s - settings.pre_command_sec - margin_sec,
                  w.stop.write_return_s + settings.inertia_horizon_sec
                  + settings.inertia_window_sec + margin_sec) for w in eligible]
    rows = [r for r in timing_rows if r.get("camera_input_host_mono_ms", "").strip()]
    host = np.asarray([float(r["camera_input_host_mono_ms"]) / 1000 for r in rows])
    if not np.all(np.isfinite(host)) or np.any(np.diff(host) < 0):
        raise ValueError("invalid or non-monotonic host capture timestamps")
    sensor = np.asarray([float(r.get("camera_sensor_timestamp_ms") or "nan") for r in rows])
    mapped, clock, scale, rmse = _frame_analysis_timeline(
        host, sensor, [r.get("camera_timestamp_domain", "") for r in rows])
    selected_ids = set()
    for row, arrival, capture in zip(rows, host, mapped):
        if any(lo <= arrival <= hi or lo <= capture <= hi for lo, hi in intervals):
            selected_ids.add(int(float(row["frame_i"])))
    indices = [i for i, frame_i in enumerate(frame_ids) if frame_i in selected_ids]
    audit = {
        "command_windows": [dict(command=w.command, strength=w.strength,
                                 command_host_s=w.start.write_return_s,
                                 stop_host_s=w.stop.write_return_s,
                                 stop_movement=w.stop.movement,
                                 next_active_host_s=w.next_active_command_s,
                                 selection_start_host_s=lo, selection_end_host_s=hi)
                            for w, (lo, hi) in zip(eligible, intervals)],
        "clock_source": clock, "clock_scale": scale, "clock_rmse_ms": rmse,
        "timing_rows_without_host_timestamp": len(timing_rows) - len(rows),
    }
    return indices, audit


def plan_recording(recording, margin_sec):
    cap = cv2.VideoCapture(str(recording.video_path))
    try:
        if not cap.isOpened():
            raise RecordingEligibilityError("cannot open source video")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    mapping = build_frame_mapping(recording.timing_path, recording.meta_path, count)
    with recording.timing_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames or []), list(reader)
    windows = load_command_windows(recording.control_path, FitSettings())
    indices, audit = select_frames(mapping.frame_ids, rows, windows, margin_sec=margin_sec)
    if not indices:
        raise RecordingEligibilityError("no timed frames around actual CAN strength-30 rotation commands")
    return dict(recording=recording, frame_ids=mapping.frame_ids, indices=indices,
                source_frames=count, width=width, height=height, fps=fps,
                fields=fields, rows=rows, audit=audit)


def export_recording(plan, target):
    recording = plan["recording"]
    indices = plan["indices"]
    wanted = set(indices)
    video = target / f"{recording.prefix}_raw.avi"
    fps = plan["fps"] if math.isfinite(plan["fps"]) and plan["fps"] > 0 else 30.0
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), fps,
                             (plan["width"], plan["height"]))
    if not writer.isOpened():
        raise RuntimeError("FFV1 lossless encoder unavailable")
    cap = cv2.VideoCapture(str(recording.video_path))
    digest = hashlib.sha256()
    written = 0
    try:
        for index in range(plan["source_frames"]):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"source decode failed at {recording.prefix}:{index}")
            if index in wanted:
                writer.write(frame)
                digest.update(frame.tobytes())
                written += 1
        if cap.read()[0]:
            raise RuntimeError("source has more frames than its declared frame count")
    finally:
        cap.release()
        writer.release()
    # Verify EVERY output pixel and frame, not just the container frame count.
    check = cv2.VideoCapture(str(video))
    actual_digest = hashlib.sha256()
    count = 0
    try:
        while True:
            ok, frame = check.read()
            if not ok:
                break
            actual_digest.update(frame.tobytes())
            count += 1
    finally:
        check.release()
    if count != written or actual_digest.digest() != digest.digest():
        raise RuntimeError(f"lossless pixel verification failed: {recording.prefix}")
    mapping = {plan["frame_ids"][index]: (out_i, index)
               for out_i, index in enumerate(indices)}
    fields = list(plan["fields"])
    for field in ("raw_video_frame_index", "source_raw_video_frame_index"):
        if field not in fields:
            fields.append(field)
    timing = target / recording.timing_path.name
    with timing.open("x", encoding="utf-8", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        writer_csv.writeheader()
        for original in plan["rows"]:
            row = dict(original)
            pair = mapping.get(int(float(row["frame_i"])))
            row["raw_video_frame_index"] = "" if pair is None else pair[0]
            row["source_raw_video_frame_index"] = "" if pair is None else pair[1]
            writer_csv.writerow(row)
    shutil.copy2(recording.meta_path, target / recording.meta_path.name)
    shutil.copy2(recording.control_path, target / recording.control_path.name)
    rebuilt = build_frame_mapping(timing, target / recording.meta_path.name, written)
    if rebuilt.frame_ids != tuple(plan["frame_ids"][i] for i in indices):
        raise RuntimeError("output frame-id mapping differs from source")
    return dict(prefix=recording.prefix, source_video=str(recording.video_path),
                source_frames=plan["source_frames"], selected_frames=written,
                video_file=video.name, video_bytes=video.stat().st_size,
                video_sha256=sha256_file(video), decoded_pixels_sha256=digest.hexdigest(),
                source_hashes={p.name: sha256_file(p) for p in (
                    recording.video_path, recording.timing_path,
                    recording.meta_path, recording.control_path)},
                **plan["audit"])


def prepare(recordings_dir, output_dir, *, margin_sec=0.5, dry_run=False):
    recordings_dir, output_dir = recordings_dir.resolve(), output_dir.resolve()
    if output_dir.exists() and not dry_run:
        raise FileExistsError(f"output already exists; choose a new directory: {output_dir}")
    recordings, exclusions = discover_recordings_with_audit(recordings_dir)
    excluded = [asdict(item) for item in exclusions]
    plans = []
    source_total = 0
    for recording in recordings:
        try:
            plan = plan_recording(recording, margin_sec)
            plans.append(plan)
            source_total += plan["source_frames"]
        except (RecordingEligibilityError, ValueError) as exc:
            excluded.append(dict(prefix=recording.prefix, exclusion_reason=str(exc)))
    selected_total = sum(len(p["indices"]) for p in plans)
    report = dict(schema_version=1, status="PLANNED", created_utc=datetime.now(timezone.utc).isoformat(),
                  source_dir=str(recordings_dir), output_dir=str(output_dir),
                  selection="actual CAN strength 30; model-independent; host OR affine sensor timeline",
                  baseline_sec=1.0, inertia_horizon_sec=2.0, inertia_window_sec=0.2,
                  margin_sec=margin_sec, codec="FFV1", original_timestamps_preserved=True,
                  source_frames=source_total, selected_frames=selected_total,
                  avoided_inference_fraction=1 - selected_total / source_total if source_total else 0,
                  recording_count=len(plans), excluded=excluded, recordings=[])
    print(json.dumps({k: v for k, v in report.items() if k != "recordings"}, ensure_ascii=False), flush=True)
    if dry_run:
        return report
    if not plans:
        raise ValueError("no eligible command windows to export")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    # Failed/interrupted work stays under its private name, never a ready dataset.
    try:
        for number, plan in enumerate(plans, 1):
            print(f'[{number}/{len(plans)}] {plan["recording"].prefix}: '
                  f'{len(plan["indices"])}/{plan["source_frames"]} frames', flush=True)
            report["recordings"].append(export_recording(plan, staging))
        report["status"] = "COMPLETE"
        (staging / "command_clips_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output_dir)
    except BaseException:
        print(f"Incomplete export retained at: {staging}", flush=True)
        raise
    print(f"READY: {output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-dir", type=Path, default=HERE.parent / "extracted/depth_cam/rec")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--margin-sec", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prepare(args.recordings_dir, args.output_dir, margin_sec=args.margin_sec, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
