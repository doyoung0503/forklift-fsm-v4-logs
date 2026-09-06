"""Render v4 pose-model detections and PnP values onto recorded raw videos."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from batch_infer_rotation_logs import (
    YoloPosePredictor,
    load_camera_metadata,
    load_live_pose_solver,
    sha256_file,
)


FRONT_COLOR = (0, 255, 255)
BACK_COLOR = (255, 190, 0)
CENTER_COLOR = (255, 80, 255)
OK_COLOR = (80, 230, 110)
FAIL_COLOR = (70, 90, 255)


def _draw_text(image, text, origin, color=(240, 240, 240), scale=0.52, thickness=1):
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv2.LINE_AA,
    )


def _draw_keypoints(image: np.ndarray, keypoints: np.ndarray) -> None:
    points = np.asarray(keypoints[:9], dtype=float)
    for indices, color in (((0, 1, 2, 3), FRONT_COLOR), ((4, 5, 6, 7), BACK_COLOR)):
        polygon = np.round(points[list(indices), :2]).astype(np.int32)
        cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
    for front, back in zip(range(4), range(4, 8)):
        a = tuple(np.round(points[front, :2]).astype(int))
        b = tuple(np.round(points[back, :2]).astype(int))
        cv2.line(image, a, b, (180, 180, 180), 1, cv2.LINE_AA)
    for index, point in enumerate(points):
        x, y = int(round(point[0])), int(round(point[1]))
        visibility = float(point[2]) if point.size >= 3 else 1.0
        color = FRONT_COLOR if index < 4 else BACK_COLOR
        if index == 8:
            color = CENTER_COLOR
            cv2.drawMarker(
                image, (x, y), color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA,
            )
        else:
            cv2.circle(image, (x, y), 5, color, -1, cv2.LINE_AA)
        _draw_text(image, f"{index}:{visibility:.2f}", (x + 6, y - 5), color, 0.42)

    finite = points[np.isfinite(points[:, :2]).all(axis=1), :2]
    if len(finite):
        x, y, width, height = cv2.boundingRect(np.round(finite).astype(np.int32))
        cv2.rectangle(image, (x, y), (x + width, y + height), OK_COLOR, 1)


def render_video(
    video_path: Path,
    meta_path: Path,
    output_path: Path,
    predictor: YoloPosePredictor,
    pose_solver,
) -> dict[str, object]:
    intrinsics, geometry = load_camera_metadata(meta_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.rendering.mp4")
    if temporary.exists():
        temporary.unlink()
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create {temporary}")

    started = time.perf_counter()
    detections = poses = yaw_outliers = z_outliers = 0
    previous_pose = None
    decoded = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok or image is None:
                break
            detection = predictor.predict(image)
            pose = None
            if detection.keypoints is not None:
                detections += 1
                _draw_keypoints(image, detection.keypoints)
                pose = pose_solver(detection.keypoints, intrinsics, geometry)
            if pose is not None:
                poses += 1

            yaw_jump = z_jump = None
            if pose is not None and previous_pose is not None:
                yaw_jump = abs((pose.yaw_deg - previous_pose.yaw_deg + 180.0) % 360.0 - 180.0)
                z_jump = abs(pose.pos_z_m - previous_pose.pos_z_m)
                yaw_outliers += int(yaw_jump > 20.0)
                z_outliers += int(z_jump > 1.0)
            is_outlier = (
                (yaw_jump is not None and yaw_jump > 20.0)
                or (z_jump is not None and z_jump > 1.0)
            )

            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (width, 84), (10, 13, 18), -1)
            cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
            status = "NO DETECTION"
            status_color = FAIL_COLOR
            details = "conf=-  yaw=-  x=-  z=-  PnP=-"
            if detection.keypoints is not None:
                status = "DETECTED / PnP FAILED"
                details = f"conf={detection.confidence:.3f}  yaw=-  x=-  z=-  PnP=failed"
            if pose is not None:
                status = "POSE OK"
                status_color = OK_COLOR
                rms = "-" if pose.rms_px is None else f"{pose.rms_px:.2f}px"
                details = (
                    f"conf={detection.confidence:.3f}  yaw={pose.yaw_deg:+.2f}deg  "
                    f"x={pose.pos_x_m:+.2f}m  z={pose.pos_z_m:.2f}m  "
                    f"PnP={pose.n_used}pts/{rms}"
                )
            _draw_text(image, f"{video_path.stem} | frame {decoded}/{frame_count - 1}", (10, 20))
            _draw_text(image, status, (10, 43), status_color, 0.58, 2)
            _draw_text(image, details, (10, 67), (235, 235, 235), 0.48)
            if is_outlier:
                cv2.rectangle(image, (2, 2), (width - 3, height - 3), FAIL_COLOR, 5)
                warning = f"RAW POSE JUMP  dyaw={yaw_jump:.1f}deg  dz={z_jump:.2f}m"
                _draw_text(image, warning, (10, height - 18), FAIL_COLOR, 0.60, 2)
            else:
                _draw_text(
                    image, "front=yellow  back=cyan  center=magenta",
                    (10, height - 14), (220, 220, 220), 0.43,
                )
            writer.write(image)
            decoded += 1
            previous_pose = pose
            if decoded % 250 == 0:
                print(f"[{video_path.stem}] {decoded}/{frame_count}", flush=True)
    finally:
        writer.release()
        capture.release()

    if decoded != frame_count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"decoded {decoded}/{frame_count} frames from {video_path.name}")
    os.replace(temporary, output_path)
    return {
        "input": str(video_path.resolve()),
        "output": str(output_path.resolve()),
        "frames": decoded,
        "fps": fps,
        "detections": detections,
        "poses": poses,
        "valid_pose_rate": poses / decoded if decoded else 0.0,
        "yaw_jump_gt20": yaw_outliers,
        "z_jump_gt1m": z_outliers,
        "elapsed_sec": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--recordings-dir", type=Path, default=Path("extracted/depth_cam/rec"))
    parser.add_argument("--output-dir", type=Path, default=Path("rotation_fit/out/v4_visualized_videos"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_path = args.model.resolve()
    recordings_dir = args.recordings_dir.resolve()
    output_dir = args.output_dir.resolve()
    videos = sorted(recordings_dir.glob("*_raw.mp4"))
    if args.only:
        videos = [video for video in videos if any(token in video.name for token in args.only)]
    predictor = YoloPosePredictor(
        model_path,
        device=args.device,
        confidence_threshold=args.confidence,
        front_class_name="front",
        use_half=args.device != "cpu",
    )
    pose_solver = load_live_pose_solver()
    reports = []
    for number, video in enumerate(videos, 1):
        prefix = video.name[: -len("_raw.mp4")]
        output = output_dir / f"{prefix}_v4_pose_visualized.mp4"
        if output.exists() and not args.overwrite:
            print(f"[{number}/{len(videos)}] skip existing {output.name}", flush=True)
            continue
        print(f"[{number}/{len(videos)}] render {video.name}", flush=True)
        reports.append(render_video(
            video,
            recordings_dir / f"{prefix}_meta.json",
            output,
            predictor,
            pose_solver,
        ))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "model_hash": f"sha256:{sha256_file(model_path)}",
        "device": predictor.device,
        "confidence": args.confidence,
        "video_count": len(reports),
        "videos": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "videos": len(reports), "output": str(output_dir)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
