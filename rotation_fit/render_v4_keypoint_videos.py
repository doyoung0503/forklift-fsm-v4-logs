"""Render raw model bbox/keypoint predictions only, without any PnP output."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from batch_infer_rotation_logs import YoloPosePredictor, sha256_file


FRONT = (0, 255, 255)
REAR = (255, 190, 0)
CENTER = (255, 70, 255)
BOX = (60, 230, 100)
TEXT = (240, 240, 240)


def _text(image, value, point, color=TEXT, scale=0.5, thickness=1):
    cv2.putText(
        image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv2.LINE_AA,
    )


def _draw_prediction(
    image: np.ndarray,
    keypoints: np.ndarray,
    bbox: np.ndarray | None,
    *,
    transform=lambda x, y: (x, y),
    detailed: bool = False,
) -> None:
    points = np.asarray(keypoints[:9], dtype=float)
    if bbox is not None:
        x1, y1 = transform(float(bbox[0]), float(bbox[1]))
        x2, y2 = transform(float(bbox[2]), float(bbox[3]))
        cv2.rectangle(
            image, (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))), BOX, 2, cv2.LINE_AA,
        )

    transformed = np.asarray([transform(point[0], point[1]) for point in points])
    for indices, color in (((0, 1, 2, 3), FRONT), ((4, 5, 6, 7), REAR)):
        polygon = np.round(transformed[list(indices)]).astype(np.int32)
        cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
    for front, rear in zip(range(4), range(4, 8)):
        cv2.line(
            image, tuple(np.round(transformed[front]).astype(int)),
            tuple(np.round(transformed[rear]).astype(int)),
            (170, 170, 170), 1, cv2.LINE_AA,
        )
    for index, (point, mapped) in enumerate(zip(points, transformed)):
        x, y = map(int, np.round(mapped))
        visibility = float(point[2]) if point.size >= 3 else 1.0
        color = FRONT if index < 4 else REAR
        if index == 8:
            color = CENTER
            cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        else:
            cv2.circle(image, (x, y), 6 if detailed else 5, color, -1, cv2.LINE_AA)
        label = (
            f"#{index}  xy=({point[0]:.0f},{point[1]:.0f})  v={visibility:.2f}"
            if detailed else f"#{index}"
        )
        _text(
            image, label, (x + 7, y - 7), color,
            0.46 if detailed else 0.48, 1 if not detailed else 2,
        )


def _zoom_panel(
    frame: np.ndarray,
    keypoints: np.ndarray,
    bbox: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    x1, y1, x2, y2 = map(float, bbox)
    box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    margin_x = max(30.0, box_w * 0.35)
    margin_y = max(30.0, box_h * 1.25)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    crop_w = max(180.0, box_w + 2.0 * margin_x)
    crop_h = max(135.0, box_h + 2.0 * margin_y)
    crop_w = max(crop_w, crop_h * width / height)
    crop_h = max(crop_h, crop_w * height / width)
    left = max(0, int(math.floor(cx - crop_w * 0.5)))
    right = min(frame.shape[1], int(math.ceil(cx + crop_w * 0.5)))
    top = max(0, int(math.floor(cy - crop_h * 0.5)))
    bottom = min(frame.shape[0], int(math.ceil(cy + crop_h * 0.5)))
    crop = frame[top:bottom, left:right]
    panel = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
    sx, sy = width / max(1, right - left), height / max(1, bottom - top)

    def transform(x, y):
        return (x - left) * sx, (y - top) * sy

    _draw_prediction(panel, keypoints, bbox, transform=transform, detailed=False)
    _text(panel, "MODEL KEYPOINT ZOOM", (10, 24), (255, 255, 255), 0.58, 2)
    return panel


def _draw_keypoint_table(canvas: np.ndarray, keypoints: np.ndarray, top: int) -> None:
    _text(
        canvas, "RAW KEYPOINT VALUES (input-image pixels; v = model visibility)",
        (12, top + 24), TEXT, 0.52, 1,
    )
    points = np.asarray(keypoints[:9], dtype=float)
    column_width = canvas.shape[1] // 3
    for index, point in enumerate(points):
        column = index // 3
        row = index % 3
        x = 14 + column * column_width
        y = top + 55 + row * 29
        visibility = float(point[2]) if point.size >= 3 else 1.0
        color = FRONT if index < 4 else REAR
        if index == 8:
            color = CENTER
        _text(
            canvas,
            f"#{index}   x={point[0]:7.1f}   y={point[1]:7.1f}   v={visibility:.3f}",
            (x, y), color, 0.50, 1,
        )


def render(video: Path, output: Path, predictor: YoloPosePredictor) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.rendering.mp4")
    temporary.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (width * 2, height + 220),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create {temporary}")

    decoded = detections = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            detection = predictor.predict(frame)
            full = frame.copy()
            if detection.keypoints is not None:
                detections += 1
                _draw_prediction(full, detection.keypoints, detection.bbox_xyxy)
                zoom = _zoom_panel(
                    frame, detection.keypoints, detection.bbox_xyxy,
                    width, height,
                )
                state = f"DETECTED  conf={detection.confidence:.3f}"
                state_color = BOX
            else:
                zoom = np.full_like(frame, 18)
                _text(zoom, "NO MODEL DETECTION", (40, height // 2), (70, 90, 255), 0.85, 2)
                state = "NO DETECTION"
                state_color = (70, 90, 255)
            canvas = np.zeros((height + 220, width * 2, 3), dtype=np.uint8)
            canvas[70:70 + height, :width] = full
            canvas[70:70 + height, width:] = zoom
            _text(canvas, f"{video.stem} | frame {decoded}/{frame_count - 1}", (10, 22), TEXT, 0.55)
            _text(canvas, state, (10, 48), state_color, 0.62, 2)
            _text(canvas, "RAW MODEL OUTPUT ONLY - NO PnP / NO TEMPORAL FILTER", (330, 48), (80, 190, 255), 0.54, 2)
            _text(
                canvas,
                "0-3 FRONT: LT RT RB LB   |   4-7 REAR: LT RT RB LB   |   8 CENTER",
                (width + 10, 22), TEXT, 0.47,
            )
            if detection.keypoints is not None:
                _draw_keypoint_table(canvas, detection.keypoints, 70 + height)
            else:
                _text(
                    canvas, "No bbox or keypoints were emitted by the model.",
                    (14, 70 + height + 55), (70, 90, 255), 0.58, 2,
                )
            writer.write(canvas)
            decoded += 1
            if decoded % 250 == 0:
                print(f"[{video.stem}] {decoded}/{frame_count}", flush=True)
    finally:
        writer.release()
        capture.release()
    if decoded != frame_count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"decoded {decoded}/{frame_count} frames from {video.name}")
    os.replace(temporary, output)
    return {
        "input": str(video.resolve()),
        "output": str(output.resolve()),
        "frames": decoded,
        "fps": fps,
        "detections": detections,
        "detection_rate": detections / decoded if decoded else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--recordings-dir", type=Path, default=Path("extracted/depth_cam/rec"))
    parser.add_argument("--output-dir", type=Path, default=Path("rotation_fit/out/v4_keypoint_videos"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model = args.model.resolve()
    recordings = args.recordings_dir.resolve()
    output_dir = args.output_dir.resolve()
    videos = sorted(recordings.glob("*_raw.mp4"))
    if args.only:
        videos = [video for video in videos if any(token in video.name for token in args.only)]
    predictor = YoloPosePredictor(
        model,
        device=args.device,
        confidence_threshold=args.confidence,
        front_class_name="front",
        use_half=args.device != "cpu",
    )
    reports = []
    for number, video in enumerate(videos, 1):
        prefix = video.name[: -len("_raw.mp4")]
        output = output_dir / f"{prefix}_v4_keypoints_only.mp4"
        if output.exists() and not args.overwrite:
            print(f"[{number}/{len(videos)}] skip existing {output.name}", flush=True)
            continue
        print(f"[{number}/{len(videos)}] render {video.name}", flush=True)
        reports.append(render(video, output, predictor))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model),
        "model_hash": f"sha256:{sha256_file(model)}",
        "device": predictor.device,
        "confidence": args.confidence,
        "contains_pnp": False,
        "video_count": len(reports),
        "videos": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "keypoint_visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "videos": len(reports), "output": str(output_dir)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
