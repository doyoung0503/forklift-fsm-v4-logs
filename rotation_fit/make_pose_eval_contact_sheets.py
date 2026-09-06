"""Create raw-frame contact sheets around the longest pose failure runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def _read_frame(video: Path, index: int):
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        return image if ok else None
    finally:
        capture.release()


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    encoded.tofile(path)


def make_sheets(
    results_dir: Path, recordings_dir: Path, worst: int, mode: str,
    model_path: Path | None = None, device: str = "cuda:0",
) -> list[Path]:
    summary = json.loads(
        (results_dir / "pose_quality_summary.json").read_text(encoding="utf-8")
    )
    output_dir = results_dir / "inspection"
    saved: list[Path] = []
    predictor = None
    if model_path is not None:
        from batch_infer_rotation_logs import YoloPosePredictor

        predictor = YoloPosePredictor(
            model_path,
            device=device,
            confidence_threshold=float(summary["confidence_threshold"]),
            front_class_name="front",
            use_half=device != "cpu",
        )
    selected = summary["videos"]
    if mode == "jumps":
        selected = sorted(
            selected,
            key=lambda item: item.get("yaw_abs_jump_max_deg") or 0.0,
            reverse=True,
        )
    for video_info in selected[:worst]:
        prefix = video_info["prefix"]
        video = recordings_dir / f"{prefix}_raw.mp4"
        csv_path = results_dir / f"{prefix}_new_pose.csv"
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        row_by_frame = {int(row["video_frame_i"]): row for row in rows}
        run = video_info["longest_failure_run"]
        if mode == "jumps":
            yaw_peak = video_info["yaw_peak_jump"]
            z_peak = video_info["z_peak_jump"]
            yaw_from = int(yaw_peak["from_video_frame"])
            yaw_to = int(yaw_peak["to_video_frame"])
            z_from = int(z_peak["from_video_frame"])
            z_to = int(z_peak["to_video_frame"])
            picks = [
                max(0, yaw_from - 1), yaw_from, yaw_to,
                z_from, z_to, min(len(rows) - 1, z_to + 1),
            ]
        elif run["frames"]:
            start = int(run["start_video_frame"])
            end = int(run["end_video_frame"])
            picks = [
                max(0, start - 1),
                start,
                (start + end) // 2,
                end,
                min(len(rows) - 1, end + 1),
                len(rows) // 2,
            ]
        else:
            picks = np.linspace(0, len(rows) - 1, 6).astype(int).tolist()
        picks = list(dict.fromkeys(picks))
        for candidate in np.linspace(0, len(rows) - 1, 6).astype(int):
            if len(picks) >= 6:
                break
            if int(candidate) not in picks:
                picks.append(int(candidate))

        cells: list[np.ndarray] = []
        for frame_index in picks[:6]:
            image = _read_frame(video, frame_index)
            if image is None:
                image = np.zeros((480, 640, 3), dtype=np.uint8)
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
            if predictor is not None:
                detection = predictor.predict(image)
                if detection.keypoints is not None:
                    for keypoint_index, keypoint in enumerate(detection.keypoints[:9]):
                        x, y = int(round(float(keypoint[0]))), int(round(float(keypoint[1])))
                        visibility = float(keypoint[2]) if len(keypoint) > 2 else 1.0
                        point_color = (0, 255, 255) if keypoint_index < 4 else (255, 180, 0)
                        cv2.circle(image, (x, y), 5, point_color, -1, cv2.LINE_AA)
                        cv2.putText(
                            image, f"{keypoint_index}:{visibility:.2f}", (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, point_color, 1, cv2.LINE_AA,
                        )
            row = row_by_frame.get(frame_index, {})
            status = "POSE OK" if row.get("pose_ok") == "1" else (
                row.get("failure_reason") or "NO ROW"
            )
            color = (80, 230, 120) if status == "POSE OK" else (80, 100, 255)
            cv2.rectangle(image, (0, 0), (640, 46), (15, 18, 22), -1)
            cv2.putText(
                image, f"frame {frame_index} | {status}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
            )
            details = (
                f"conf={row.get('confidence') or '-'}  "
                f"yaw={row.get('yaw_deg') or '-'}  z={row.get('pos_z_m') or '-'}"
            )
            cv2.putText(
                image, details, (10, 39), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (230, 230, 230), 1, cv2.LINE_AA,
            )
            cells.append(image)
        while len(cells) < 6:
            cells.append(np.zeros((480, 640, 3), dtype=np.uint8))
        sheet = np.vstack((np.hstack(cells[:3]), np.hstack(cells[3:6])))
        destination = output_dir / f"{prefix}_{mode}_contact.jpg"
        _save_image(destination, sheet)
        saved.append(destination)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("recordings_dir", type=Path)
    parser.add_argument("--worst", type=int, default=5)
    parser.add_argument("--mode", choices=("failure", "jumps"), default="failure")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    saved = make_sheets(
        args.results_dir.resolve(), args.recordings_dir.resolve(), args.worst,
        args.mode, None if args.model is None else args.model.resolve(), args.device,
    )
    print("\n".join(str(path) for path in saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
