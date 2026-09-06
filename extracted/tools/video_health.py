"""Inspect MP4 container atoms and verify that OpenCV can decode the video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def contains_atom(path: Path, atom: bytes) -> bool:
    overlap = max(0, len(atom) - 1)
    tail = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            data = tail + chunk
            if atom in data:
                return True
            tail = data[-overlap:] if overlap else b""
    return False


def inspect(path: Path, full_decode: bool = False) -> dict[str, object]:
    size = path.stat().st_size
    has_ftyp = contains_atom(path, b"ftyp")
    has_mdat = contains_atom(path, b"mdat")
    has_moov = contains_atom(path, b"moov")

    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    first_ok = False
    last_ok = False
    decoded_frames = None
    if opened:
        if full_decode:
            decoded_frames = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded_frames += 1
            first_ok = decoded_frames > 0
            last_ok = decoded_frames == frame_count
        else:
            first_ok, _ = capture.read()
            if frame_count > 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                last_ok, _ = capture.read()
            else:
                last_ok = first_ok
    capture.release()

    healthy = bool(
        size > 0
        and has_ftyp
        and has_mdat
        and has_moov
        and opened
        and frame_count > 0
        and first_ok
        and last_ok
    )
    return {
        "path": str(path),
        "size": size,
        "healthy": healthy,
        "ftyp": has_ftyp,
        "mdat": has_mdat,
        "moov": has_moov,
        "opened": opened,
        "frames": frame_count,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "first_ok": bool(first_ok),
        "last_ok": bool(last_ok),
        "decoded_frames": decoded_frames,
    }


def iter_videos(inputs: list[Path]) -> list[Path]:
    videos: set[Path] = set()
    for item in inputs:
        if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES:
            videos.add(item.resolve())
        elif item.is_dir():
            videos.update(
                path.resolve()
                for path in item.rglob("*")
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
            )
    return sorted(videos)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--only-bad", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    bad = 0
    for path in iter_videos(args.paths):
        result = inspect(path, full_decode=args.full)
        if not result["healthy"]:
            bad += 1
        if not args.only_bad or not result["healthy"]:
            print(json.dumps(result, ensure_ascii=False))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
