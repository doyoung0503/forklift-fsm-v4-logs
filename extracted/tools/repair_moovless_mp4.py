"""Recover OpenCV MP4V recordings whose final ``moov`` atom is missing.

OpenCV's MP4 writer stores an MPEG-4 Part 2 elementary stream directly in the
``mdat`` atom. If a process is killed before ``VideoWriter.release()``, the
stream remains usable even though ordinary players cannot open the container.
This tool extracts that stream, decodes it, and writes a finalized MP4.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import struct
import tempfile

import cv2


def find_mdat_payload(path: Path) -> tuple[int, int]:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        offset = 0
        while offset + 8 <= file_size:
            stream.seek(offset)
            header = stream.read(8)
            atom_size, atom_type = struct.unpack(">I4s", header)
            header_size = 8
            if atom_size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    break
                atom_size = struct.unpack(">Q", extended)[0]
                header_size = 16
            elif atom_size == 0:
                atom_size = file_size - offset

            if atom_size < header_size or offset + atom_size > file_size:
                raise ValueError(f"invalid MP4 atom at byte {offset}")
            if atom_type == b"mdat":
                start = offset + header_size
                return start, offset + atom_size - start
            offset += atom_size
    raise ValueError("mdat atom not found")


def extract_range(source: Path, start: int, size: int, target: Path) -> None:
    with source.open("rb") as src, target.open("wb") as dst:
        src.seek(start)
        remaining = size
        while remaining:
            chunk = src.read(min(1024 * 1024, remaining))
            if not chunk:
                raise EOFError("unexpected end of mdat payload")
            dst.write(chunk)
            remaining -= len(chunk)


def repair(source: Path, target: Path, fps_override: float | None) -> int:
    payload_start, payload_size = find_mdat_payload(source)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="mp4_repair_"))
    elementary = temp_dir / "payload.m4v"
    temp_output = target.with_name(f".{target.stem}.writing{target.suffix}")
    capture = None
    writer = None
    try:
        extract_range(source, payload_start, payload_size, elementary)
        capture = cv2.VideoCapture(str(elementary))
        if not capture.isOpened():
            raise RuntimeError("the extracted MPEG-4 stream could not be opened")

        detected_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps_override if fps_override is not None else detected_fps
        if not 0.1 <= fps <= 240.0:
            raise ValueError(
                f"invalid detected fps {detected_fps!r}; pass --fps explicitly"
            )

        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("the extracted MPEG-4 stream has no decodable frames")
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(temp_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("failed to open the repaired MP4 writer")

        frame_count = 0
        while ok and frame is not None:
            writer.write(frame)
            frame_count += 1
            ok, frame = capture.read()
        writer.release()
        writer = None
        capture.release()
        capture = None

        check = cv2.VideoCapture(str(temp_output))
        check_opened = check.isOpened()
        check_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT)) if check_opened else 0
        first_ok, _ = check.read() if check_opened else (False, None)
        if check_opened and check_frames > 1:
            check.set(cv2.CAP_PROP_POS_FRAMES, check_frames - 1)
            last_ok, _ = check.read()
        else:
            last_ok = first_ok
        check.release()
        if not (check_opened and first_ok and last_ok and check_frames == frame_count):
            raise RuntimeError(
                "repaired MP4 validation failed "
                f"(written={frame_count}, reported={check_frames})"
            )

        os.replace(temp_output, target)
        print(
            f"recovered {frame_count} frames at {fps:.3f} fps, "
            f"{width}x{height}: {target}"
        )
        return frame_count
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if temp_output.exists():
            temp_output.unlink()
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--fps", type=float)
    args = parser.parse_args()
    repair(args.source.resolve(), args.target.resolve(), args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
