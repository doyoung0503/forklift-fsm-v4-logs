from __future__ import annotations

import csv
import io
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import cv2


WORKSPACE = Path(r"C:\Users\DELL\Downloads\로그수집")
REC_DIR = WORKSPACE / "extracted" / "depth_cam" / "rec"
OUTPUT_ZIP = WORKSPACE / "forklift_20260903_every_10_frames.zip"
FRAME_INTERVAL = 10


def main() -> None:
    videos = sorted(REC_DIR.glob("forklift_v4_recording_20260903_*_raw.mp4"))
    if not videos:
        raise RuntimeError("no 2026-09-03 raw videos found")
    if OUTPUT_ZIP.exists():
        raise FileExistsError(OUTPUT_ZIP)

    manifest: list[list[str]] = []
    video_summaries: list[tuple[str, int, int]] = []
    total_decoded = 0
    total_extracted = 0

    with ZipFile(OUTPUT_ZIP, "x", compression=ZIP_STORED, allowZip64=True) as archive:
        for video in videos:
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open video: {video}")
            frame_index = 0
            extracted = 0
            session = video.stem.removesuffix("_raw")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % FRAME_INTERVAL == 0:
                    encoded_ok, encoded = cv2.imencode(
                        ".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]
                    )
                    if not encoded_ok:
                        raise RuntimeError(
                            f"PNG encode failed: {video.name}, frame {frame_index}"
                        )
                    arcname = f"images/{session}_frame_{frame_index:06d}.png"
                    archive.writestr(arcname, encoded.tobytes())
                    manifest.append([
                        arcname,
                        video.name,
                        str(frame_index),
                        str(FRAME_INTERVAL),
                    ])
                    extracted += 1
                    total_extracted += 1
                    if total_extracted % 100 == 0:
                        print(
                            f"extracted {total_extracted} images; "
                            f"current={video.name} frame={frame_index}",
                            flush=True,
                        )
                frame_index += 1
            cap.release()
            total_decoded += frame_index
            expected = (frame_index + FRAME_INTERVAL - 1) // FRAME_INTERVAL
            if extracted != expected:
                raise RuntimeError(
                    f"extraction count mismatch for {video.name}: "
                    f"{extracted} != {expected}"
                )
            video_summaries.append((video.name, frame_index, extracted))

        manifest_buffer = io.StringIO(newline="")
        writer = csv.writer(manifest_buffer)
        writer.writerow([
            "archive_path", "source_video", "source_frame_index", "frame_interval"
        ])
        writer.writerows(manifest)
        archive.writestr("MANIFEST.csv", "\ufeff" + manifest_buffer.getvalue())

        readme_lines = [
            "촬영일: 2026-09-03",
            "대상: rec 폴더의 2026-09-03 _raw.mp4 원본 영상",
            "추출 규칙: 각 영상에서 0번 프레임부터 10프레임마다 1장",
            "출력 형식: PNG",
            f"원본 영상: {len(videos)}개",
            f"전체 디코딩 프레임: {total_decoded}장",
            f"추출 이미지: {total_extracted}장",
            "",
            "영상별 수량:",
        ]
        readme_lines.extend(
            f"- {name}: 원본 {decoded}프레임, 추출 {extracted}장"
            for name, decoded, extracted in video_summaries
        )
        archive.writestr("README.txt", "\n".join(readme_lines) + "\n")

    with ZipFile(OUTPUT_ZIP, "r") as archive:
        image_entries = [
            item for item in archive.infolist() if item.filename.startswith("images/")
        ]
        if len(image_entries) != total_extracted:
            raise RuntimeError(
                f"archive image count mismatch: {len(image_entries)} != {total_extracted}"
            )
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"CRC failure: {bad}")

    print(f"videos={len(videos)}", flush=True)
    print(f"decoded_frames={total_decoded}", flush=True)
    print(f"extracted_images={total_extracted}", flush=True)
    print(f"output={OUTPUT_ZIP}", flush=True)
    print(f"bytes={OUTPUT_ZIP.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
