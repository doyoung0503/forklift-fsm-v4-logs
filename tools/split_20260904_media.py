from __future__ import annotations

import csv
import io
import json
from datetime import date
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import cv2


WORKSPACE = Path(r"C:\Users\DELL\Downloads\로그수집")
REC_DIR = WORKSPACE / "extracted" / "depth_cam" / "rec"
IMAGE_DIR = Path(
    r"C:\Users\DELL\Documents\GitHub\pallet-6d-before"
    r"\data\pallet\real_data\real_unlabeled"
)
OUTPUT_DIR = WORKSPACE / "9-4_2way_split"
CAPTURE_DATE = date(2026, 9, 4)
PEOPLE = ("김민재", "김지훈")


def split_sizes(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if i < remainder else 0) for i in range(parts)]


def owner_for_index(index: int, sizes: list[int]) -> int:
    boundary = 0
    for owner, size in enumerate(sizes):
        boundary += size
        if index < boundary:
            return owner
    raise IndexError(index)


def video_frame_counts(videos: list[Path]) -> list[int]:
    counts: list[int] = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video}")
        count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
        if count <= 0:
            raise RuntimeError(f"invalid frame count for video: {video}")
        counts.append(count)
    return counts


def main() -> None:
    videos = sorted(REC_DIR.glob("forklift_v4_recording_20260904_*_raw.mp4"))
    images = sorted(
        p
        for p in IMAGE_DIR.glob("*.png")
        if date.fromtimestamp(p.stat().st_mtime) == CAPTURE_DATE
    )
    if not videos:
        raise RuntimeError("no 2026-09-04 raw videos found")
    if not images:
        raise RuntimeError("no 2026-09-04 captured PNG images found")

    counts = video_frame_counts(videos)
    total_video_frames = sum(counts)
    video_sizes = split_sizes(total_video_frames, len(PEOPLE))
    image_sizes = split_sizes(len(images), len(PEOPLE))
    total_sizes = [video_sizes[i] + image_sizes[i] for i in range(len(PEOPLE))]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_paths = [OUTPUT_DIR / f"{person}_9-4.zip" for person in PEOPLE]
    existing = [path for path in zip_paths if path.exists()]
    if existing:
        raise FileExistsError("output already exists: " + ", ".join(map(str, existing)))

    print(json.dumps({
        "videos": len(videos),
        "video_frames": total_video_frames,
        "captured_images": len(images),
        "video_split": video_sizes,
        "image_split": image_sizes,
        "total_split": total_sizes,
    }, ensure_ascii=False), flush=True)

    archives: list[ZipFile] = []
    manifests: list[list[list[str]]] = [[] for _ in PEOPLE]
    written_video = [0] * len(PEOPLE)
    written_images = [0] * len(PEOPLE)
    try:
        archives = [ZipFile(path, "x", compression=ZIP_STORED, allowZip64=True) for path in zip_paths]
        global_frame_index = 0
        for video, expected_count in zip(videos, counts):
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open video: {video}")
            decoded = 0
            session = video.stem.removesuffix("_raw")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                owner = owner_for_index(global_frame_index, video_sizes)
                encoded_ok, encoded = cv2.imencode(
                    ".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]
                )
                if not encoded_ok:
                    raise RuntimeError(f"PNG encode failed: {video}, frame {decoded}")
                arcname = f"video_frames/{session}_frame_{decoded:06d}.png"
                archives[owner].writestr(arcname, encoded.tobytes())
                manifests[owner].append([
                    "video_frame", arcname, video.name, str(decoded), str(global_frame_index)
                ])
                written_video[owner] += 1
                decoded += 1
                global_frame_index += 1
                if global_frame_index % 250 == 0:
                    print(
                        f"video {global_frame_index}/{total_video_frames} "
                        f"split={written_video}", flush=True
                    )
            cap.release()
            if decoded != expected_count:
                raise RuntimeError(
                    f"decoded frame mismatch for {video.name}: {decoded} != {expected_count}"
                )

        for index, image in enumerate(images):
            owner = owner_for_index(index, image_sizes)
            arcname = f"captured_images/{image.name}"
            archives[owner].write(image, arcname)
            manifests[owner].append([
                "captured_image", arcname, image.name, "", str(index)
            ])
            written_images[owner] += 1
            if (index + 1) % 500 == 0:
                print(
                    f"images {index + 1}/{len(images)} split={written_images}", flush=True
                )

        for owner, archive in enumerate(archives):
            manifest_buffer = io.StringIO(newline="")
            writer = csv.writer(manifest_buffer)
            writer.writerow([
                "source_type", "archive_path", "source_name", "source_frame_index",
                "source_global_index",
            ])
            writer.writerows(manifests[owner])
            archive.writestr("MANIFEST.csv", "\ufeff" + manifest_buffer.getvalue())
            readme = (
                f"담당자: {PEOPLE[owner]}\n"
                f"촬영일: 2026-09-04\n"
                f"원본 영상 추출 프레임: {written_video[owner]}장\n"
                f"real_unlabeled 직접 촬영 이미지: {written_images[owner]}장\n"
                f"전체 이미지: {written_video[owner] + written_images[owner]}장\n"
                "영상 프레임은 추가 손실을 막기 위해 PNG로 저장했습니다.\n"
            )
            archive.writestr("README.txt", readme)
    finally:
        for archive in archives:
            archive.close()

    if written_video != video_sizes or written_images != image_sizes:
        raise RuntimeError(
            f"split mismatch: video={written_video}/{video_sizes}, "
            f"images={written_images}/{image_sizes}"
        )

    for path, expected in zip(zip_paths, total_sizes):
        with ZipFile(path, "r") as archive:
            media_entries = [
                item for item in archive.infolist()
                if item.filename.startswith(("video_frames/", "captured_images/"))
            ]
            if len(media_entries) != expected:
                raise RuntimeError(
                    f"archive count mismatch for {path.name}: {len(media_entries)} != {expected}"
                )
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"CRC failure in {path.name}: {bad}")
        print(f"verified {path.name}: {expected} images, {path.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
