from __future__ import annotations

import csv
import io
import json
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / 'extracted/depth_cam/rec'
NAMES = ['김지훈', '김민재']


def main():
    videos = sorted(REC.glob('*20260906*_raw.mp4'))
    assert videos, 'No source videos'
    outputs = [ROOT / f'{name}_9월6일.zip' for name in NAMES]
    for path in outputs:
        if path.exists():
            raise FileExistsError(path)
    snapshots = {v: (v.stat().st_size, v.stat().st_mtime_ns) for v in videos}
    rows = [[], []]
    summaries = []
    total = 0
    with ExitStack() as stack:
        archives = [stack.enter_context(ZipFile(p, 'x', compression=ZIP_STORED)) for p in outputs]
        for video in videos:
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open {video}')
            expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_index = 0
            counts = [0, 0]
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index % 10 == 0:
                        ok, encoded = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                        if not ok:
                            raise RuntimeError(f'Encode failed: {video.name}:{frame_index}')
                        part = total % 2
                        filename = f'images/{video.stem.removesuffix("_raw")}_frame_{frame_index:06d}.png'
                        archives[part].writestr(filename, encoded.tobytes())
                        rows[part].append([filename, video.name, frame_index, 10])
                        counts[part] += 1
                        total += 1
                    frame_index += 1
            finally:
                cap.release()
            if frame_index != expected_frames:
                raise RuntimeError(f'Incomplete decode: {video.name}: {frame_index}/{expected_frames}')
            if snapshots[video] != (video.stat().st_size, video.stat().st_mtime_ns):
                raise RuntimeError(f'Source changed: {video}')
            assert sum(counts) == (frame_index + 9) // 10
            summaries.append({'video': video.name, 'frames': frame_index, 'images_per_part': counts})
            print(f'{video.name}: frames={frame_index}, images={sum(counts)}, total={total}', flush=True)
        for part, archive in enumerate(archives):
            buf = io.StringIO(newline='')
            writer = csv.writer(buf)
            writer.writerow(['archive_path', 'source_video', 'source_frame_index', 'frame_interval'])
            writer.writerows(rows[part])
            archive.writestr('MANIFEST.csv', ('\ufeff' + buf.getvalue()).encode('utf-8'))
            archive.writestr('README.txt', (
                f'담당자: {NAMES[part]}\n촬영일: 2026-09-06\n'
                '원본: rec 폴더의 _raw.mp4 영상 20개\n'
                '추출: 각 영상의 0번 프레임부터 10프레임 간격 (0, 10, 20, ...)\n'
                '형식: 원본 해상도 640x480 PNG\n'
                '분배: 영상명과 프레임 순서로 번갈아 배정, 두 ZIP 간 이미지 중복 없음\n'
                f'이 ZIP의 이미지 수: {len(rows[part])}\n전체 이미지 수: {total}\n'
                '각 이미지의 출처는 MANIFEST.csv 참조\n'
            ).encode('utf-8'))
            archive.writestr('SOURCE_SUMMARY.json', json.dumps(summaries, ensure_ascii=False, indent=2).encode('utf-8'))

    sets = []
    for part, path in enumerate(outputs):
        print(f'Validating part {part + 1}...', flush=True)
        with ZipFile(path) as archive:
            entries = [n for n in archive.namelist() if n.endswith('.png')]
            assert len(entries) == len(rows[part])
            assert len(set(entries)) == len(entries)
            assert archive.testzip() is None
            for name in entries:
                frame = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                assert frame is not None and frame.shape == (480, 640, 3), name
            sets.append(set(entries))
        print(f'part={part + 1} images={len(entries)} bytes={path.stat().st_size}', flush=True)
    assert not sets[0].intersection(sets[1])
    assert len(sets[0] | sets[1]) == total
    assert abs(len(sets[0]) - len(sets[1])) <= 1
    print(f'COMPLETE videos={len(videos)} images={total} counts={[len(r) for r in rows]}', flush=True)


if __name__ == '__main__':
    main()
