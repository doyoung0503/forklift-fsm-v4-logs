"""Offline comparison: unmodified FSM PnP versus all-nine EPNP + LM.

Inference runs once per frame; both solvers receive exactly the same keypoints.
No camera, controller or FSM is started and no runtime configuration is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from render_all_rec import ROOT, YOLO, probe, select_points, verify
from render_video_comparison import EDGES

sys.path.insert(0, str(ROOT / 'extracted/depth_cam'))
from calib import config, geometry


def original_frame_pose(rvec, tvec, face):
    """Undo runtime face canonicalisation for labelled-corner reprojection."""
    w, d = config.PALLET_FACE_W, config.PALLET_BODY_D
    frames = {
        'z_min': (np.eye(3), [0, 0, 0]),
        'z_max': (np.diag([-1, 1, -1]), [0, 0, d]),
        'x_min': (np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), [-w/2, 0, d/2]),
        'x_max': (np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), [w/2, 0, d/2]),
    }
    basis, center = frames.get(face, (np.eye(3), [0, 0, 0]))
    rotation = cv2.Rodrigues(rvec)[0] @ basis.T
    translation = np.asarray(tvec).reshape(3) - rotation @ np.asarray(center)
    return cv2.Rodrigues(rotation)[0], translation.reshape(3, 1)


def solve_all(points, intrin, K, dist):
    info = dict(indices=np.arange(9), inliers=np.ones(9, dtype=bool),
                n_used=9, src='all9_epnp_lm')
    fail = (False, None, None, None, None, None, None,
            dict(info, n_used=0, inliers=np.zeros(9, dtype=bool)))
    if points is None or points.shape[0] != 9 or not np.isfinite(points[:, :2]).all():
        return fail
    obj = geometry.pallet_keypoints_3d()
    img = np.ascontiguousarray(points[:, :2], dtype=np.float64)
    try:
        ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return fail
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv, tv)
        if not np.isfinite(rv).all() or not np.isfinite(tv).all():
            return fail
        rotation, center, face_info = geometry.largest_projected_vertical_face(
            cv2.Rodrigues(rv)[0], tv, K, dist)
        normal = rotation[:, 2]
        yaw = float(np.degrees(np.arctan2(normal[0], normal[2])))
        pitch = float(np.degrees(np.arctan2(-normal[1], normal[2])))
        roll = float(np.degrees(np.arctan2(-rotation[1, 0], rotation[0, 0])))
        if not np.isfinite([yaw, pitch, roll, *center]).all() or center[2] <= 0:
            return fail
        return (True, yaw, pitch, roll, center, cv2.Rodrigues(rotation)[0],
                center.reshape(3, 1), dict(info, **face_info))
    except cv2.error:
        return fail


def describe(result, points, K, dist):
    ok, yaw, pitch, roll, center, rv, tv, info = result
    data = dict(ok=bool(ok), candidates=info.get('indices'), used=[],
                source=info.get('src'))
    if not ok:
        return data
    used = np.flatnonzero(info['inliers'])
    original_rv, original_tv = original_frame_pose(rv, tv, info['selected_front_face'])
    projected = cv2.projectPoints(geometry.pallet_keypoints_3d(), original_rv,
                                 original_tv, K, dist)[0].reshape(9, 2)
    error2 = np.sum((projected - points[:, :2]) ** 2, axis=1)
    data.update(used=used, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                center_m=center, front=info['selected_front_face'],
                face_polygon=info['selected_face_corners_px'], projected=projected,
                rms_all9_px=float(np.sqrt(error2.mean())),
                rms_used_px=float(np.sqrt(error2[used].mean())),
                face_areas_px2=info['projected_face_areas_px2'])
    return data


def text_line(frame, message, y, color=(240, 240, 240), size=.48):
    cv2.putText(frame, message, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 1, cv2.LINE_AA)


def panel(frame, points, data, title, index, fps):
    canvas = frame.copy()
    if data['ok']:
        xy = np.clip(np.rint(data['projected']), -10000, 10000).astype(int)
        for a, b in EDGES:
            cv2.line(canvas, tuple(xy[a]), tuple(xy[b]), (255, 220, 0), 2, cv2.LINE_AA)
        if data['face_polygon'] is not None:
            polygon = np.clip(np.rint(data['face_polygon']), -10000, 10000).astype(np.int32)
            cv2.polylines(canvas, [polygon], True, (0, 255, 255), 2, cv2.LINE_AA)
    if points is not None:
        used = set(data['used'])
        for i, pt in enumerate(points):
            if not np.isfinite(pt[:2]).all():
                continue
            xy = tuple(np.clip(np.rint(pt[:2]), -10000, 10000).astype(int))
            color = (80, 255, 80) if i in used else (80, 80, 255)
            cv2.drawMarker(canvas, xy, color, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
            cv2.putText(canvas, str(i), (xy[0]+4, xy[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
    # Separate pose data from the scene; do not connect raw model points.
    cv2.rectangle(canvas, (0, 0), (639, 145), (16, 16, 16), -1)
    text_line(canvas, f'{title} | {index/fps:.1f}s | frame {index}', 21, size=.52)
    candidate_ids = data['candidates'] if data['candidates'] is not None else []
    text_line(canvas, 'Input IDs: ' + ','.join(map(str, candidate_ids)), 43)
    text_line(canvas, 'Used IDs: ' + ','.join(map(str, data['used'])), 65, (80, 255, 80))
    if data['ok']:
        text_line(canvas, f"Front={data['front']}  yaw={data['yaw_deg']:+.2f} deg  "
                  f"X={data['center_m'][0]:+.3f}  Z={data['center_m'][2]:.3f} m", 87)
        text_line(canvas, f"RMS all9={data['rms_all9_px']:.2f}px  "
                  f"used={data['rms_used_px']:.2f}px | {data['source']}", 109)
    else:
        text_line(canvas, 'PnP FAILED' if points is not None else 'NO DETECTION', 87, (80, 80, 255))
    text_line(canvas, 'Cyan: PnP box | Yellow: selected front | No smoothing', 132, size=.46)
    text_line(canvas, 'Cross+ID: model point | Green: used | Red: excluded', 450, size=.46)
    if points is not None:
        text_line(canvas, 'Scores 0..8: ' + ' '.join(f'{v:.2f}' for v in points[:, 2]), 472, size=.43)
    return canvas


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    stem = 'forklift_v4_recording_20260903_192254'
    source = ROOT / 'extracted/depth_cam/rec' / (stem + '_raw.mp4')
    meta_path = source.with_name(stem + '_meta.json')
    model_path = ROOT / 'third_party/pallet_pose_yolo26n_cleanlabel/pallet_yolo26n_pose_cleanlabel.pt'
    metadata = json.loads(meta_path.read_text(encoding='utf8'))
    intrin = SimpleNamespace(**metadata['intrinsics'])
    K = np.array([[intrin.fx, 0., intrin.ppx], [0., intrin.fy, intrin.ppy], [0., 0., 1.]])
    dist = np.asarray(intrin.coeffs, dtype=np.float64)
    meta = probe(source)
    assert (meta['width'], meta['height']) == (640, 480)
    assert (intrin.width, intrin.height) == (640, 480)
    model = YOLO(str(model_path))
    assert list(model.model.model[-1].kpt_shape) == [9, 3]
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / '20260903_192254_cleanlabel_current_vs_all9_pnp.mp4'
    partial = output.with_name(output.stem + '.partial.mp4')
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*'mp4v'), meta['fps'], (1280, 800))
    assert cap.isOpened() and writer.isOpened()
    rows, raw_points = [], []
    started = last_update = time.perf_counter()
    print('START cleanlabel current vs all9 PnP: 744 frames', flush=True)
    try:
        while True:
            frames = []
            for _ in range(8):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break
            padded = [cv2.copyMakeBorder(f, 100, 100, 100, 100, cv2.BORDER_REFLECT_101) for f in frames]
            if model.predictor is None:
                model.predict(padded[:1], conf=.4, imgsz=640, device='cpu', verbose=False)
                torch.set_num_threads(8)
            results = model.predict(padded, conf=.4, imgsz=640, device='cpu', verbose=False)
            for frame, result in zip(frames, results):
                index = len(rows)
                points = select_points(result, 'max_conf')
                if points is not None:
                    points = points.copy()
                    points[:, :2] -= 100
                raw_points.append(points if points is not None else np.full((9, 3), np.nan))
                current = describe(geometry.pose_from_visible_kpts_pnp(points, intrin), points, K, dist)
                if points is not None and current['candidates'] is None:
                    current['candidates'] = geometry.visible_pnp_correspondences(points)[2]
                all9 = describe(solve_all(points, intrin, K, dist), points, K, dist)
                panels = [panel(frame, points, current, 'CURRENT: score + RANSAC', index, meta['fps']),
                          panel(frame, points, all9, 'ALL 9: EPNP + LM', index, meta['fps'])]
                finite_xy = points[:, :2][np.isfinite(points[:, :2]).all(axis=1)] if points is not None else []
                center = np.mean(finite_xy, axis=0) if len(finite_xy) else [320, 320]
                left = int(np.clip(center[0]-160, 0, 320))
                top = int(np.clip(center[1]-80, 0, 320))
                canvas = np.hstack([np.vstack([p, cv2.resize(p[top:top+160, left:left+320],
                                   (640, 320), interpolation=cv2.INTER_NEAREST)]) for p in panels])
                writer.write(canvas)
                rows.append(dict(frame=index, time_sec=index/meta['fps'], current=current, all9=all9))
            if time.perf_counter() - last_update >= 15:
                print(f'PROGRESS {len(rows)}/{meta["frames"]}', flush=True)
                last_update = time.perf_counter()
    finally:
        writer.release()
        cap.release()
    assert len(rows) == meta['frames']
    verification = verify(partial, meta)
    partial.rename(output)
    np.savez_compressed(args.out / 'model_keypoints.npz', keypoints=np.asarray(raw_points))
    (args.out / 'frame_results.json').write_text(json.dumps(rows, default=json_default), encoding='utf8')
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {}
    for name in ('current', 'all9'):
        good = [r[name] for r in rows if r[name]['ok']]
        counts = {}
        for data in good:
            count = str(len(data['used']))
            counts[count] = counts.get(count, 0) + 1
        summary[name] = dict(success=len(good), failures=len(rows)-len(good), used_count_histogram=counts,
                             median_rms_all9_px=float(np.median([d['rms_all9_px'] for d in good])) if good else None)
    report = dict(state='complete', inputs=[dict(source=str(source), **meta)],
                  completed=[dict(source=str(source), output=output.name, **meta, verification=verification)],
                  models={'cleanlabel': dict(path=str(model_path), sha256=sha(model_path))},
                  settings=dict(conf=.4, padding=100, padding_mode='BORDER_REFLECT_101', imgsz=640,
                                device='cpu', batch=8, same_predictions=True, smoothing=False,
                                current='Unmodified runtime pose_from_visible_kpts_pnp',
                                all9='All 9 points, equal weight, EPNP + LM, no RANSAC or score filtering',
                                visibility_threshold=config.POSE_KPT_VIS_THR,
                                ransac_error_px=config.PNP_RANSAC_REPROJ_ERROR_PX,
                                pallet_size_m=[config.PALLET_FACE_W, config.PALLET_FACE_H, config.PALLET_BODY_D]),
                  camera_metadata_path=str(meta_path), intrinsics=metadata['intrinsics'],
                  geometry_sha256=sha(Path(geometry.__file__)), config_sha256=sha(Path(config.__file__)),
                  summary=summary, elapsed_seconds=time.perf_counter()-started)
    (args.out / 'status.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    (args.out / 'INDEX.md').write_text(
        '# Cleanlabel: 현재 PnP / 전체 9점 PnP\n\n'
        f'[비교 영상]({output.name})\n\n'
        '왼쪽: 실제 FSM PnP 함수 그대로(점수 0.5 필터, RANSAC, inlier LM; 전면 4점 IPPE 폴백 포함).\n'
        '오른쪽: 0~8번 전체를 동일 가중치로 EPNP + LM. 점수 필터와 RANSAC 이상치 제거 없음.\n'
        '같은 프레임의 cleanlabel 추론 결과를 양쪽에 동일하게 사용했습니다.\n\n'
        '청록색 선은 PnP 재투영 박스, 노란 선은 선택된 전면입니다. '
        '십자와 번호는 모델의 원시 키포인트이며, 초록은 최종 사용점, 빨강은 제외점입니다.\n'
        'Input IDs는 후보점, Used IDs는 최종 사용점입니다. 실패 프레임은 PnP FAILED로 표시합니다.\n'
        '아래쪽은 양쪽 동일 영역의 2배 확대입니다. RMS all9는 모든 점에 대한 재투영 오차이며 '
        'RMS used는 각 방법이 사용한 점에 대한 오차입니다. 재투영 오차는 실제 3D 정확도의 정답 평가가 아닙니다.\n\n'
        '두 방식 모두 동일한 카메라 내부 파라미터와 현재 FSM 치수(1.1 x 0.15 x 1.1 m), '
        '동일한 최대 투영 면적 전면 선택 함수를 사용합니다. 시간 평활화는 없습니다. '
        '원본 744프레임, 17.508 FPS를 유지합니다. 실제 FSM 설정은 변경하지 않았습니다.\n\n'
        '프레임별 사용점, 자세, 투영 좌표는 frame_results.json에, 공통 모델 출력은 model_keypoints.npz에 저장했습니다.\n',
        encoding='utf8')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
