"""Render original rec videos with two or three models, preserving raw keypoint IDs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np
import torch

# A custom training module must be supplied locally, not installed by guessing a PyPI name.
os.environ["YOLO_AUTOINSTALL"] = "false"
from ultralytics import YOLO

from render_video_comparison import ROOT, annotate


def probe(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        return {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def select_points(result, selection="aspect"):
    if not len(result.boxes):
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    ratios = (boxes[:, 2] - boxes[:, 0]) / np.maximum(boxes[:, 3] - boxes[:, 1], 1.)
    best = (int(conf.argmax()) if selection == "max_conf" else
            min(range(len(boxes)), key=lambda k: (abs(ratios[k] - 1.1 / .15), -conf[k])))
    return result.keypoints.data[best].cpu().numpy()


def compose(frame, results, index, fps, candidate_name="c4", padding=0, selection="aspect", baseline_name="v4", third_name=None):
    panels, points = [], []
    baseline_label = "v4 ORIGINAL" if baseline_name == "v4" else baseline_name.upper()
    labels = [baseline_label, candidate_name.upper()]
    if third_name is not None:
        labels.append(third_name.upper())
    if len(results) != len(labels):
        raise ValueError("Each model must have one result and one label")
    for name, result in zip(labels, results):
        pts = select_points(result, selection)
        if pts is not None:
            pts = pts.copy()
            pts[:, :2] -= padding
            points.append(pts[:, :2])
        panels.append(annotate(frame, pts, f"{name} | {index / fps:.1f}s | frame {index}"))
    center = np.concatenate(points).mean(0) if points else np.array([320, 320])
    left = int(np.clip(center[0] - 160, 0, 320))
    top = int(np.clip(center[1] - 80, 0, 320))
    panels = [np.vstack([p, cv2.resize(p[top:top + 160, left:left + 320], (640, 320),
                                     interpolation=cv2.INTER_NEAREST)]) for p in panels]
    canvas = np.hstack(panels)
    cv2.putText(canvas, "RAW 0-8 | No PnP / no smoothing | Common 2x crop below", (10, 474),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def verify(path, expected, width=1280):
    actual = probe(path)
    if actual["frames"] != expected["frames"] or abs(actual["fps"] - expected["fps"]) > .01:
        raise RuntimeError(f"Frame count/FPS mismatch: {path}: {actual} vs {expected}")
    if (actual["width"], actual["height"]) != (width, 800):
        raise RuntimeError(f"Output resolution mismatch: {path}")
    cap = cv2.VideoCapture(str(path))
    indices = sorted({0, expected["frames"] // 2, expected["frames"] - 1})
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame.shape[:2] != (800, width):
                raise RuntimeError(f"Cannot decode {path} frame {index}")
    finally:
        cap.release()
    return {"metadata": actual, "decoded_indices": indices}


def save_status(out, report):
    temporary = out / "status.tmp.json"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    temporary.replace(out / "status.json")
    candidate_label = report.get("candidate_name", "c4").upper()
    baseline_label = report.get("baseline_name", "v4").upper()
    labels = [baseline_label, candidate_label]
    if report.get("third_name"):
        labels.append(report["third_name"].upper())
    settings = report["settings"]
    selection_text = ("최고 신뢰도 박스" if settings.get("selection") == "max_conf" else "기존 실행 코드의 박스 비율 우선 규칙")
    rows = [f"# rec 원본: {' / '.join(labels)} 키포인트 비교", "",
            f"왼쪽부터: {' / '.join(labels)}. 아래: 모든 모델에 동일한 영역의 2배 확대.",
            "모델이 출력한 0~8번 좌표를 그대로 표시합니다. PnP, 전면부 재지정, 시간 평활화는 적용하지 않습니다.",
            f"여러 검출이 있으면 {selection_text}으로 하나를 표시합니다.",
            f"모든 모델 동일 조건: imgsz={settings['imgsz']}, conf={settings['conf']}, "
            f"반사 패딩={settings.get('padding', 0)}px. 표시 좌표에서는 패딩을 뺍니다.",
            "원본 FPS, 전체 프레임을 유지합니다. 아래 링크는 저장 및 검증이 끝난 영상입니다.", "",
            f"완료: {len(report['completed'])}/{len(report['inputs'])}개", "",
            "| 녹화 | 프레임 | 길이 | 비교 영상 |", "|---|---:|---:|---|"]
    for item in report["completed"]:
        seconds = item["frames"] / item["fps"]
        rows.append(f"| {item['group']} | {item['frames']} | {int(seconds)//60}:{int(seconds)%60:02d} | "
                    f"[재생]({item['output']}) |")
    rows += ["", "검출 프레임 수는 정답 기반 정확도가 아닙니다. 번호가 다르더라도 대칭적으로 같은 박스일 수 있습니다.",
             "모델 경로와 SHA256, 처리 진행률, 프레임 수 및 디코딩 검증은 status.json에 기록됩니다."]
    (out / "INDEX.md").write_text("\n".join(rows) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec", type=Path, default=ROOT / "extracted/depth_cam/rec")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "extracted/pallet_yolo26n_pose_livegt_v4.pt")
    parser.add_argument("--baseline-name", default="v4", help="Left model label in titles, filenames and reports")
    parser.add_argument("--candidate", type=Path,
                        default=ROOT / "training/symmetry/models/pallet_yolo26n_pose_v4_c4_ft_selected.pt")
    parser.add_argument("--candidate-name", default="c4", help="Candidate label used in video titles, filenames and reports")
    parser.add_argument("--third-model", type=Path, help="Optional third model, displayed on the right")
    parser.add_argument("--third-name", help="Required label when third-model is supplied")
    parser.add_argument("--padding", type=int, default=0, help="BORDER_REFLECT_101 pixels on each input edge, for all models")
    parser.add_argument("--conf", type=float, default=.3)
    parser.add_argument("--selection", choices=("aspect", "max_conf"), default="aspect")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--only", nargs="+", help="Optional exact raw filename stems to process")
    args = parser.parse_args()
    if args.batch < 1 or args.threads < 1:
        parser.error("batch and threads must be positive")
    if args.padding < 0 or not 0 < args.conf < 1:
        parser.error("padding must be nonnegative and conf must be between zero and one")
    if (args.third_model is None) != (args.third_name is None):
        parser.error("third-model and third-name must be supplied together")
    names = [args.baseline_name, args.candidate_name]
    if args.third_name is not None:
        names.append(args.third_name)
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names):
        parser.error("model names must be lowercase identifiers")
    if len(set(names)) != len(names):
        parser.error("model names must differ")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    videos = sorted(args.rec.rglob("*_raw.mp4"))
    if args.only:
        missing = set(args.only) - {p.stem for p in videos}
        if missing:
            raise FileNotFoundError(f"Requested recordings not found: {sorted(missing)}")
        videos = [p for p in videos if p.stem in args.only]
    if not videos:
        raise RuntimeError("No raw source videos found")
    if len({p.stem for p in videos}) != len(videos):
        raise ValueError("Duplicate source stems; output filenames would collide")
    metadata = [dict(source=str(p.resolve()), **probe(p)) for p in videos]
    for item in metadata:
        if (item["width"], item["height"]) != (640, 480) or item["fps"] <= 0 or item["frames"] <= 0:
            raise ValueError(f"Unsupported source: {item}")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    paths = {args.baseline_name: args.baseline, args.candidate_name: args.candidate}
    if args.third_model is not None:
        paths[args.third_name] = args.third_model
    output_width = 640 * len(paths)
    # Validate all models before creating a run directory or claiming inference is running.
    models = [YOLO(str(p)) for p in paths.values()]
    for name, model in zip(paths, models):
        shape = getattr(model.model.model[-1], "kpt_shape", None)
        if model.task != "pose" or shape is None or list(shape) != [9, 3]:
            raise ValueError(f"{name}: this visualizer requires a pose model with kpt_shape=[9, 3], got {shape}")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"state": "running", "inputs": metadata, "completed": [], "candidate_name": args.candidate_name,
              "baseline_name": args.baseline_name, "third_name": args.third_name,
              "model_order": list(paths),
              "total_frames": sum(m["frames"] for m in metadata), "processed_frames": 0,
              "settings": {"conf": args.conf, "imgsz": 640, "device": "cpu", "batch": args.batch,
                           "padding": args.padding, "padding_mode": "BORDER_REFLECT_101",
                           "selection": args.selection,
                           "raw_keypoints": True, "pnp": False, "smoothing": False},
              "models": {name: {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for name, p in paths.items()}}
    save_status(args.out, report)
    started = time.perf_counter()
    try:
        for video, meta in zip(videos, metadata):
            group = video.stem.removeprefix("forklift_v4_recording_").removesuffix("_raw")
            output = args.out / f"{group}_{'_vs_'.join(paths)}_raw_keypoints.mp4"
            partial = output.with_name(output.stem + ".partial.mp4")
            cap = cv2.VideoCapture(str(video))
            writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (output_width, 800))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot create {partial}")
            processed = 0
            detected = {name: 0 for name in paths}
            last_update = time.perf_counter()
            print(f"START {group}: {meta['frames']} frames", flush=True)
            # Offload rendering/encoding while the next batch performs CPU inference.
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = None
                try:
                    while True:
                        frames = []
                        for _ in range(args.batch):
                            ok, frame = cap.read()
                            if not ok:
                                break
                            frames.append(frame)
                        if not frames:
                            break
                        inference_frames = [cv2.copyMakeBorder(f, args.padding, args.padding,
                                            args.padding, args.padding, cv2.BORDER_REFLECT_101)
                                            for f in frames] if args.padding else frames
                        results = []
                        for model in models:
                            if model.predictor is None:
                                # Ultralytics CPU setup resets the global PyTorch thread count.
                                model.predict(inference_frames[:1], conf=args.conf, imgsz=640, device="cpu", verbose=False)
                                torch.set_num_threads(args.threads)
                            results.append(model.predict(inference_frames, conf=args.conf, imgsz=640, device="cpu", verbose=False))
                        report["settings"]["effective_torch_threads"] = torch.get_num_threads()
                        for name, result_batch in zip(detected, results):
                            detected[name] += sum(bool(len(r.boxes)) for r in result_batch)
                        if pending is not None:
                            pending.result()

                        def encode(batch_frames, batch_results, offset):
                            for j, source_frame in enumerate(batch_frames):
                                writer.write(compose(source_frame, [r[j] for r in batch_results],
                                                     offset + j, meta["fps"], args.candidate_name,
                                                     args.padding, args.selection, args.baseline_name, args.third_name))

                        pending = pool.submit(encode, frames, results, processed)
                        processed += len(frames)
                        report["processed_frames"] += len(frames)
                        if time.perf_counter() - last_update >= 15:
                            report["current"] = {"group": group, "frames": processed, "total": meta["frames"]}
                            report["elapsed_seconds"] = time.perf_counter() - started
                            save_status(args.out, report)
                            print(f"PROGRESS {group} {processed}/{meta['frames']} | "
                                  f"all {report['processed_frames']}/{report['total_frames']} | "
                                  f"{report['elapsed_seconds']:.0f}s", flush=True)
                            last_update = time.perf_counter()
                    if pending is not None:
                        pending.result()
                finally:
                    cap.release()
                    writer.release()
            if processed != meta["frames"]:
                raise RuntimeError(f"Incomplete source decode: {video}: {processed}/{meta['frames']}")
            verification = verify(partial, meta, output_width)
            partial.rename(output)
            report["completed"].append({"group": group, "source": str(video.resolve()),
                                        "output": output.name, "frames": processed, "fps": meta["fps"],
                                        "detected_frames": detected, "verification": verification})
            save_status(args.out, report)
            print(f"DONE {group} | {len(report['completed'])}/{len(videos)} videos", flush=True)
        report["state"] = "complete"
        report.pop("current", None)
        report["elapsed_seconds"] = time.perf_counter() - started
        save_status(args.out, report)
        print(f"COMPLETE {len(videos)} videos, {report['processed_frames']} frames", flush=True)
    except Exception as exc:
        report["state"] = "failed"
        report["error"] = repr(exc)
        save_status(args.out, report)
        raise


if __name__ == "__main__":
    main()
