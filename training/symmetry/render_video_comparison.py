"""Continuous raw-keypoint video comparison; numbering remains model output numbering."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from symmetry_pose import symmetry_permutations

ROOT = Path(__file__).resolve().parents[2]
PERMS = symmetry_permutations()
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def annotate(frame, pts, title):
    canvas = frame.copy()
    cv2.rectangle(canvas, (0,0), (640,38), (20,20,20), -1)
    cv2.putText(canvas, title, (10,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
    if pts is None:
        cv2.putText(canvas, "No detection", (10,60), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 2)
        return canvas
    xy = np.rint(pts[:,:2]).astype(int)
    for a,b in EDGES:
        cv2.line(canvas, tuple(xy[a]), tuple(xy[b]), (170,170,170), 1, cv2.LINE_AA)
    for i,pt in enumerate(xy):
        color = (0,230,255) if i<4 else (255,190,0) if i<8 else (255,0,255)
        cv2.circle(canvas, tuple(pt), 3, color, -1)
        cv2.putText(canvas, str(i), (pt[0]+4,pt[1]-4), cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--only", nargs="+", help="Recording suffixes; default is the four regression clips")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing video outputs: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    models = {"v4": YOLO(str(ROOT/"extracted/pallet_yolo26n_pose_livegt_v4.pt")), "c4": YOLO(str(args.candidate))}
    groups = ["20260904_190700", "20260903_191906", "20260903_192254", "20260904_150944"]
    if args.only:
        if not set(args.only) <= set(groups):
            raise ValueError("Unknown regression recording")
        groups = args.only
    report = {}
    for group in groups:
        video = ROOT / "extracted/depth_cam/rec" / f"forklift_v4_recording_{group}_raw.mp4"
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        output = args.out / f"{group}_v4_vs_c4_raw_keypoints.mp4"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280,800))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output}")
        stats = {name: {"detected":0,"steps":[],"numbering_switches":0} for name in models}
        previous = {name:None for name in models}
        processed = 0
        try:
            limit = args.frames if args.frames > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for index in range(limit):
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (480,640):
                    raise ValueError(f"Unexpected video resolution: {frame.shape}")
                panels = []
                current_points = []
                for name, model in models.items():
                    result = model.predict(frame, conf=.3, imgsz=640, device="cpu", verbose=False)[0]
                    pts = None
                    if len(result.boxes):
                        # Same geometric bbox preference as live perception, followed by confidence.
                        boxes = result.boxes.xyxy.cpu().numpy()
                        conf = result.boxes.conf.cpu().numpy()
                        ratios = (boxes[:,2]-boxes[:,0])/np.maximum(boxes[:,3]-boxes[:,1],1.)
                        best = min(range(len(boxes)), key=lambda k:(abs(ratios[k] - 1.1/.15), -conf[k]))
                        pts = result.keypoints.data[best].cpu().numpy()
                        stats[name]["detected"] += 1
                        if previous[name] is not None:
                            distances = np.linalg.norm(pts[None,:,:2] - previous[name][PERMS,:2], axis=-1).mean(-1)
                            chosen = int(distances.argmin())
                            stats[name]["steps"].append(float(distances[chosen]))
                            stats[name]["numbering_switches"] += int(chosen != 0)
                    previous[name] = pts
                    if pts is not None:
                        current_points.append(pts[:,:2])
                    panels.append(annotate(frame, pts, f"{name} | frame {index} | RAW KEYPOINTS"))
                # One common crop for both models; full frame above preserves motion context.
                center = np.concatenate(current_points).mean(0) if current_points else np.array([320,320])
                left = int(np.clip(center[0]-160, 0, 320))
                top = int(np.clip(center[1]-80, 0, 320))
                panels = [np.vstack([p, cv2.resize(p[top:top+160,left:left+320], (640,320),
                                                  interpolation=cv2.INTER_NEAREST)]) for p in panels]
                writer.write(np.hstack(panels))
                processed += 1
                if processed % 100 == 0:
                    print(f"VIDEO {group}: {processed} frames", flush=True)
        finally:
            cap.release()
            writer.release()
        for item in stats.values():
            steps = item.pop("steps")
            item["matched_adjacent_pairs"] = len(steps)
            item["c4_aligned_mean_step_px"] = float(np.mean(steps)) if steps else None
            item["c4_aligned_p95_step_px"] = float(np.percentile(steps,95)) if steps else None
            item["c4_aligned_max_step_px"] = float(np.max(steps)) if steps else None
        report[group] = {"frames":processed,"fps":fps,"output":str(output),"models":stats}
    report["notes"] = ["Raw prediction positions/numbers, no PnP fitting or temporal smoothing.",
                       f"{'All' if args.frames <= 0 else 'Up to '+str(args.frames)} contiguous frames per recording, original playback FPS.",
                       "Inter-frame motion includes real camera/object movement; these are not accuracy metrics.",
                       "C4 alignment permits whole-cuboid rotation only and is used for step statistics, not drawn points."]
    (args.out/"video_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
