"""Held-out comparisons using whole-cuboid C4 alignment, with raw prediction overlays."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from symmetry_pose import symmetry_permutations

ROOT = Path(__file__).resolve().parents[2]
PERMS = symmetry_permutations()


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def bbox_iou(boxes, target):
    lo = np.maximum(boxes[:, :2], target[:2])
    hi = np.minimum(boxes[:, 2:], target[2:])
    intersection = np.maximum(hi-lo, 0).prod(1)
    return intersection / (np.maximum(boxes[:, 2:]-boxes[:, :2], 0).prod(1)
                           + np.maximum(target[2:]-target[:2], 0).prod() - intersection + 1e-9)


def measure(result, row):
    obj = row["objects"][0]
    gt = np.asarray(obj["xy"])
    vis = np.asarray(obj["visibility"])
    rec = {"stem": row["stem"], "group": row["group"], "split": row["split"],
           "matched": False, "detections": len(result.boxes), "manual_points": int((vis == 2).sum())}
    if len(result.boxes) == 0:
        return rec
    boxes = result.boxes.xyxy.cpu().numpy()
    overlaps = bbox_iou(boxes, np.asarray(obj["bbox_xyxy"]))
    index = int(overlaps.argmax())
    rec["bbox_iou"] = float(overlaps[index])
    if overlaps[index] < .5:
        return rec
    pred = result.keypoints.data[index].cpu().numpy()
    distances = np.linalg.norm(pred[None, :, :2] - gt[PERMS], axis=-1)
    weights = np.where(vis[PERMS] == 2, 1., .25) * (vis[PERMS] > 0)
    # One global permutation; no independent reassignment of individual corners.
    candidate_error = (distances * weights).sum(1) / weights.sum(1).clip(1e-9)
    choice = int(candidate_error.argmin())
    mask = vis[PERMS[choice]]
    manual_errors = distances[choice][mask == 2]
    generated_errors = distances[choice][mask == 1]
    rec.update(matched=True, confidence=float(result.boxes.conf[index]), permutation=choice,
               manual_errors_px=manual_errors.tolist(), generated_errors_px=generated_errors.tolist(),
               manual_mean_px=float(manual_errors.mean()), generated_mean_px=float(generated_errors.mean()),
               manual_pck5=float((manual_errors <= 5).mean()),
               pred_xyv=pred.tolist(), gt_aligned_xy=gt[PERMS[choice]].tolist(),
               gt_aligned_visibility=mask.tolist())
    return rec


def aggregate(records):
    matched = [r for r in records if r["matched"]]
    errors = [v for r in matched for v in r["manual_errors_px"]]
    generated = [v for r in matched for v in r["generated_errors_px"]]
    total_manual = sum(r["manual_points"] for r in records)
    return {"images": len(records), "matched": len(matched), "recall_iou50": len(matched)/max(1,len(records)),
            "unmatched_detections": sum(r["detections"] - int(r["matched"]) for r in records),
            "manual_mean_px": float(np.mean(errors)) if errors else None,
            "manual_median_px": float(np.median(errors)) if errors else None,
            "manual_p95_px": float(np.percentile(errors,95)) if errors else None,
            "manual_max_px": float(np.max(errors)) if errors else None,
            "generated_mean_px": float(np.mean(generated)) if generated else None,
            "manual_pck5_including_misses": sum(v<=5 for v in errors)/max(1,total_manual),
            "manual_pck10_including_misses": sum(v<=10 for v in errors)/max(1,total_manual),
            "permutation_histogram": np.bincount([r["permutation"] for r in matched], minlength=4).tolist()}


def draw_panel(image, rec, title, bbox=None):
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 44), (20, 20, 20), -1)
    metric = f"manual error {rec['manual_mean_px']:.2f}px" if rec["matched"] else "NO MATCH"
    cv2.putText(panel, f"{title} | {metric}", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
    if rec["matched"]:
        pred = np.asarray(rec["pred_xyv"])
        gt = np.asarray(rec["gt_aligned_xy"])
        for i in range(9):
            pt = tuple(np.rint(pred[i,:2]).astype(int))
            truth = tuple(np.rint(gt[i]).astype(int))
            color = (0,255,0) if rec["gt_aligned_visibility"][i] == 2 else (0,180,255)
            cv2.circle(panel, truth, 5, color, 1)
            cv2.circle(panel, pt, 2, (255,0,255), -1)
            cv2.line(panel, truth, pt, (170,170,170), 1)
            cv2.putText(panel, str(i), (pt[0]+3,pt[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,0,255), 1)
    # Shared GT-centered 2x crop makes a 1-3px keypoint error visible without refitting predictions.
    center = (np.asarray(bbox[:2]) + np.asarray(bbox[2:])) / 2 if bbox is not None else np.array([320,320])
    left = int(np.clip(center[0]-160, 0, 320))
    top = int(np.clip(center[1]-80, 0, 320))
    zoom = cv2.resize(panel[top:top+160, left:left+320], (640,320), interpolation=cv2.INTER_NEAREST)
    cv2.putText(zoom, "2x zoom | rings: GT | magenta: raw prediction", (10,22),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    return np.vstack([panel, zoom])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=checkpoint pairs")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/"datasets/pallet_symmetry_v1/manifest.json").read_text(encoding="utf8"))
    rows = [r for r in manifest if r["split"] == args.split]
    all_results, summary = {}, {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        print(f"EVALUATING {name}: {path}", flush=True)
        model = YOLO(path)
        results = []
        # Stream a list of paths in batches; prediction coordinates are original image pixels.
        for result, row in zip(model.predict([r["image"] for r in rows], imgsz=640, conf=.3,
                                             device=args.device, batch=8, stream=True, verbose=False), rows):
            results.append(measure(result, row))
        if len(results) != len(rows):
            raise RuntimeError("Incomplete prediction stream")
        all_results[name] = results
        groups = defaultdict(list)
        for r in results:
            groups[r["group"]].append(r)
        summary[name] = {"checkpoint": str(Path(path).resolve()),
                         "sha256": hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest(),
                         "overall": aggregate(results), "by_recording": {k: aggregate(v) for k,v in groups.items()}}
        del model
        torch.cuda.empty_cache()
        print(json.dumps(summary[name]["overall"]), flush=True)
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    (args.out/"predictions.json").write_text(json.dumps(all_results, indent=2), encoding="utf8")
    with (args.out/"summary.csv").open("w", newline="", encoding="utf8") as f:
        columns = ["model"] + list(next(iter(summary.values()))["overall"])
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({"model": name, **s["overall"]} for name,s in summary.items())
    # All held-out images in order, 2fps; no PnP fitting of predictions in this visualization.
    names = list(all_results)
    video_path = args.out / "heldout_keypoint_comparison.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2., (640*len(names), 800))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write video {video_path}")
    ranked = []
    try:
        for i, row in enumerate(rows):
            image = read_image(row["image"])
            panels = [draw_panel(image, all_results[name][i], name, row["objects"][0]["bbox_xyxy"]) for name in names]
            canvas = np.hstack(panels)
            cv2.putText(canvas, row["stem"], (10,790), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            writer.write(canvas)
            baseline = all_results[names[0]][i].get("manual_mean_px", 100.)
            candidate = all_results[names[1]][i].get("manual_mean_px", 100.) if len(names)>1 else baseline
            ranked.append((abs(candidate-baseline), i, canvas))
    finally:
        writer.release()
    for rank, (_, i, canvas) in enumerate(sorted(ranked, key=lambda x:x[0], reverse=True)[:8]):
        ok, encoded = cv2.imencode(".jpg", canvas)
        if ok:
            (args.out/f"comparison_{rank:02d}_{rows[i]['stem']}.jpg").write_bytes(encoded.tobytes())
    print(f"SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
