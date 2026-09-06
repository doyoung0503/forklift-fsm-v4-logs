"""Screen rendered overlays for jitter; this is NOT a raw model-coordinate metric."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import cv2
import numpy as np


def open_video(path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        raise RuntimeError(path)
    return cap


def points_from_overlay(panel, raw):
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 140, 140), (40, 255, 255))
    mask |= cv2.inRange(hsv, (85, 140, 140), (110, 255, 255))
    mask |= cv2.inRange(hsv, (140, 140, 140), (170, 255, 255))
    channels = cv2.split(cv2.absdiff(panel, raw))
    difference = cv2.max(channels[0], cv2.max(channels[1], channels[2]))
    mask[difference < 65] = 0
    mask[:42] = 0
    mask[450:] = 0
    # Suppress thin strokes. Some thicker glyph fragments remain: this is only a screening metric.
    mask = cv2.erode(mask, np.ones((2, 2), np.uint8))
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    selected = (stats[1:, cv2.CC_STAT_AREA] >= 6) & (stats[1:, cv2.CC_STAT_AREA] <= 70)
    points = centers[1:][selected].astype(np.float32)
    return points if 5 <= len(points) <= 24 else None


def local_motion(previous_gray, gray, points):
    mask = np.zeros_like(gray)
    low = np.maximum(points.min(0) / 2 - 20, [0, 0]).astype(int)
    high = np.minimum(points.max(0) / 2 + 20, [319, 239]).astype(int)
    cv2.rectangle(mask, tuple(low), tuple(high), 255, -1)
    features = cv2.goodFeaturesToTrack(previous_gray, 60, .02, 4, mask=mask)
    if features is None or len(features) < 6:
        return None
    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray, gray, features, None, winSize=(15, 15), maxLevel=2)
    if tracked is None:
        return None
    good = (status[:, 0] == 1) & (error[:, 0] < 30)
    a, b = features[good, 0], tracked[good, 0]
    if len(a) < 6:
        return None
    transform, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                   ransacReprojThreshold=2, maxIters=300)
    if transform is None or inliers.sum() < 6:
        return None
    scale = np.sqrt(np.linalg.det(transform[:, :2]))
    if not .8 < scale < 1.2:
        return None
    transform[:, 2] *= 2
    return transform


def compare_points(previous, current, transform):
    projected = previous @ transform[:, :2].T + transform[:, 2]
    distances = np.linalg.norm(projected[:, None] - current[None], axis=2)
    # Ignore color and IDs: a symmetry-equivalent renumbering alone should not be a large jump.
    return float((distances.min(0).mean() + distances.min(1).mean()) / 2)


def analyze(item, directory, out):
    raw_cap = open_video(item["source"])
    compare_cap = open_video(directory / item["output"])
    previous_gray = None
    previous_points = [None, None]
    scores = np.full((item["frames"], 2), np.nan, dtype=np.float32)
    counts = np.zeros((item["frames"], 2), dtype=np.int16)
    recovered = np.full((item["frames"], 2, 24, 2), np.nan, dtype=np.float32)
    for index in range(item["frames"]):
        raw_ok, raw = raw_cap.read()
        video_ok, video = compare_cap.read()
        if not raw_ok or not video_ok:
            raise RuntimeError(f"Decode stopped: {item['group']} frame {index}")
        gray = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), (320, 240))
        current_points = [points_from_overlay(video[:480, k * 640:(k + 1) * 640], raw) for k in range(2)]
        for k, points in enumerate(current_points):
            if points is not None:
                counts[index, k] = len(points)
                recovered[index, k, :len(points)] = points
                if previous_points[k] is not None:
                    transform = local_motion(previous_gray, gray, previous_points[k])
                    if transform is not None:
                        scores[index, k] = compare_points(previous_points[k], points, transform)
        previous_gray, previous_points = gray, current_points
    raw_cap.release()
    compare_cap.release()
    np.savez_compressed(out / f"{item['group']}_screening.npz", scores=scores, counts=counts,
                        recovered_overlay_centers=recovered)
    result = {"group": item["group"], "video": item["output"], "frames": item["frames"],
              "fps": item["fps"], "models": {}}
    for k, name in enumerate(("v4", "c4")):
        valid = np.isfinite(scores[:, k])
        values = scores[valid, k]
        if len(values) < 30:
            result["models"][name] = {"valid_pairs": len(values), "p95_overlay_residual_px": None}
            continue
        peaks = []
        for index in np.argsort(np.nan_to_num(scores[:, k], nan=-1))[::-1]:
            if all(abs(index - p["frame"]) > item["fps"] * 3 for p in peaks):
                peaks.append({"frame": int(index), "seconds": float(index / item["fps"]),
                              "residual_px": float(scores[index, k])})
            if len(peaks) == 5:
                break
        result["models"][name] = {"valid_pairs": len(values), "coverage": float(valid.mean()),
                                   "p95_overlay_residual_px": float(np.percentile(values, 95)),
                                   "mean_overlay_residual_px": float(values.mean()),
                                   "over_5px_fraction": float((values > 5).mean()), "peaks": peaks}
    result["ranking_score"] = max((m["p95_overlay_residual_px"] or 0) for m in result["models"].values())
    print(f"DONE {item['group']}: score={result['ranking_score']:.2f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    items = json.loads((args.directory / "status.json").read_text(encoding="utf8"))["completed"]
    if args.only:
        items = [item for item in items if item["group"] in args.only]
    out = args.directory / "jitter_screening"
    out.mkdir(exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(analyze, item, args.directory, out) for item in items]):
            results.append(future.result())
    results.sort(key=lambda r: r["ranking_score"], reverse=True)
    report = {"method": "Approximate rendered-marker centroid set motion after local image-flow affine compensation. "
                        "Not raw model keypoints, not exact C4 permutation matching, not ground-truth accuracy. "
                        "IDs/colors ignored; circle extraction, compression and real perspective motion can affect scores. "
                        "Rank by worse model's 95th percentile residual. Visually inspect candidates before selecting.",
              "ranked": results}
    name = "ranking.json" if not args.only else "probe_ranking.json"
    (out / name).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(results[:8], indent=2))


if __name__ == "__main__":
    main()
