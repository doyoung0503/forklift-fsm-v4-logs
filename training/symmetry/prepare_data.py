"""Audit ZIP JSON pose annotations and create a recording-disjoint YOLO dataset."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
VAL = {"20260903_190743", "20260904_142318", "20260904_142958"}
TEST = {"20260903_192254", "20260904_144614", "20260904_150335", "20260904_150944"}


def split_for(group):
    suffix = group.removeprefix("forklift_v4_recording_")
    return "val" if suffix in VAL else "test" if suffix in TEST else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/pallet_symmetry_v1")
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite prepared dataset: {out}")
    records, duplicates, seen, rejected = [], [], {}, []
    examples = {}
    archives = sorted((ROOT / "datasets").glob("*_manual_gt.zip"))
    if len(archives) != 2:
        raise ValueError(f"Expected two manual GT archives, got {archives}")
    archive_hashes = {}
    for archive in archives:
        archive_hashes[archive.name] = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
        with zipfile.ZipFile(archive) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".json")):
                annotation = json.loads(z.read(name))
                stem = Path(name).stem
                group = stem.split("_frame_")[0]
                split = split_for(group)
                raw = z.read(str(Path(name).with_suffix(".png")).replace("\\", "/"))
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Cannot decode {name}")
                height, width = img.shape[:2]
                cam = annotation["camera_data"]
                if (width, height) != (cam["width"], cam["height"]):
                    raise ValueError(f"Image/label coordinate size mismatch: {name}: {width,height} vs {cam}")
                digest = hashlib.sha256(img.tobytes()).hexdigest()
                if digest in seen:
                    duplicates.append({"name": name, "same_as": seen[digest]})
                    continue
                seen[digest] = name
                objects, lines = [], []
                for obj in annotation["objects"]:
                    dims = obj["dimensions_m"]
                    if not np.allclose([dims[k] for k in ("width", "height", "depth")], [1.1, .15, 1.1]):
                        raise ValueError(f"Unexpected cuboid size: {name}: {dims}")
                    anns = obj["keypoint_annotations"]
                    if len(anns) != 9:
                        raise ValueError(f"Expected 9 points: {name}")
                    xy = np.array([k["xy"] for k in anns], dtype=float)
                    if not np.isfinite(xy).all():
                        raise ValueError(f"Nonfinite points: {name}")
                    if not np.allclose(xy, obj["manual_kps"]):
                        raise ValueError(f"Conflicting coordinates: {name}")
                    visibility = np.array([k["visibility"] for k in anns], dtype=float)
                    if not np.isin(visibility, [0, 1, 2]).all():
                        raise ValueError(f"Unexpected visibility: {name}")
                    manual = np.array([k["source"] == "manual_click" for k in anns])
                    if not np.array_equal(manual, visibility == 2):
                        raise ValueError(f"Manual/visibility encoding differs: {name}")
                    in_frame = ((xy >= 0) & (xy < [width, height])).all(1)
                    visibility[~in_frame] = 0
                    lo = np.maximum(xy[:8].min(0), [0, 0])
                    hi = np.minimum(xy[:8].max(0), [width, height])
                    if np.any(hi - lo < 2):
                        raise ValueError(f"Invalid box: {name}")
                    norm = xy / [width, height]
                    norm[visibility == 0] = 0
                    box = np.r_[(hi + lo) / 2 / [width, height], (hi - lo) / [width, height]]
                    kpts = np.column_stack([norm, visibility])
                    lines.append("0 " + " ".join(f"{v:.9f}" for v in np.r_[box, kpts.ravel()]))
                    objects.append({"xy": xy.tolist(), "visibility": visibility.tolist(),
                                    "sources": [k["source"] for k in anns], "bbox_xyxy": np.r_[lo, hi].tolist(),
                                    "reproj_error_px": obj["reproj_error_px"],
                                    "pose_status": obj.get("pose_status"),
                                    "dimensions_m": dims})
                if not objects:
                    raise ValueError(f"Unreviewed empty annotation: {name}")
                image_path = out / "images" / split / f"{stem}.png"
                label_path = out / "labels" / split / f"{stem}.txt"
                if image_path.exists():
                    raise FileExistsError(image_path)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(raw)
                label_path.write_text("\n".join(lines) + "\n", encoding="utf8")
                records.append({"stem": stem, "group": group, "split": split,
                                "archive": archive.name, "annotation_member": name,
                                "image": str(image_path), "label": str(label_path),
                                "image_sha256": digest, "width": width, "height": height,
                                "camera_data": cam, "objects": objects})
                if group not in examples:
                    canvas = img.copy()
                    for obj in objects:
                        for i, (x, y) in enumerate(obj["xy"]):
                            color = (0, 255, 0) if obj["sources"][i] == "manual_click" else (0, 170, 255)
                            pt = (int(round(x)), int(round(y)))
                            cv2.circle(canvas, pt, 3, color, -1)
                            cv2.putText(canvas, str(i), (pt[0]+4, pt[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                    cv2.putText(canvas, group[-15:] + " " + split, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                    examples[group] = canvas
    config = {"path": out.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test",
              "names": {0: "item"}, "kpt_shape": [9, 3], "flip_idx": list(range(9))}
    (out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf8")
    (out / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf8")
    groups = defaultdict(Counter)
    for r in records:
        groups[r["split"]][r["group"]] += 1
    report = {"archives_sha256": archive_hashes, "counts": dict(Counter(r["split"] for r in records)),
              "groups": {k: dict(v) for k, v in groups.items()}, "duplicates": duplicates, "rejected": rejected,
              "total": len(records), "point_sources": dict(Counter(s for r in records for o in r["objects"] for s in o["sources"]))}
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    tiles = [cv2.resize(examples[k], (480, 360)) for k in sorted(examples)]
    tiles += [np.zeros_like(tiles[0])] * ((-len(tiles)) % 3)
    sheet = np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)])
    ok, encoded = cv2.imencode(".jpg", sheet)
    if not ok:
        raise RuntimeError("Contact sheet encoding failed")
    (out / "label_audit.jpg").write_bytes(encoded.tobytes())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
