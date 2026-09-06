"""Describe the view coverage in the supplied annotations; not independent pose GT."""
from collections import defaultdict
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = json.loads((ROOT/"datasets/pallet_symmetry_v1/manifest.json").read_text(encoding="utf8"))
    archives = {}
    output = []
    try:
        for row in rows:
            archive = row["archive"]
            if archive not in archives:
                archives[archive] = zipfile.ZipFile(ROOT/"datasets"/archive)
            data = json.loads(archives[archive].read(row["annotation_member"]))
            obj = data["objects"][0]
            transform = np.asarray(obj["pose_transform"])
            R, t = transform[:3,:3], transform[:3,3]
            pts = np.asarray(obj["projected_cuboid"], dtype=np.float32)
            centered = np.array([[-.55,-.075,-.55],[.55,-.075,-.55],[.55,.075,-.55],[-.55,.075,-.55],
                                 [-.55,-.075,.55],[.55,-.075,.55],[.55,.075,.55],[-.55,.075,.55],[0,0,0]])
            cam = data["camera_data"]["intrinsics"]
            K = np.array([[cam["fx"],0,cam["cx"]],[0,cam["fy"],cam["cy"]],[0,0,1]])
            projected, _ = cv2.projectPoints(centered, cv2.Rodrigues(R)[0], t, K, np.zeros(5))
            generated = np.array([a["source"] != "manual_click" for a in obj["keypoint_annotations"]])
            generated_error = np.linalg.norm(projected.reshape(9,2) - np.array(obj["manual_kps"]), axis=1)[generated]
            # Distinguish centered-pose metadata from the runtime front-origin convention.
            # Allow rounding/small inconsistencies in stored generated coordinates.
            front_projected, _ = cv2.projectPoints(centered + [0,0,.55], cv2.Rodrigues(R)[0], t, K, np.zeros(5))
            front_error = np.linalg.norm(front_projected.reshape(9,2) - np.array(obj["manual_kps"]), axis=1)[generated]
            if generated_error.mean() >= front_error.mean():
                raise ValueError(f"Ambiguous pose origin: {row['stem']}")
            visible = []
            for indices, normal, center in [([0,1,2,3],[0,0,-1],[0,0,-.55]),
                                            ([4,5,6,7],[0,0,1],[0,0,.55]),
                                            ([0,3,7,4],[-1,0,0],[-.55,0,0]),
                                            ([1,5,6,2],[1,0,0],[.55,0,0])]:
                center_cam = R @ center + t
                if np.dot(R @ normal, -center_cam) > 0:
                    visible.append(abs(cv2.contourArea(pts[indices])))
            visible.sort(reverse=True)
            ratio = visible[0]/visible[1] if len(visible)>1 and visible[1]>0 else None
            output.append({"stem":row["stem"],"split":row["split"],"visible_face_areas":visible,
                           "generated_reprojection_max_px":float(generated_error.max()),
                           "dominance_ratio":ratio,"annotation_yaw_deg":float(np.degrees(np.arctan2(R[0,2],R[2,2])))})
    finally:
        for z in archives.values():
            z.close()
    grouped = defaultdict(list)
    for r in output:
        grouped[r["split"]].append(r)
    summary = {}
    for split, rs in grouped.items():
        ratios = [r["dominance_ratio"] for r in rs if r["dominance_ratio"] is not None]
        summary[split] = {"images":len(rs),"two_visible_vertical_faces":len(ratios),
                          "max_generated_reprojection_residual_px":max(r["generated_reprojection_max_px"] for r in rs),
                          "near_tie_ratio_lt_1_1":sum(r<1.1 for r in ratios),
                          "near_tie_ratio_lt_1_25":sum(r<1.25 for r in ratios),
                          "ratio_percentiles":np.percentile(ratios,[0,25,50,75,100]).tolist() if ratios else [],
                          "annotation_yaw_percentiles":np.percentile([r["annotation_yaw_deg"] for r in rs],[0,25,50,75,100]).tolist()}
    report = {"summary":summary,"records":output,
              "notes":"Uses label PnP R/t with centered cuboid, so this is annotation coverage, not independent true pose."}
    (ROOT/"training/symmetry/geometry_coverage.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
