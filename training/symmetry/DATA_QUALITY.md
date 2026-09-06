# Dataset audit observations

Both archives contain 245 PNG/JSON pairs in total, no duplicates and no corrupt labels.
Every object has four manual clicks, four PnP-generated corners and one generated center.
All dimensions are 1.10 x 0.15 x 1.10 m and images/label coordinates are 640 x 480.
Signed physical axis is explicitly unconfirmed in the JSON; the chosen target is a cuboid modulo C4.

All 13 recordings have different intrinsics in label JSON versus the accompanying recording metadata:

| Parameter | Label JSON | Recording metadata |
|---|---:|---:|
| fx | 614.18 | 605.906494 |
| fy | 614.31 | 605.969788 |
| cx / ppx | 329.28 | 317.596191 |
| cy / ppy | 234.53 | 256.292297 |

The audit does not establish which calibration is correct or whether JSON values were deliberately recalibrated.
Manual image coordinates remain usable. Generated targets can inherit calibration-dependent errors;
they are therefore weighted 0.25 and are not treated as independent physical ground truth.
No coordinates or calibration fields in the original archives are modified.

The v4 checkpoint was exported on 2026-09-04 at 18:32 and refers to an unavailable `live_gt_v4` training dataset.
The new video-based split prevents leakage within this fine-tune, but original-model exposure to these images
cannot be excluded. Treat the final result as a local regression comparison, not a clean unseen-domain benchmark.

Baseline validation (38 frames, v4, C4-aligned GT, conf=0.3, bbox IoU>=0.5):
38/38 matched, manual-point mean error 1.618px, PCK5 including misses 98.68%.
Best geometric correspondence was identity in 16 frames and a 90-degree-equivalent permutation in 22 frames.
Raw fixed-index errors would therefore confuse numbering disagreement with actual localization error.

View coverage (`geometry_coverage.json`, derived from annotation poses and projected corners):

- Training annotation yaw range: -13.10 to +15.85 degrees; median -1.83 degrees.
- Only 7/162 training annotations have two camera-facing vertical faces; the smallest
  largest/second-largest projected area ratio is 5.09.
- No train/validation/test annotation has a visible-face area ratio below 1.25.
- This dataset therefore does not cover the nearly equal-face transition that motivated C4 training.
  C4 loss can accept equivalent numbering, but fine-tuning cannot learn or validate an unrepresented view regime.
- Coverage uses annotation PnP, not independent physical pose measurements. The centered-pose convention is checked
  against generated points before calculating the face normals.
