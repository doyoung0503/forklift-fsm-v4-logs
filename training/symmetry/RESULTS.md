# Symmetry fine-tune experiment results

[한국어 결과 및 원인 분석](REVIEW_KO.md)

Source: v4; data: 162 train / 38 validation / 45 test images, split by recording.
Both fine-tunes use the same data, initialization, source weights and settings; only permutation allowance differs.
Checkpoint selection uses validation recall then manual-point pixel error. Test was not used for selection.

| Model | Matched / test | Manual mean px | Manual p95 px | PCK 5px incl. misses | Generated mean px |
|---|---:|---:|---:|---:|---:|
| v4 | 45/45 | 1.304 | 2.494 | 100.0% | 1.712 |
| c4 | 45/45 | 3.109 | 5.269 | 92.8% | 2.818 |
| fixed | 45/45 | 1.557 | 3.839 | 97.8% | 1.333 |

C4 improved held-out mean manual-point error without reducing recall: False.
This is an experiment result, not a guarantee of improved live FSM behavior.

## Artifacts

- C4 model: `training\symmetry\models\pallet_yolo26n_pose_v4_c4_ft_selected.pt`
- Control model: `training\symmetry\models\pallet_yolo26n_pose_v4_fixed_ft_selected.pt`
- Per-recording metrics: `training/symmetry/evaluation/final_test/summary.json`
- Raw keypoint comparison: `training/symmetry/evaluation/final_test/heldout_keypoint_comparison.mp4`
- Continuous comparisons: `training/symmetry/evaluation/continuous_videos/`
- Validation selection audit: `training/symmetry/selection.json`

## Interpretation

Green rings are manually clicked GT; orange rings are generated GT; magenta points/numbers are raw predictions.
GT is aligned by one valid C4 permutation for comparison. Predicted coordinates are never fitted through PnP in this video.
The video is a 2fps slideshow of held-out labelled frames, not continuous camera playback.
Runtime chooses the largest visible vertical face after PnP; the live model path is unchanged.
Test groups are held out from this fine-tune only. Original v4 training-set overlap is unknown.
Only 45 test frames from four recordings and no background-only labels: generalization/false-positive claims are limited.
PnP-generated coordinates are not independent physical ground truth; manual-point error is the primary location metric.

Label JSON and recording metadata use different camera intrinsics; see `DATA_QUALITY.md`.
The audit does not establish which calibration is correct. Original labels were preserved.
