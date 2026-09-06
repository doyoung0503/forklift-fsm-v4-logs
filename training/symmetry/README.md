# Pallet C4 symmetry fine-tuning

Run from the workspace root with Python 3.12, torch 2.7.1+cu118 and ultralytics 8.4.138.
No site-packages files are modified. Training code depends on that specific Ultralytics loss API.

```powershell
python training/symmetry/prepare_data.py
python training/symmetry/test_symmetry.py
python training/symmetry/train.py --name v4_c4_manualgt_20260905 --device cpu --workers 0
python training/symmetry/train.py --mode fixed --name v4_fixed_manualgt_20260905 --device cpu --workers 0
python training/symmetry/finalize.py
```

The prepared dataset and run names must not already exist; original ZIPs and source weights are preserved.
For subsequent experiments use a new `--name`. `--device 0` enables CUDA when normal GPU clocks are available.
The current run uses CPU because GPU firmware/driver reported a 10W cap and 210MHz clock; it was slower than CPU.

If both training processes are already running, `python training/symmetry/finalize.py --watch` waits for exports,
selects checkpoints on validation only, then evaluates the untouched test split and writes `RESULTS.md`.
`pipeline_status.json`, `c4_train.log`, `fixed_train.log` and the per-run `results.csv` record progress.

Loss details: C4 means four 90-degree Y-axis rotations of the square cuboid. Center point 8 remains fixed.
One whole nine-point permutation is selected per positive prediction using pose*12 + presence + RLE.
Manual coordinate/RLE weight is 1.0; generated point weight is 0.25. Presence means labelled/in-frame (v>0),
not certainty that the physical corner is visible. Number permutations move coordinates AND source masks together.
The fixed-index control uses precisely the same code/weights with only the identity permutation allowed.
Both use C4-aware validation OKS/mAP. During training best.pt is chosen using that mAP fitness;
the finalizer additionally compares saved best/last/periodic snapshots by validation recall then manual pixel error.
C4 matching handles numbering ambiguity. It does not constrain every predicted point set to be an exact
projective cuboid, and no new geometric or temporal loss is introduced in this experiment.

Exports in `models/` are ordinary `ultralytics.nn.tasks.PoseModel` checkpoints and require no custom loader:

```python
from ultralytics import YOLO
model = YOLO("training/symmetry/models/pallet_yolo26n_pose_v4_c4_ft_selected.pt")
result = model.predict("image.png")[0]
raw_keypoints = result.keypoints.data  # no fixed front-face identity
```

For further symmetry-aware training, use this custom trainer again. Ordinary `YOLO(...).train()` reverts to
the stock fixed-index loss even when the source checkpoint was symmetry-trained.
Existing runtime `pose_from_visible_kpts_pnp` already canonicalizes to the largest camera-facing vertical face.
Seven tests cover geometry-preserving permutations, invalid swaps, mask alignment, finite gradients,
RLE agreement with upstream, disjoint splits and runtime post-PnP invariance under C4 renumbering.

The live FSM model is not automatically replaced. Inspect held-out error and temporal behavior before selecting a new default.
The existing largest-face selector has no new tie hysteresis in this experiment; equal-area face transitions
remain a separate postprocessing concern.

## Full rec video comparison

```powershell
python training/symmetry/render_all_rec.py --out training/symmetry/evaluation/all_rec_v4_vs_c4_20260905
```

Use a new output directory for another run; existing results are never overwritten.
The script discovers every `*_raw.mp4` recursively under `extracted/depth_cam/rec`.
It runs the original v4 and the validation-selected C4 model on every frame at confidence 0.3,
then writes original-FPS, 1280x800 side-by-side videos with raw keypoint IDs and a common 2x crop.
No PnP, front-face canonicalization or smoothing is applied. Detection selection follows the live
box-aspect-ratio preference. `INDEX.md` links completed videos; `status.json` records source/output
frame counts, model hashes, progress and first/middle/last-frame decode verification.
CPU batches overlap inference with rendering; `--batch` and `--threads` both default to 8.
Partial videos use `.partial.mp4` and are renamed only after length/FPS and decode checks pass.

After rendering, optionally decode every frame of every source and comparison video:

```powershell
python training/symmetry/verify_full_video_decode.py training/symmetry/evaluation/all_rec_v4_vs_c4_20260905
```

This records per-file native decoder warnings separately from frame-count failures in
`full_decode_verification.json`. The completed 32-video run has no comparison-video decode warnings;
one original source (`20260901_175419_raw.mp4`) reports MPEG-4 damage but still yields all 577 frames.

To compare the previously fine-tuned C4 candidate against the separate C5 inference export:

```powershell
python training/symmetry/render_all_rec.py --baseline training/symmetry/models/pallet_yolo26n_pose_v4_c4_ft_selected.pt --baseline-name c4 --candidate training/symmetry/models/pallet_yolo26n_pose_c5_inference.pt --candidate-name c5 --padding 100 --conf 0.4 --selection max_conf --out training/symmetry/evaluation/shaky3_c4_vs_c5_20260906 --only forklift_v4_recording_20260903_192254_raw forklift_v4_recording_20260903_192417_raw forklift_v4_recording_20260904_190700_raw
```

`--baseline` / `--baseline-name` control the left model; `--candidate` / `--candidate-name`
control the right. Defaults still compare original v4 versus local C4. Both models receive
identical preprocessing and confidence/selection settings. The local C4 candidate is NOT
the public Hugging Face checkpoint whose filename was changed to C5.
