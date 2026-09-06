"""Select checkpoints on validation only, then evaluate held-out test once.

Can watch already-running training processes. No model is installed into the live FSM.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from ultralytics import YOLO
from evaluate import aggregate, measure
from train import export_standard

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "training/symmetry"
RUNS = {"c4": "v4_c4_manualgt_20260905", "fixed": "v4_fixed_manualgt_20260905"}


def score_checkpoint(path, rows):
    model = YOLO(str(path))
    records = [measure(result, row) for result, row in zip(
        model.predict([r["image"] for r in rows], imgsz=640, conf=.3, batch=8,
                      device="cpu", stream=True, verbose=False), rows)]
    if len(records) != len(rows):
        raise RuntimeError("Incomplete validation stream")
    return aggregate(records)


def selection_key(item):
    score = item["validation"]
    # Freeze this criterion before looking at test: detection recall, then manual pixel error.
    error = score["manual_mean_px"]
    return (score["recall_iou50"], -(error if error is not None else 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="Finish report from already evaluated/rendered artifacts")
    args = ap.parse_args()
    torch.set_num_threads(4)
    status_file = BASE / "pipeline_status.json"

    def status(state, **extra):
        message = {"state": state, "updated_local": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
        status_file.write_text(json.dumps(message, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps(message, ensure_ascii=False), flush=True)

    if args.report_only:
        selection = json.loads((BASE / "selection.json").read_text(encoding="utf8"))
        if not (BASE / "evaluation/continuous_videos/video_metrics.json").is_file():
            raise FileNotFoundError("Continuous video comparison is not complete")
        finish_report({k:Path(v["export"]) for k,v in selection.items()}, status)
        return

    needed = [BASE / "models" / (name + ".pt") for name in RUNS.values()]
    deadline = time.monotonic() + 4 * 3600
    while not all(p.exists() for p in needed):
        if not args.watch:
            raise FileNotFoundError("Training exports not complete")
        for prefix in ["c4", "fixed"]:
            error_path = BASE / f"{prefix}_train.stderr.log"
            if error_path.exists() and "Traceback (most recent call last)" in error_path.read_text(encoding="utf8", errors="replace"):
                status("training_failed", log=str(error_path))
                raise RuntimeError(f"Training failed; see {error_path}")
        if time.monotonic() > deadline:
            status("watch_timeout", note="Training processes are not terminated by this watcher")
            raise TimeoutError("Training exports not available within four hours")
        status("training", awaiting=[str(p) for p in needed if not p.exists()])
        time.sleep(30)
    status("selecting_on_validation")
    manifest = json.loads((ROOT / "datasets/pallet_symmetry_v1/manifest.json").read_text(encoding="utf8"))
    rows = [r for r in manifest if r["split"] == "val"]
    selection = {}
    outputs = {}
    for label, name in RUNS.items():
        run = BASE / "runs" / name
        candidates = sorted((run / "weights").glob("*.pt"))
        scored = []
        for path in candidates:
            print(f"VALIDATION SELECTION {label} {path.name}", flush=True)
            scored.append({"checkpoint": str(path), "validation": score_checkpoint(path, rows)})
        best = max(scored, key=selection_key)
        metadata = json.loads((run / "experiment.json").read_text(encoding="utf8"))
        metadata["checkpoint_selection"] = {"criterion": "max recall_iou50 then min manual_mean_px on val",
                                             "selected": best}
        output = BASE / "models" / f"pallet_yolo26n_pose_v4_{label}_ft_selected.pt"
        export_standard(Path(best["checkpoint"]), output, metadata)
        selection[label] = {"candidates": scored, "selected": best, "export": str(output)}
        outputs[label] = output
    (BASE / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf8")
    status("evaluating_test", models={k:str(v) for k,v in outputs.items()})
    subprocess.run([sys.executable, str(BASE / "evaluate.py"), "--models",
                    f"v4={ROOT / 'extracted/pallet_yolo26n_pose_livegt_v4.pt'}",
                    f"c4={outputs['c4']}", f"fixed={outputs['fixed']}",
                    "--split", "test", "--device", "cpu", "--out", str(BASE / "evaluation/final_test")], check=True)
    status("rendering_continuous_videos")
    subprocess.run([sys.executable, str(BASE / "render_video_comparison.py"),
                    "--candidate", str(outputs["c4"]), "--out", str(BASE / "evaluation/continuous_videos")], check=True)
    finish_report(outputs, status)


def finish_report(outputs, status):
    summary = json.loads((BASE / "evaluation/final_test/summary.json").read_text(encoding="utf8"))
    baseline = summary["v4"]["overall"]
    candidate = summary["c4"]["overall"]
    improved = (candidate["recall_iou50"] >= baseline["recall_iou50"]
                and candidate["manual_mean_px"] < baseline["manual_mean_px"])
    lines = ["# Symmetry fine-tune experiment results", "",
             "Source: v4; data: 162 train / 38 validation / 45 test images, split by recording.",
             "Both fine-tunes use the same data, initialization, source weights and settings; only permutation allowance differs.",
             "Checkpoint selection uses validation recall then manual-point pixel error. Test was not used for selection.", "",
             "| Model | Matched / test | Manual mean px | Manual p95 px | PCK 5px incl. misses | Generated mean px |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, info in summary.items():
        s = info["overall"]
        lines.append(f"| {name} | {s['matched']}/{s['images']} | {s['manual_mean_px']:.3f} | {s['manual_p95_px']:.3f} | {s['manual_pck5_including_misses']:.1%} | {s['generated_mean_px']:.3f} |")
    lines += ["", f"C4 improved held-out mean manual-point error without reducing recall: {improved}.",
              "This is an experiment result, not a guarantee of improved live FSM behavior.", "",
              "## Artifacts", "",
              f"- C4 model: `{outputs['c4'].relative_to(ROOT)}`",
              f"- Control model: `{outputs['fixed'].relative_to(ROOT)}`",
              "- Per-recording metrics: `training/symmetry/evaluation/final_test/summary.json`",
              "- Raw keypoint comparison: `training/symmetry/evaluation/final_test/heldout_keypoint_comparison.mp4`",
              "- Continuous comparisons: `training/symmetry/evaluation/continuous_videos/`",
              "- Korean review and findings: [REVIEW_KO.md](REVIEW_KO.md)",
              "- Validation selection audit: `training/symmetry/selection.json`", "",
              "## Interpretation", "",
              "Green rings are manually clicked GT; orange rings are generated GT; magenta points/numbers are raw predictions.",
              "GT is aligned by one valid C4 permutation for comparison. Predicted coordinates are never fitted through PnP in this video.",
              "The video is a 2fps slideshow of held-out labelled frames, not continuous camera playback.",
              "Runtime chooses the largest visible vertical face after PnP; the live model path is unchanged.",
              "Test groups are held out from this fine-tune only. Original v4 training-set overlap is unknown.",
              "Only 45 test frames from four recordings and no background-only labels: generalization/false-positive claims are limited.",
              "PnP-generated coordinates are not independent physical ground truth; manual-point error is the primary location metric."]
    lines += ["", "Label JSON and recording metadata use different camera intrinsics; see `DATA_QUALITY.md`.",
              "The audit does not establish which calibration is correct. Original labels were preserved."]
    (BASE / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    status("complete", improved_manual_error=improved, report=str(BASE/"RESULTS.md"), models={k:str(v) for k,v in outputs.items()})


if __name__ == "__main__":
    main()
