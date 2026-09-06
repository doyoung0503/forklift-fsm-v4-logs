"""Train an isolated candidate, preserving the deployed v4 checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import ultralytics
from ultralytics.nn.tasks import PoseModel
from symmetry_pose import FixedIndexTrainer, SymmetryPoseTrainer, symmetry_permutations

ROOT = Path(__file__).resolve().parents[2]


def epoch_status(trainer):
    criterion = trainer.model.criterion
    branches = {"one2many": criterion.one2many, "one2one": criterion.one2one} if hasattr(criterion, "one2many") else {"pose": criterion}
    counts = {name: loss.selection_counts.detach().cpu().tolist() for name, loss in branches.items()}
    record = {"epoch": trainer.epoch + 1, "permutation_counts": counts}
    with (trainer.save_dir / "symmetry_choices.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps(record) + "\n")
    print("SYMMETRY_CHOICES", json.dumps(record), flush=True)
    for loss in branches.values():
        loss.selection_counts.zero_()


def export_standard(source, output, metadata):
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint["model"]
    model.__class__ = PoseModel
    if hasattr(model, "criterion"):
        delattr(model, "criterion")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    checkpoint.update(model=model, ema=None, optimizer=None, scaler=None, epoch=-1,
                      symmetry_training=metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save(checkpoint, output)
    print(f"EXPORTED {output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["symmetry", "fixed"], default="symmetry")
    ap.add_argument("--name", default="v4_c4_manualgt_20260905")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--lr0", type=float, default=.0003)
    args = ap.parse_args()
    if ultralytics.__version__ != "8.4.138":
        raise RuntimeError("Custom loss validated for ultralytics==8.4.138; review source before changing version")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this run")
    torch.set_num_threads(4)
    model_path = ROOT / "extracted/pallet_yolo26n_pose_livegt_v4.pt"
    project = ROOT / "training/symmetry/runs"
    if (project / args.name).exists():
        raise FileExistsError(project / args.name)
    metadata = {"source": str(model_path), "sha256": hashlib.file_digest(model_path.open("rb"), "sha256").hexdigest(),
                "ultralytics": ultralytics.__version__, "torch": torch.__version__, "python": sys.executable,
                "mode": args.mode, "permutations": symmetry_permutations(4 if args.mode == "symmetry" else 1).tolist(),
                "generated_coordinate_weight": .25, "front_selection": "largest visible vertical face after PnP",
                "data": str(ROOT / "datasets/pallet_symmetry_v1/data.yaml")}
    trainer_cls = SymmetryPoseTrainer if args.mode == "symmetry" else FixedIndexTrainer
    trainer = trainer_cls(overrides=dict(
        model=str(model_path), data=metadata["data"], project=str(project), name=args.name,
        epochs=args.epochs, patience=12, batch=args.batch, imgsz=args.imgsz,
        device=args.device, workers=args.workers, optimizer="AdamW", lr0=args.lr0, lrf=.1,
        weight_decay=.0005, warmup_epochs=2, warmup_bias_lr=args.lr0,
        cos_lr=True, seed=42, deterministic=True,
        amp=True, cache=False, pretrained=True, resume=False, plots=True, save=True,
        save_period=10, val=True, mosaic=0., close_mosaic=0, mixup=0., cutmix=0.,
        copy_paste=0., degrees=0., translate=.03, scale=.15, shear=0., perspective=0.,
        flipud=0., fliplr=0., hsv_h=.01, hsv_s=.3, hsv_v=.25,
        pose=12., kobj=1., rle=1.,))
    # select_device can reset CPU thread count; constrain it after trainer creation.
    torch.set_num_threads(4)
    (trainer.save_dir / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    trainer.add_callback("on_train_epoch_end", epoch_status)
    trainer.train()
    source = trainer.best if trainer.best.exists() else trainer.last
    export_standard(source, ROOT / "training/symmetry/models" / (args.name + ".pt"), metadata)


if __name__ == "__main__":
    main()
