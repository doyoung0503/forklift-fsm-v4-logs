"""Inference-only compatibility export of the pinned public C4 checkpoint (locally named C5).

The author's model card explicitly states that inference is stock YOLO and the C4
change is training-loss-only. Original custom training code is NOT supplied or reconstructed.
Only the exact audited checkpoint/class is accepted; original source is never modified.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import types

os.environ["YOLO_AUTOINSTALL"] = "false"
import torch
import ultralytics
from ultralytics.nn.tasks import PoseModel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/pallet_pose_yolo26n_c4/pallet_yolo26n_pose_c4.pt"
OUTPUT = ROOT / "training/symmetry/models/pallet_yolo26n_pose_c5_inference.pt"
SHA256 = "38c8bc1a2fc128f320af83e6edb709d083dd168e6883649aad176f6ddc1a7a76"
REVISION = "2401953a1a58cccf378cf0b1f2efa53dc226a4ec"


class InferenceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("pallet_yolo_loss.model", "ChallengeC4PoseModel"):
            return PoseModel
        return super().find_class(module, name)


def main():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Only the audited public checkpoint is supported")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(4)
    backend = types.ModuleType("c5_inference_pickle")
    backend.Unpickler = InferenceUnpickler
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False, pickle_module=backend)
    model = checkpoint.get("ema") or checkpoint["model"]
    assert type(model) is PoseModel
    assert list(model.model[-1].kpt_shape) == [9, 3]
    custom_modules = [(n, type(m).__module__) for n, m in model.named_modules()
                      if not type(m).__module__.startswith(("torch.", "ultralytics."))]
    if custom_modules or getattr(model, "criterion", None) is not None:
        raise ValueError(f"Unexpected custom inference graph/training criterion: {custom_modules}")
    original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Rebuild the stock graph from checkpoint YAML; strict loading rules out missing/extra parameters.
    stock = PoseModel(cfg=copy.deepcopy(model.yaml), ch=3, nc=len(model.names), verbose=False)
    stock.load_state_dict(original_state, strict=True)
    stock.eval()
    mapped = copy.deepcopy(model).float().eval()
    stock_types = [(n, type(m)) for n, m in stock.named_modules()]
    mapped_types = [(n, type(m)) for n, m in mapped.named_modules()]
    assert stock_types == mapped_types, "Serialized graph differs from the standard YAML graph"
    torch.manual_seed(0)
    with torch.inference_mode():
        sample = torch.rand(1, 3, 128, 160)
        a, b = mapped(sample), stock(sample)
    differences = []

    def check(a, b):
        if isinstance(a, torch.Tensor):
            assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
            differences.append(float((a - b).abs().max()))
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                check(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                check(x, y)
        else:
            assert a == b

    check(a, b)
    metadata = {"repository": "CanelE452/pallet-pose-yolo26n-c4", "revision": REVISION,
                "source_sha256": SHA256, "original_class": "pallet_yolo_loss.model.ChallengeC4PoseModel",
                "export_class": "ultralytics.nn.tasks.PoseModel", "inference_only": True,
                "basis": "Pinned author README: C4 affects training loss only; inference is stock YOLO.",
                "limitation": "Original custom Python source is absent; no independent comparison against its implementation.",
                "strict_stock_state_load": True, "stock_graph_types_match": True,
                "stock_forward_max_abs_diff": max(differences), "ultralytics": ultralytics.__version__}
    model.eval()
    checkpoint.update(model=model, ema=None, optimizer=None, scaler=None, epoch=-1,
                      inference_compatibility=metadata)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, OUTPUT)
    # Normal torch.load, with no custom pickle or import hooks, must work.
    reloaded = torch.load(OUTPUT, map_location="cpu", weights_only=False)["model"]
    assert type(reloaded) is PoseModel
    assert reloaded.state_dict().keys() == original_state.keys()
    assert all(v.dtype == reloaded.state_dict()[k].dtype and torch.equal(v, reloaded.state_dict()[k])
               for k, v in original_state.items()), "Export changed tensors"
    metadata["all_state_tensors_bit_exact"] = True
    metadata["output_sha256"] = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print(json.dumps(metadata, indent=2))
    print(f"EXPORTED {OUTPUT}")


if __name__ == "__main__":
    main()
