"""C4-aware YOLO26 pose training; tested against Ultralytics 8.4.138.

The scalar objective is the per-positive minimum of gain-weighted pose, presence
and RLE terms. All nine coordinates, masks and source weights share ONE chosen
permutation. The ordinary detection assignment/loss and E2E branches are retained.
"""
from copy import copy

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.loss import E2ELoss, PoseLoss26
from ultralytics.utils.metrics import kpt_iou
from ultralytics.utils.ops import xyxy2xywh


def cuboid_points():
    return np.array([[-.55, -.075, -.55], [.55, -.075, -.55],
                     [.55, .075, -.55], [-.55, .075, -.55],
                     [-.55, -.075, .55], [.55, -.075, .55],
                     [.55, .075, .55], [-.55, .075, .55], [0, 0, 0]])


def symmetry_permutations(order=4):
    if order not in (1, 2, 4):
        raise ValueError("Only identity, C2 and C4 rotations are allowed")
    points = cuboid_points()
    perms = []
    for angle in np.arange(order) * 2 * np.pi / order:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        distances = np.linalg.norm((points @ rotation.T)[:, None] - points[None], axis=-1)
        perm = distances.argmin(1)
        if not np.all(distances[np.arange(9), perm] < 1e-7):
            raise ValueError("Rotation does not preserve cuboid")
        perms.append(perm)
    return np.asarray(perms)


class SymmetryPoseLoss(PoseLoss26):
    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        if list(self.kpt_shape) != [9, 3]:
            raise ValueError(f"Expected [9,3] output, got {self.kpt_shape}")
        self.permutations = torch.as_tensor(symmetry_permutations(model.symmetry_order), device=self.device)
        self.generated_weight = model.generated_weight
        self.selection_counts = torch.zeros(len(self.permutations), device=self.device, dtype=torch.long)

    def candidate_terms(self, pred, gt, area):
        """Return [positive, symmetry] loss components before their gains."""
        targets = gt[:, self.permutations]  # N,G,9,3
        valid = targets[..., 2] > 0
        weights = torch.where(targets[..., 2] >= 2, 1.0, self.generated_weight) * valid
        denominator = weights.sum(-1).clamp_min(1e-9)
        delta = pred[:, None, :, :2].float() - targets[..., :2].float()
        distance = delta.square().sum(-1)
        sigma = self.keypoint_loss.sigmas.float().view(1, 1, -1)
        exponent = distance / ((2 * sigma).square() * area.float().view(-1, 1, 1).clamp_min(1e-9) * 2)
        pose = ((1 - (-exponent).exp()) * weights).sum(-1) / denominator
        logits = pred[:, None, :, 2].float().expand_as(valid)
        presence = F.binary_cross_entropy_with_logits(logits, valid.float(), reduction="none").mean(-1)
        rle = pose * 0
        if self.rle_loss is not None and pred.shape[-1] == 5:
            sigmas = pred[:, None, :, -2:].float().sigmoid().clamp_min(1e-6).expand_as(delta)
            errors = (delta / sigmas).clamp(-100, 100)
            flat_errors = errors[valid]
            if flat_errors.numel():
                # Flow parameters may be half precision during EMA validation.
                flow_dtype = next(self.flow_model.parameters()).dtype
                log_phi = self.flow_model.log_prob(flat_errors.to(flow_dtype)).float()
                log_phi_full = torch.zeros_like(distance).masked_scatter(valid, log_phi)
                point_nll = (sigmas.log() - log_phi_full[..., None]
                             + (2 * sigmas).log() + errors.abs()).sum(-1)
                rle = ((point_nll * weights).sum(-1) / denominator).clamp_min(0)
            else:
                rle = rle + self._rle_zero(pred)
        return pose, presence, rle

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        selected = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        if not masks.any():
            zero = pred_kpts.sum() * 0
            return zero, zero, zero
        gt = selected[masks].clone()
        strides = stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks]
        gt[..., :2] /= strides[:, None, None]
        boxes = target_bboxes[masks] / strides[:, None]
        area = xyxy2xywh(boxes)[:, 2:].prod(-1, keepdim=True)
        terms = self.candidate_terms(pred_kpts[masks], gt, area)
        total = terms[0] * self.hyp.pose + terms[1] * self.hyp.kobj + terms[2] * self.hyp.rle
        choice = total.detach().argmin(1)
        self.selection_counts += torch.bincount(choice, minlength=len(self.permutations))
        rows = torch.arange(len(choice), device=choice.device)
        return tuple(term[rows, choice].mean() for term in terms)


class SymmetryPoseModel(PoseModel):
    symmetry_order = 4
    generated_weight = .25

    def init_criterion(self):
        return E2ELoss(self, SymmetryPoseLoss) if self.end2end else SymmetryPoseLoss(self)


class SymmetryPoseValidator(PoseValidator):
    """Max whole-cuboid OKS across C4, for both control and symmetry models."""
    def _process_batch(self, preds, batch):
        tp = DetectionValidator._process_batch(self, preds, batch)
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp["tp_p"] = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            area = xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * .53
            perms = torch.as_tensor(symmetry_permutations(), device=batch["keypoints"].device)
            overlaps = torch.stack([kpt_iou(batch["keypoints"][:, p], preds["keypoints"],
                                           sigma=self.sigma, area=area) for p in perms]).amax(0)
            tp["tp_p"] = self.match_predictions(preds["cls"], batch["cls"], overlaps).cpu().numpy()
        return tp


class SymmetryPoseTrainer(PoseTrainer):
    symmetry_order = 4
    generated_weight = .25

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(SymmetryPoseModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"], verbose=verbose))
        model.symmetry_order = self.symmetry_order
        model.generated_weight = self.generated_weight
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        return SymmetryPoseValidator(self.test_loader, save_dir=self.save_dir,
                                     args=copy(self.args), _callbacks=self.callbacks)


class FixedIndexTrainer(SymmetryPoseTrainer):
    symmetry_order = 1
