"""Regression tests for valid permutations, complete-instance matching and gradients."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys

import numpy as np
import torch
from symmetry_pose import SymmetryPoseLoss, cuboid_points, symmetry_permutations


class SymmetryTests(unittest.TestCase):
    def criterion(self, order=4):
        loss = SymmetryPoseLoss.__new__(SymmetryPoseLoss)
        loss.permutations = torch.tensor(symmetry_permutations(order))
        loss.generated_weight = .25
        loss.rle_loss = None
        loss.keypoint_loss = SimpleNamespace(sigmas=torch.ones(9) / 9)
        return loss

    def test_rotations_preserve_edges_height_and_center(self):
        points = cuboid_points()
        distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
        perms = symmetry_permutations()
        self.assertEqual(len(set(map(tuple, perms))), 4)
        for p in perms:
            np.testing.assert_allclose(distances, distances[p][:, p])
            np.testing.assert_allclose(points[:, 1], points[p, 1])
            self.assertEqual(p[8], 8)
        np.testing.assert_array_equal(perms[2], [5, 4, 7, 6, 1, 0, 3, 2, 8])

    def test_all_rotations_are_zero_error_and_fixed_indices_are_not(self):
        torch.manual_seed(42)
        gt = torch.cat((torch.rand(1, 9, 2) * 100, torch.tensor([[[2.], [2.], [2.], [1.], [1.], [2.], [1.], [1.], [1.]]])), -1)
        for perm in symmetry_permutations():
            pred = gt[:, perm].clone()
            pred[..., 2] = 5
            terms = self.criterion().candidate_terms(pred, gt, torch.tensor([[10000.]]))
            self.assertAlmostEqual(terms[0].min().item(), 0, places=7)
            if not np.array_equal(perm, np.arange(9)):
                self.assertGreater(self.criterion(1).candidate_terms(pred, gt, torch.tensor([[10000.]]))[0].item(), .01)

    def test_arbitrary_corner_swap_is_not_a_symmetry(self):
        gt = torch.tensor([[[0, 0, 2], [100, 0, 2], [100, 15, 2], [0, 15, 1],
                            [20, -30, 1], [90, -30, 2], [90, -20, 1], [20, -20, 1], [50, -5, 1]]], dtype=torch.float32)
        pred = gt.clone()
        pred[:, [0, 1]] = pred[:, [1, 0]]
        self.assertGreater(self.criterion().candidate_terms(pred, gt, torch.tensor([[6000.]]))[0].min().item(), .01)

    def test_missing_points_finite_gradient_and_source_weights_follow_permutation(self):
        gt = torch.rand(2, 9, 3) * 10
        gt[..., 2] = 1
        gt[:, :4, 2] = 2
        gt[0, 0, 2] = 0
        gt[1, :, 2] = 0
        pred = torch.randn(2, 9, 3, requires_grad=True)
        terms = self.criterion().candidate_terms(pred, gt, torch.ones(2, 1) * 100)
        total = terms[0] * 12 + terms[1]
        total.min(1).values.mean().backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(pred.grad[0, 0, :2].shape, (2,))
        shifted = gt[:, symmetry_permutations()[1]]
        other = self.criterion().candidate_terms(pred.detach(), shifted, torch.ones(2, 1) * 100)
        torch.testing.assert_close(total.min(1).values.detach(), (other[0] * 12 + other[1]).min(1).values)

    def test_recording_disjoint_dataset(self):
        root = Path(__file__).resolve().parents[2]
        rows = json.loads((root / "datasets/pallet_symmetry_v1/manifest.json").read_text(encoding="utf8"))
        groups, hashes = {}, {}
        for row in rows:
            groups.setdefault(row["split"], set()).add(row["group"])
            hashes.setdefault(row["split"], set()).add(row["image_sha256"])
            values = np.loadtxt(row["label"]).reshape(-1)
            self.assertEqual(len(values), 32)
            self.assertTrue(np.isfinite(values).all())
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            self.assertFalse(groups[a] & groups[b])
            self.assertFalse(hashes[a] & hashes[b])

    def test_rle_matches_upstream_for_single_identity_target(self):
        from ultralytics.utils.loss import RLELoss

        class Flow(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.tensor(0.))

            def log_prob(self, x):
                return -.5 * x.square().sum(-1) + self.offset

        loss = self.criterion(1)
        loss.generated_weight = 1.
        loss.rle_loss = RLELoss(use_target_weight=True)
        loss.flow_model = Flow()
        loss.target_weights = torch.ones(9)
        gt = torch.rand(1, 9, 3)
        gt[..., 2] = 2
        pred = torch.rand(1, 9, 5, requires_grad=True)
        terms = loss.candidate_terms(pred, gt, torch.tensor([[100.]]))
        reference = loss.calculate_rle_loss(pred, gt, gt[...,2]>0).clamp_min(0)
        torch.testing.assert_close(terms[2].mean(), reference)
        (terms[0]+terms[1]+terms[2]).sum().backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad[..., -2:].abs().sum().item(), 0)
        self.assertIsNotNone(loss.flow_model.offset.grad)

    def test_runtime_front_is_invariant_under_c4_numbering(self):
        import cv2
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "extracted/depth_cam"))
        from calib.geometry import pallet_keypoints_3d, pose_from_visible_kpts_pnp
        intrin = SimpleNamespace(fx=614.18, fy=614.31, ppx=329.28, ppy=234.53, coeffs=[0.]*5)
        K = np.array([[intrin.fx, 0, intrin.ppx], [0, intrin.fy, intrin.ppy], [0,0,1]])
        for angle in [-70, -20, 0, 20, 70, 170]:
            pts, _ = cv2.projectPoints(pallet_keypoints_3d(), np.array([0., np.deg2rad(angle), 0.]),
                                       np.array([.1,.2,4.]), K, np.zeros(5))
            kpts = np.c_[pts.reshape(9,2), np.full(9,.99)]
            reference = None
            for perm in symmetry_permutations():
                result = pose_from_visible_kpts_pnp(kpts[perm], intrin)
                self.assertTrue(result[0])
                canonical = np.r_[result[1:4], result[4]]
                if reference is None:
                    reference = canonical
                else:
                    np.testing.assert_allclose(canonical, reference, atol=.02)


if __name__ == "__main__":
    unittest.main()
