from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import cv2
import numpy as np


# geometry.py의 depth 전용 의존성이 없는 환경에서도 PnP 테스트만 실행한다.
try:
    import pyrealsense2  # noqa: F401
except ImportError:
    sys.modules["pyrealsense2"] = types.ModuleType("pyrealsense2")

try:
    import sklearn.linear_model  # noqa: F401
except ImportError:
    sklearn_module = types.ModuleType("sklearn")
    linear_model_module = types.ModuleType("sklearn.linear_model")
    linear_model_module.RANSACRegressor = object
    sklearn_module.linear_model = linear_model_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.linear_model"] = linear_model_module

DEPTH_CAM_DIR = Path(__file__).resolve().parents[1]
if str(DEPTH_CAM_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_CAM_DIR))

from calib.geometry import (  # noqa: E402
    pallet_keypoints_3d,
    pose_from_visible_kpts_pnp,
    visible_pnp_correspondences,
)


class Intrinsics:
    fx = 605.9
    fy = 606.0
    ppx = 317.6
    ppy = 256.3
    coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]


class VisibleKeypointPnPTests(unittest.TestCase):
    def setUp(self):
        self.intrin = Intrinsics()
        self.camera_matrix = np.array([
            [self.intrin.fx, 0.0, self.intrin.ppx],
            [0.0, self.intrin.fy, self.intrin.ppy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.object_points = pallet_keypoints_3d()
        self.true_rvec = np.array([[0.0], [np.deg2rad(20.0)], [0.0]])
        self.true_tvec = np.array([[0.18], [-0.04], [2.6]])
        image_points, _ = cv2.projectPoints(
            self.object_points, self.true_rvec, self.true_tvec,
            self.camera_matrix, np.zeros((5, 1), dtype=np.float64),
        )
        self.keypoints = np.column_stack(
            (image_points.reshape(-1, 2), np.full(9, 0.95)),
        ).astype(np.float32)

    def test_all_nine_visible_points_are_selected(self):
        obj, img, indices = visible_pnp_correspondences(self.keypoints)
        self.assertEqual(obj.shape, (9, 3))
        self.assertEqual(img.shape, (9, 2))
        np.testing.assert_array_equal(indices, np.arange(9))

    def test_occluded_front_point_is_excluded_but_pose_still_uses_others(self):
        keypoints = self.keypoints.copy()
        keypoints[0, 2] = 0.1
        _obj, _img, indices = visible_pnp_correspondences(keypoints)
        self.assertNotIn(0, indices)
        self.assertEqual(len(indices), 8)

        ok, yaw, _pitch, _roll, center, _rvec, _tvec, info = (
            pose_from_visible_kpts_pnp(keypoints, self.intrin)
        )
        self.assertTrue(ok)
        self.assertEqual(info["n_used"], 8)
        self.assertAlmostEqual(yaw, 20.0, delta=0.25)
        np.testing.assert_allclose(center, self.true_tvec.reshape(3), atol=0.01)

    def test_ransac_rejects_one_visible_outlier(self):
        keypoints = self.keypoints.copy()
        keypoints[6, :2] += np.array([100.0, -80.0], dtype=np.float32)
        ok, yaw, _pitch, _roll, center, _rvec, _tvec, info = (
            pose_from_visible_kpts_pnp(keypoints, self.intrin)
        )
        self.assertTrue(ok)
        self.assertLess(info["n_used"], 9)
        self.assertAlmostEqual(yaw, 20.0, delta=0.25)
        np.testing.assert_allclose(center, self.true_tvec.reshape(3), atol=0.01)

    def test_four_front_points_keep_ippe_fallback(self):
        keypoints = self.keypoints.copy()
        keypoints[4:, 2] = 0.1
        ok, yaw, _pitch, _roll, center, _rvec, _tvec, info = (
            pose_from_visible_kpts_pnp(keypoints, self.intrin)
        )
        self.assertTrue(ok)
        self.assertEqual(info["src"], "pnp4_ippe")
        self.assertAlmostEqual(yaw, 20.0, delta=0.25)
        np.testing.assert_allclose(center, self.true_tvec.reshape(3), atol=0.01)

    def test_largest_projected_vertical_face_becomes_front(self):
        cases = (
            (0.0, "z_min", 0.0),
            (70.0, "x_max", -20.0),
            (-70.0, "x_min", 20.0),
            (180.0, "z_max", 0.0),
        )
        for source_yaw, expected_face, expected_canonical_yaw in cases:
            with self.subTest(source_yaw=source_yaw):
                source_rvec = np.array([
                    [0.0], [np.deg2rad(source_yaw)], [0.0],
                ])
                image_points, _ = cv2.projectPoints(
                    self.object_points, source_rvec, self.true_tvec,
                    self.camera_matrix, np.zeros((5, 1), dtype=np.float64),
                )
                keypoints = np.column_stack(
                    (image_points.reshape(-1, 2), np.full(9, 0.95)),
                ).astype(np.float32)
                ok, yaw, _pitch, _roll, _center, _rvec, _tvec, info = (
                    pose_from_visible_kpts_pnp(keypoints, self.intrin)
                )
                self.assertTrue(ok)
                self.assertEqual(info["selected_front_face"], expected_face)
                self.assertAlmostEqual(yaw, expected_canonical_yaw, delta=0.25)
                self.assertGreater(info["selected_face_area_px2"], 0.0)


if __name__ == "__main__":
    unittest.main()
