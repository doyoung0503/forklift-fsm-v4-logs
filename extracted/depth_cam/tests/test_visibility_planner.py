from __future__ import annotations

import math
import sys
import unittest
from unittest.mock import patch
from pathlib import Path


DEPTH_CAM_DIR = Path(__file__).resolve().parents[1]
if str(DEPTH_CAM_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_CAM_DIR))

from calib.fsm_v4 import config as cfg  # noqa: E402
from calib.fsm_v4.planner import (  # noqa: E402
    action_keeps_front_visible,
    adaptive_forward_limit_m,
    safe_straight_continuation_m,
    action_visibility_margin_deg,
    configured_visible_half_angle_deg,
    plan_waypoint,
    staging_position_errors,
    staging_position_reached,
)
from calib.fsm_v4.pose import (  # noqa: E402
    VisualPose,
    rotation_center_in_pallet,
    wrap_180,
)


def make_pose(yaw_deg: float, x_m: float, z_m: float) -> VisualPose:
    rot_x, rot_z = rotation_center_in_pallet(yaw_deg, x_m, z_m)
    return VisualPose(
        yaw_deg, x_m, z_m, rot_x, rot_z,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )


def apply_action(pose: VisualPose, turn_deg: float, forward_m: float) -> VisualPose:
    angle = math.radians(turn_deg)
    c, s = math.cos(angle), math.sin(angle)
    dx = pose.pallet_x_m - cfg.CAMERA_TO_ROT_CENTER_X_M
    dz = pose.pallet_z_m - cfg.CAMERA_TO_ROT_CENTER_Z_M
    x_m = c * dx - s * dz + cfg.CAMERA_TO_ROT_CENTER_X_M
    z_m = (
        s * dx + c * dz + cfg.CAMERA_TO_ROT_CENTER_Z_M - forward_m
    )
    return make_pose(wrap_180(pose.yaw_deg - turn_deg), x_m, z_m)


def synthetic_vision_meta(pose: VisualPose, vertical_offset_m: float = 0.0):
    width_px, height_px = 1280.0, 720.0
    fx = 0.5 * width_px / math.tan(math.radians(35.0))
    fy = fx
    yaw = math.radians(pose.yaw_deg)
    half_w = 0.5 * cfg.PALLET_FRONT_VISIBILITY_WIDTH_M
    half_h = 0.075
    corners = []
    for local_x, local_y in (
        (-half_w, -half_h), (half_w, -half_h),
        (half_w, half_h), (-half_w, half_h),
    ):
        corners.append([
            pose.pallet_x_m + math.cos(yaw) * local_x,
            vertical_offset_m + local_y,
            pose.pallet_z_m - math.sin(yaw) * local_x,
        ])
    return {
        "face_corners_camera_m": corners,
        "fx": fx,
        "fy": fy,
        "ppx": width_px * 0.5,
        "ppy": height_px * 0.5,
        "frame_width": width_px,
        "frame_height": height_px,
    }


class VisibilityGeometryTests(unittest.TestCase):
    def test_configured_55_degree_fov_and_eight_percent_margin(self):
        self.assertAlmostEqual(configured_visible_half_angle_deg(), 23.61862, 5)

    def test_rotation_then_forward_counterexample_is_rejected(self):
        pose = make_pose(0.0, 0.4, 3.0)
        turn = math.degrees(math.atan2(0.4, 1.5))
        self.assertGreater(action_visibility_margin_deg(pose, turn, 0.0), 0.0)
        self.assertFalse(action_keeps_front_visible(pose, turn, 0.8))
        self.assertLess(action_visibility_margin_deg(pose, turn, 0.8), 0.0)

    def test_vertical_front_face_clipping_is_rejected_with_3d_metadata(self):
        pose = make_pose(0.0, 0.0, 3.0)
        meta = synthetic_vision_meta(pose, vertical_offset_m=2.0)
        self.assertFalse(action_keeps_front_visible(pose, 0.0, 0.0, meta))
        result = plan_waypoint(pose, 0.0, 0.20, 2.2, meta)
        self.assertIsNone(result.waypoint)
        self.assertFalse(result.needs_recenter)
        self.assertIn("vertical", result.reason)

    def test_stream_intrinsics_can_narrow_the_55_degree_fallback(self):
        pose = make_pose(0.0, 0.7, 3.0)
        # A narrower stream profile must override even the conservative
        # configured fallback when its intrinsics are available.
        meta = synthetic_vision_meta(pose)
        meta.update({
            "fx": 772.55,
            "fy": 772.55,
            "ppx": 320.0,
            "ppy": 240.0,
            "frame_width": 640.0,
            "frame_height": 480.0,
        })
        self.assertGreater(action_visibility_margin_deg(pose, 0.0, 0.0), 0.0)
        self.assertFalse(action_keeps_front_visible(pose, 0.0, 0.0, meta))
        del meta["face_corners_camera_m"]
        self.assertFalse(action_keeps_front_visible(pose, 0.0, 0.0, meta))


class JointPlannerTests(unittest.TestCase):
    def test_adaptive_cap_thresholds_and_signs(self):
        for sign in (-1, 1):
            for lateral, expected in ((0., 1.5), (.1, 1.5), (.15, 1.15),
                                      (.2, .8), (.4, .8)):
                with self.subTest(sign=sign, lateral=lateral):
                    pose = make_pose(0., cfg.CAMERA_TO_ROT_CENTER_X_M - sign * lateral, 4.)
                    self.assertAlmostEqual(adaptive_forward_limit_m(pose), expected)

    def test_extended_drive_bounds_predicted_lateral(self):
        pose = make_pose(0., cfg.CAMERA_TO_ROT_CENTER_X_M, 4.)
        for turn in (-10., 10.):
            cap = adaptive_forward_limit_m(pose, turn)
            self.assertGreater(cap, .8)
            self.assertLess(cap, 1.5)
            self.assertAlmostEqual(abs(apply_action(pose, turn, cap).rot_x_pallet_m), .2)
        self.assertAlmostEqual(adaptive_forward_limit_m(pose, 20.), .8)

    def test_continuation_rechecks_pose_request_and_budget(self):
        pose = make_pose(0., 0., 4.)
        self.assertAlmostEqual(safe_straight_continuation_m(pose, 1.5, 5.), 1.5)
        self.assertAlmostEqual(safe_straight_continuation_m(pose, 1.5, .6), .6)
        self.assertAlmostEqual(safe_straight_continuation_m(pose, .7, 5.), .7)
        changed = make_pose(0., .25, 4.)
        self.assertLessEqual(safe_straight_continuation_m(changed, 1.5, 5.), .8)
        turned = apply_action(pose, 10., 0.)
        self.assertLessEqual(safe_straight_continuation_m(turned, 1.5, 5.),
                               adaptive_forward_limit_m(turned))

    def test_legacy_planner_uses_adaptive_cap(self):
        pose = make_pose(0., 0., 4.)
        with patch.object(cfg, 'PREDICTIVE_VISIBILITY_PLANNER_ENABLED', False):
            result = plan_waypoint(pose, 0., .2, 5., None)
        self.assertAlmostEqual(result.waypoint.forward_m, 1.5)

    def test_logged_margin_breach_plans_directly_without_recenter(self):
        # 2026-09-06 16:54:18: visible, but about 0.55 degrees outside ROI.
        pose = make_pose(-20.986, -.884, 3.312)
        for meta in (None, synthetic_vision_meta(pose)):
            with self.subTest(metadata=meta is not None):
                self.assertFalse(action_keeps_front_visible(pose, 0., 0., meta))
                result = plan_waypoint(pose, -15., .01, 3., meta)
                self.assertFalse(result.needs_recenter)
                self.assertIsNotNone(result.waypoint)
                w = result.waypoint
                self.assertTrue(action_keeps_front_visible(
                    pose, w.turn_deg, 0., meta,
                    half_angle_deg=cfg.CAMERA_HORIZONTAL_FOV_DEG / 2,
                    edge_margin_norm=0.,
                ))
                turned = apply_action(pose, w.turn_deg, 0.)
                self.assertTrue(action_keeps_front_visible(
                    turned, 0., w.forward_m,
                    None if meta is None else synthetic_vision_meta(turned),
                ))
                self.assertGreaterEqual(w.forward_m, cfg.FWD_RELIABLE_MIN_DISTANCE_M)

    def test_physically_clipped_start_has_no_blind_recovery_plan(self):
        pose = make_pose(-20., -2., 3.)
        result = plan_waypoint(pose, -34., -.1, 3., synthetic_vision_meta(pose))
        self.assertIsNone(result.waypoint)
        self.assertFalse(result.needs_recenter)

    def test_centered_target_gets_full_safe_macro_step(self):
        pose = make_pose(0.0, 0.0, 4.0)
        result = plan_waypoint(
            pose, 0.0, 0.20, cfg.FWD_MAX_TOTAL_CORRECTION_M,
            synthetic_vision_meta(pose),
        )
        self.assertIsNotNone(result.waypoint)
        waypoint = result.waypoint
        self.assertEqual(waypoint.turn_deg, 0.0)
        self.assertAlmostEqual(waypoint.forward_m, 1.5)
        self.assertGreaterEqual(waypoint.visibility_min_margin_deg, 0.0)

    def test_offset_target_does_not_turn_directly_into_forward_clipping(self):
        pose = make_pose(0.0, 0.75, 3.0)
        first = plan_waypoint(
            pose, math.degrees(math.atan2(0.75, 3.0)), 0.20,
            cfg.FWD_MAX_TOTAL_CORRECTION_M, synthetic_vision_meta(pose),
        )
        self.assertIsNotNone(first.waypoint)
        self.assertLess(first.waypoint.turn_deg, 26.0)
        turned = apply_action(pose, first.waypoint.turn_deg, 0.0)
        second = plan_waypoint(
            turned,
            math.degrees(math.atan2(turned.pallet_x_m, turned.pallet_z_m)),
            0.20, cfg.FWD_MAX_TOTAL_CORRECTION_M,
            synthetic_vision_meta(turned),
        )
        self.assertIsNotNone(second.waypoint)
        self.assertEqual(second.waypoint.turn_deg, 0.0)
        self.assertTrue(action_keeps_front_visible(
            turned, 0.0, second.waypoint.forward_m,
            synthetic_vision_meta(turned),
        ))

    def test_sub_commandable_turn_is_never_returned(self):
        pose = make_pose(0.0, 0.05, 3.0)
        result = plan_waypoint(
            pose, math.degrees(math.atan2(0.05, 3.0)), 0.20,
            cfg.FWD_MAX_TOTAL_CORRECTION_M, synthetic_vision_meta(pose),
        )
        self.assertIsNotNone(result.waypoint)
        turn = abs(result.waypoint.turn_deg)
        self.assertTrue(turn == 0.0 or turn >= cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG)

    def test_closed_loop_moderate_offsets_stays_visible_and_reaches_staging(self):
        for initial_x in (-0.2, -0.1, 0.0, 0.1, 0.2):
            with self.subTest(initial_x=initial_x):
                pose = make_pose(0.0, initial_x, 3.0)
                used_forward = 0.0
                for _ in range(cfg.MAX_CORRECTION_CYCLES):
                    if staging_position_reached(pose):
                        break
                    meta = synthetic_vision_meta(pose)
                    result = plan_waypoint(
                        pose,
                        math.degrees(math.atan2(
                            pose.pallet_x_m, pose.pallet_z_m,
                        )),
                        0.20,
                        cfg.FWD_MAX_TOTAL_CORRECTION_M - used_forward,
                        meta,
                    )
                    self.assertIsNotNone(result.waypoint, result.reason)
                    waypoint = result.waypoint
                    self.assertTrue(action_keeps_front_visible(
                        pose, waypoint.turn_deg, waypoint.forward_m, meta,
                    ))
                    if abs(waypoint.turn_deg) >= cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG:
                        pose = apply_action(pose, waypoint.turn_deg, 0.0)
                    else:
                        pose = apply_action(pose, 0.0, waypoint.forward_m)
                        used_forward += waypoint.forward_m
                self.assertTrue(staging_position_reached(pose))
                self.assertLessEqual(used_forward, cfg.FWD_MAX_TOTAL_CORRECTION_M)

    def test_infeasible_large_offset_preserves_staging_room(self):
        pose = make_pose(0.0, 0.75, 3.0)
        used_forward = 0.0
        stopped_without_path = False
        for _ in range(cfg.MAX_CORRECTION_CYCLES):
            meta = synthetic_vision_meta(pose)
            result = plan_waypoint(
                pose,
                math.degrees(math.atan2(pose.pallet_x_m, pose.pallet_z_m)),
                0.20,
                cfg.FWD_MAX_TOTAL_CORRECTION_M - used_forward,
                meta,
            )
            if result.waypoint is None:
                stopped_without_path = True
                break
            waypoint = result.waypoint
            self.assertTrue(action_keeps_front_visible(
                pose, waypoint.turn_deg, waypoint.forward_m, meta,
            ))
            if abs(waypoint.turn_deg) >= cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG:
                pose = apply_action(pose, waypoint.turn_deg, 0.0)
            else:
                pose = apply_action(pose, 0.0, waypoint.forward_m)
                used_forward += waypoint.forward_m
        self.assertTrue(stopped_without_path)
        lateral, longitudinal = staging_position_errors(pose)
        self.assertGreater(abs(lateral), cfg.FINAL_LATERAL_TOL_M)
        self.assertLessEqual(longitudinal, -cfg.FINAL_DISTANCE_TOL_M + 1e-6)


if __name__ == "__main__":
    unittest.main()
