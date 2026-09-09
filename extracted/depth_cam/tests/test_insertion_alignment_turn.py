"""Offline insertion alignment planning and FSM routing, no hardware."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.fsm_v4 import planner as p, config as cfg
from calib.fsm_v4.pose import VisualPoseFilter
from calib.fsm_v4.top import CalibrationFSMV4


class InsertionTurnTests(unittest.TestCase):
    def setUp(self):
        width = patch.object(cfg, 'FORK_WIDTH_M', .10)
        width.start()
        self.addCleanup(width.stop)

    def pose(self, x=0., z=1.8, yaw=0.):
        return VisualPoseFilter().seed(yaw, x, z, 100.)

    def test_aligned_needs_no_turn_and_out_of_range_is_not_eligible(self):
        self.assertEqual(p.insertion_alignment_turn(self.pose()), 0.)
        self.assertIsNone(p.insertion_alignment_turn(
            self.pose(z=cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M + .001)))

    def test_commandable_left_correction_meets_all_gates(self):
        pose = self.pose(yaw=-5.)
        turn = p.insertion_alignment_turn(pose)
        self.assertLess(turn, 0.)
        self.assertGreaterEqual(abs(turn), cfg.INSERT_FINE_MIN_TURN_DEG)
        predicted = p._pose_after_action(pose, turn, 0.)
        self.assertTrue(p.fork_opening_alignment(predicted)[0])
        self.assertLessEqual(abs(predicted.yaw_deg), cfg.FINAL_YAW_TOL_DEG)
        self.assertTrue(p.action_keeps_front_visible(pose, turn, 0., None))

    def test_old_front_only_log_is_rejected_and_fine_turn_clears_blocks(self):
        old = self.pose(x=-.051774755, z=1.7944546, yaw=-9.2359786)
        self.assertIsNone(p.insertion_alignment_turn(old))
        pose = self.pose(yaw=-5.)
        turn = p.insertion_alignment_turn(pose)
        self.assertEqual(turn, -.5)
        self.assertTrue(p.fork_opening_alignment(p._pose_after_action(pose, turn, 0.))[0])

    def test_visibility_rejection_prevents_rotation(self):
        with patch('calib.fsm_v4.planner.action_keeps_front_visible',
                   side_effect=lambda pose, turn, forward, meta: turn == 0.):
            self.assertIsNone(p.insertion_alignment_turn(self.pose(x=-.1)))

    def test_detected_last_run_pose_cannot_bypass_internal_walls(self):
        pose = self.pose(x=-.3997410834, z=2.1479372978, yaw=-15.4230880354)
        self.assertFalse(p.action_keeps_front_visible(pose, 0., 0.))
        self.assertIsNone(p.insertion_alignment_turn(pose))
        f = self.fsm()
        f._near_insertion_step(pose, None, [])
        f._begin_rotation.assert_not_called()
        # Replanning may inspect the remaining approach room, but this pose
        # still cannot bypass the planner's visibility or insertion checks.
        f._set_state.assert_called_once_with('STAGING_PLAN')
        f._fail.assert_not_called()
        self.assertTrue(f._near_approach_active)

    def fsm(self):
        f = object.__new__(CalibrationFSMV4)
        f._insertion_alignment_attempts = 0
        f._insertion_fine_started_mono = None
        f._forward_used_m = 0.
        f._correction_cycles = 0
        f._alignment_started_mono = None
        f.state = 'FINAL_POSE_LOCK'
        f._begin_rotation = Mock(return_value=True)
        f._set_state = Mock()
        f._fail = Mock()
        return f

    def test_entry_distance_is_not_reapplied_after_alignment_turn(self):
        pose = self.pose(x=-.1, z=2.1995, yaw=-5.)
        turn = p.insertion_alignment_turn(pose)
        self.assertIsNotNone(turn)
        predicted = p._pose_after_action(pose, turn, 0.)
        self.assertGreater(predicted.pallet_z_m, cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M)
        self.assertTrue(p.fork_opening_alignment(
            predicted, enforce_entry_distance=False)[0])
        # A fresh run still cannot enter from outside the activation distance.
        self.assertIsNone(p.insertion_alignment_turn(predicted))
        fresh = self.fsm()
        self.assertFalse(fresh._near_insertion_step(predicted, None, []))
        fresh._begin_rotation.assert_not_called()
        # After entry, recheck the observed pose and capture its actual Z.
        f = self.fsm()
        self.assertTrue(f._near_insertion_step(pose, None, []))
        f._begin_rotation.assert_called_once()
        self.assertTrue(f._near_insertion_step(predicted, None, []))
        f._fail.assert_not_called()
        f._set_state.assert_called_once_with('READY_TO_INSERT')
        self.assertAlmostEqual(f._insert_accepted_z_m, predicted.pallet_z_m)
        self.assertAlmostEqual(f._insert_remaining_m,
                               predicted.pallet_z_m - cfg.INSERT_CAMERA_Z_REMAINDER_M)

    def test_live_edge_does_not_interrupt_planned_rotations(self):
        for mode in ('insert_align', 'waypoint', 'final'):
            with self.subTest(mode=mode):
                f = self.fsm()
                f._rotation_mode = mode
                f._rotation_error = Mock(return_value=-6.5)
                f._direction_changes = 0
                update = SimpleNamespace(
                    timed_out=False, should_stop=False, ready=False,
                    predicted_error_deg=-6.5, stop_margin_deg=.5,
                    elapsed_sec=.093, plan_hold_sec=1.813,
                    coast_margin_deg=0., measured_error_rate_deg_s=0.,
                )
                f._rotation = Mock(command='ROT_LEFT', start_error_deg=-6.5)
                f._rotation.update.return_value = update
                f._exec = Mock()
                f.execu = Mock()
                f.status = Mock()
                f._capture_rotation_stop = Mock()
                f._begin_settle = Mock()
                with patch.object(cfg, 'ROT_USE_FITTED_RESPONSE', True), \
                     patch.object(cfg, 'IMAGE_EDGE_VISIBILITY_GUARD_ENABLED', True):
                    f._rotation_step(self.pose(), 0., {'bbox_margin_norm': 0.01},
                                     'FINAL_SETTLE', [])
                f.execu.exec.assert_called_once_with('ROT_LEFT')
                f._exec.assert_not_called()
                f._begin_settle.assert_not_called()

    def test_fsm_turns_then_requires_final_recheck(self):
        f = self.fsm()
        self.assertTrue(f._near_insertion_step(self.pose(yaw=-5.), None, []))
        self.assertEqual(f._begin_rotation.call_args.args[:2], ('FINAL_ROTATE', 'insert_align'))
        f._set_state.assert_not_called()  # no immediate insertion on predicted fit
        self.assertEqual(f._insertion_alignment_attempts, 1)

    def test_fsm_aligned_goes_to_ready_and_budget_stops_retries(self):
        f = self.fsm()
        self.assertTrue(f._near_insertion_step(self.pose(), None, []))
        f._set_state.assert_called_once_with('READY_TO_INSERT')
        f = self.fsm()
        f._insertion_alignment_attempts = cfg.INSERT_ALIGNMENT_MAX_CORRECTIONS
        with patch('calib.fsm_v4.top.insertion_alignment_turn', return_value=-3.):
            f._near_insertion_step(self.pose(yaw=-5.), None, [])
        f._fail.assert_called_once()
        f._begin_rotation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
