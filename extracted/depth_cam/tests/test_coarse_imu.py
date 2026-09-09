import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.imu_stream import ImuVideoStream
from calib import control
from calib.fsm_v4 import config as cfg
from calib.fsm_v4.coarse import coarse_plan
from calib.fsm_v4.top import CalibrationFSMV4


class CoarseTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.clock = patch('time.monotonic', side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.addCleanup(control.configure_motion_deadline, None)
        self.f = object.__new__(CalibrationFSMV4)
        self.f._coarse_reset()
        self.f._samples = []
        self.f._exec = Mock()
        self.f._trace_begin = Mock()
        self.f._trace_end = Mock()
        self.f._pose_filter = Mock()
        self.f._fail = Mock(side_effect=lambda *a: setattr(self.f, 'state', 'FAILED'))
        self.f.state = 'FACE_SETTLE'
        self.f.imu_source = Mock()
        self.yaw = 0.0
        self.rate = 0.0
        self.pose = SimpleNamespace(yaw_deg=40., rot_x_pallet_m=.8, rot_z_pallet_m=-3.)

    def tick(self, dt=.05, yaw=None, rate=0.):
        if self.f.state == 'FAILED':
            return
        self.now += dt
        if yaw is not None:
            self.yaw = yaw
        self.f.imu_source.snapshot.return_value = (self.yaw, self.now, rate, None)
        self.f._coarse_step(self.now, [], False, None, None, None, None)

    def start(self):
        self.assertTrue(self.f._maybe_begin_coarse(self.pose, self.now, []))
        for _ in range(72):
            self.tick()
        self.assertEqual(self.f.state, 'COARSE_ROTATE')

    def settle(self):
        for _ in range(43):
            self.tick()

    def test_plan_right_offset(self):
        first, distance, duration, last = coarse_plan(self.pose)
        self.assertEqual((first, distance, last), (-50., .8, 90.))
        self.assertGreater(duration, 0.)

    def test_bias_excludes_first_two_seconds(self):
        self.f._maybe_begin_coarse(self.pose, self.now, [])
        for _ in range(38):
            self.tick(rate=12.)
        self.assertEqual(self.f._coarse_bias_samples, [])
        self.assertEqual(self.f.state, 'COARSE_PREPARE')
        for _ in range(35):
            self.tick(rate=.1)
        self.assertEqual(self.f.state, 'COARSE_ROTATE')
        self.assertAlmostEqual(self.f._coarse_bias, .1)

    def test_turn_settle_requires_post_two_second_sample(self):
        self.start()
        self.tick(yaw=-50.)
        for _ in range(39):
            self.tick()
        self.assertEqual(self.f.state, 'COARSE_ROTATE_SETTLE')
        for _ in range(4):
            self.tick()
        self.assertEqual(self.f.state, 'COARSE_DRIVE')

    def test_coarse_travel_caps_can_be_disabled_without_changing_normal_drive(self):
        from calib.fsm_v4.motion import forward_seconds
        self.pose.rot_x_pallet_m = 6.0
        with patch.object(cfg, 'COARSE_TRAVEL_LIMITS_ENABLED', False):
            _, distance, duration, _ = coarse_plan(self.pose)
        self.assertEqual(distance, 6.0)
        self.assertGreater(duration, 15.0)
        self.assertEqual(forward_seconds(6.0), cfg.FWD_COMMAND_MAX_SEC)
        with patch.object(cfg, 'COARSE_TRAVEL_LIMITS_ENABLED', True):
            with self.assertRaisesRegex(ValueError, 'travel limits'):
                coarse_plan(self.pose)
            self.pose.rot_x_pallet_m = 4.0
            with self.assertRaisesRegex(ValueError, 'time range'):
                coarse_plan(self.pose)

    def test_plan_left_offset(self):
        self.pose.yaw_deg, self.pose.rot_x_pallet_m = -40., -.8
        first, distance, _, last = coarse_plan(self.pose)
        self.assertEqual((first, distance, last), (50., .8, -90.))

    def test_invalid_geometry_rejected(self):
        for field, value in [('yaw_deg', float('nan')), ('yaw_deg', 95.),
                             ('rot_z_pallet_m', -1.), ('rot_x_pallet_m', 0.)]:
            pose = SimpleNamespace(**vars(self.pose))
            setattr(pose, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                coarse_plan(pose)

    def test_small_lateral_skips_even_with_large_yaw(self):
        self.pose.rot_x_pallet_m = .01
        self.assertFalse(self.f._maybe_begin_coarse(self.pose, self.now, []))
        self.f._exec.assert_not_called()

    def test_full_sequence_both_directions_and_relative_ninety(self):
        for side in (1., -1.):
            with self.subTest(side=side):
                self.f._coarse_reset()
                self.f.state = 'FACE_SETTLE'
                self.f._observe_pose = Mock(return_value=(None, 'missing'))
                self.yaw = 0.
                self.pose.yaw_deg, self.pose.rot_x_pallet_m = side*40., side*.8
                self.start()
                self.f._exec.assert_called_with('ROT_LEFT' if side > 0 else 'ROT_RIGHT')
                self.tick(yaw=-side*51.)  # one-degree threshold overshoot
                self.assertEqual(self.f.state, 'COARSE_ROTATE_SETTLE')
                self.settle()
                self.assertEqual(self.f.state, 'COARSE_DRIVE')
                self.f._exec.assert_called_with('FWD')
                while self.f.state == 'COARSE_DRIVE':
                    self.tick()
                self.assertEqual(self.f.state, 'COARSE_DRIVE_SETTLE')
                self.settle()
                self.assertEqual(self.f.state, 'COARSE_RETURN_ROTATE')
                self.f._exec.assert_called_with('ROT_RIGHT' if side > 0 else 'ROT_LEFT')
                self.tick(yaw=side*38.)
                self.assertEqual(self.f.state, 'COARSE_RETURN_ROTATE')
                self.tick(yaw=side*39.)  # +90 relative to actual -51, not initial target
                self.assertEqual(self.f.state, 'COARSE_RETURN_SETTLE')
                self.settle()
                self.assertEqual(self.f.state, 'COARSE_REACQUIRE')
                self.f._observe_pose = Mock(return_value=(self.pose, 'ok'))
                self.f._vision_values = Mock(return_value=(0., .3))
                self.f._stable_observation = Mock(return_value=(self.pose, 0., .3))
                self.tick()
                self.assertEqual(self.f.state, 'ACQUIRE_VERIFY')
                self.assertFalse(self.f._coarse_required(self.pose))

    def test_stale_sensor_stops(self):
        self.start()
        self.f.imu_source.snapshot.return_value = (0., self.now - 1., 0., None)
        self.f._coarse_step(self.now, [], False, None, None, None, None)
        self.assertEqual(self.f.state, 'FAILED')
        self.f._exec.assert_called_with('STOP')

    def test_wrong_direction_stops(self):
        self.start()
        self.tick(yaw=6.)
        self.assertEqual(self.f.state, 'FAILED')

    def test_command_lease_stops_independently(self):
        self.start()
        control._set_tx_state('rotate_left_slow')
        self.now += cfg.COARSE_COMMAND_LEASE_SEC + .01
        self.assertEqual(control._get_tx_state()[0], 'stop')
        self.tick(dt=0.)
        self.assertEqual(self.f.state, 'FAILED')

    def test_missing_sensor_stops(self):
        self.f._maybe_begin_coarse(self.pose, self.now, [])
        self.f.imu_source = None
        self.f._coarse_step(self.now, [], False, None, None, None, None)
        self.assertEqual(self.f.state, 'FAILED')

    def test_reacquisition_timeout(self):
        self.f._coarse_started = self.now
        self.f._set_state('COARSE_REACQUIRE', 1.)
        self.tick(1.1)
        self.assertEqual(self.f.state, 'FAILED')

    def test_gyro_integrates_all_samples_and_unwraps(self):
        stream = ImuVideoStream(None)
        for i in range(501):
            stream.feed_gyro(math.pi / 2., i*10., self.now)
        self.assertAlmostEqual(stream.snapshot()[0], 450.)
        stream.feed_gyro(0., 6000., self.now)
        self.assertIsNotNone(stream.snapshot()[3])

    def test_blind_states_do_not_require_vision(self):
        from calib.fsm_v4.coarse import COARSE_BLIND_STATES
        for state in COARSE_BLIND_STATES:
            self.f.state = state
            self.assertTrue(self.f.vision_independent)
        self.f.state = 'COARSE_REACQUIRE'
        self.assertFalse(self.f.vision_independent)

    def test_acquisition_routes_large_yaw_before_near_distance_shortcut(self):
        from calib.fsm_v4.pose import VisualPoseFilter
        f = self.f
        f.execu, f.status, f._rotation = Mock(), Mock(), Mock()
        f.reset()
        f.state = 'ACQUIRE_VERIFY'
        f._state_deadline_mono = self.now + 10.
        f._initial_visibility_complete = True
        pose = VisualPoseFilter().seed(40., 0., 3.5, self.now)
        f._observe_pose = Mock(return_value=(pose, 'ok'))
        f._vision_values = Mock(return_value=(0., .2))
        f._stable_observation = Mock(return_value=(pose, 0., .2))
        f._route_start_distance = Mock(return_value=True)
        f.step(True, 1.1, 3.5, 40., 0.)
        self.assertEqual(f.state, 'COARSE_PREPARE')
        f._route_start_distance.assert_not_called()

    def test_face_settle_enters_coarse(self):
        from calib.fsm_v4.pose import VisualPoseFilter
        f = self.f
        f.execu, f.status, f._rotation = Mock(), Mock(), Mock()
        f.reset()
        f.state = 'FACE_SETTLE'
        pose = VisualPoseFilter().seed(-40., 0., 3.5, self.now)
        f._observe_pose = Mock(return_value=(pose, 'ok'))
        f._vision_values = Mock(return_value=(0., .2))
        f._settle_observation = Mock(return_value=(pose, 0., .2))
        f._update_rotation_adaptation = Mock()
        f.step(True, 1.1, 3.5, -40., 0.)
        self.assertEqual(f.state, 'COARSE_PREPARE')

    def test_drive_heading_drift_stops(self):
        self.start()
        self.tick(yaw=-50.)
        self.settle()
        self.tick(yaw=-56.)
        self.assertEqual(self.f.state, 'FAILED')

    def test_first_turn_large_coast_stops_before_forward(self):
        self.start()
        self.tick(yaw=-58.)
        self.settle()
        self.assertEqual(self.f.state, 'FAILED')


if __name__ == '__main__':
    unittest.main()
