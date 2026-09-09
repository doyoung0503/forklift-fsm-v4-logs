"""Exact virtual CAN deadlines, independent of FSM/perception updates."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4_runtime import control, cfg
from current_can_adapter import CurrentCanTransport
from protocol_world import EPOCH


class TimedRotationTransportTests(unittest.TestCase):
    def setUp(self):
        self.bus = CurrentCanTransport(control)
        with self.bus.bind(0.):
            control.configure_motion_deadline(None)
            self.bus.pump(.5)

    def start(self, hold, first_write=.7):
        with self.bus.bind(.5):
            control.arm_timed_rotation('ROT_RIGHT', hold)
            control.issue_command_rotate_in_place(-1)
            self.assertIsNone(control.timed_rotation_deadline())
        with self.bus.bind(first_write):
            frames=self.bus.pump(first_write)
            status=control.timed_rotation_status()
        self.assertAlmostEqual(status['started'], EPOCH+first_write)
        return frames

    def test_deadline_uses_first_write_and_stops_without_fsm_ticks(self):
        hold=cfg.ROTATION_RESPONSE.command_seconds(.3)
        self.start(hold)
        with self.bus.bind(.7):
            frames=self.bus.pump(3.)  # No FSM/vision call throughout this interval.
            status=control.timed_rotation_status()
        stops=[f for f in frames if f['id']==0x1e3 and f['data']==[127]*8]
        self.assertTrue(stops)
        self.assertAlmostEqual(stops[0]['t'], .7+hold, places=8)
        self.assertAlmostEqual(status['stopped']-status['started'],hold,places=8)
        self.assertTrue(status['expired'])

    def test_repeated_rotation_cannot_resurrect_expired_deadline(self):
        self.start(.1)
        with self.bus.bind(.7):
            self.bus.pump(1.)
            control.issue_command_rotate_in_place(-1)
            frames=self.bus.pump(1.1)
            self.assertEqual(control._get_tx_state()[0], 'stop')
        self.assertTrue(all(f['data']==[127]*8 for f in frames if f['id']==0x1e3))

    def test_small_angles_have_distinct_holds_without_vision_quantization(self):
        response = cfg.ROTATION_RESPONSE
        self.start(response.command_seconds(.3))
        with self.bus.bind(.7):
            self.bus.pump(3.)
            first = control.timed_rotation_status()
            control.issue_command_stop()
            self.bus.pump(3.)
            control.arm_timed_rotation('ROT_LEFT', response.command_seconds(.5))
            control.issue_command_rotate_in_place(1)
            self.bus.pump(5.)
            second = control.timed_rotation_status()
        for timer, angle in ((first, .3), (second, .5)):
            actual = response.slope_deg_s * (
                timer['stopped'] - timer['started'] - response.startup_delay_sec)
            self.assertAlmostEqual(actual, angle, places=7)

    def test_early_stop_cancels_timer_and_next_motion_runs(self):
        self.start(2.)
        with self.bus.bind(.8):
            control.issue_command_stop()
            self.bus.pump(.8)
            self.assertIsNone(control.timed_rotation_deadline())
            control.issue_command_forward()
            self.bus.pump(3.)
            self.assertEqual(control._get_tx_state()[0], 'forward')

    def test_new_rotation_has_its_own_clock(self):
        self.start(.1)
        with self.bus.bind(.7):
            self.bus.pump(1.)
            control.issue_command_stop()
            self.bus.pump(1.)
            control.arm_timed_rotation('ROT_LEFT', .2)
            control.issue_command_rotate_in_place(1)
            self.bus.pump(1.5)
            status=control.timed_rotation_status()
        self.assertAlmostEqual(status['started'],EPOCH+1.)
        self.assertAlmostEqual(status['stopped'],EPOCH+1.2)


if __name__=='__main__':
    unittest.main()
