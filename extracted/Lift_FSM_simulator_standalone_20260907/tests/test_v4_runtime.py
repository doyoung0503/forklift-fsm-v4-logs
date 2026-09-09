"""Environment fidelity, real-controller replay and failure classification tests."""
import contextlib
import copy
import io
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v4_runtime import (Session, Plant, scenario_options, replay_trace, environment,
                        fingerprint, cfg, control, top, command_status)
from protocol_world import World, can_frame, SPEC
from calibrate_lateral import placement


class RuntimeTests(unittest.TestCase):
    def run_case(self, options=None, overrides=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return Session({"perception":"oracle",**(options or {})}, overrides).run()

    def test_original_class_and_constructor_precheck_are_used(self):
        s=Session()
        self.assertIs(type(s.fsm),top.CalibrationFSMV4)
        self.assertEqual(s.initial["state"],"PRECHECK")
        self.assertEqual(s.step()["state"],"SEARCH_SWEEP")

    def test_original_replay_all_fields_and_detects_corrupted_can(self):
        r=self.run_case({"x":.3,"yaw":0})
        self.assertTrue(replay_trace(r["trace"])["passed"])
        bad=copy.deepcopy(r["trace"])
        bad[10]["can"]["data"][2]=67
        result=replay_trace(bad)
        self.assertEqual(len(result["mismatches"]),1)
        self.assertEqual(result["mismatches"][0]["frame"],10)

    def test_done_does_not_hide_residual_insertion(self):
        r=self.run_case({'drive_scale':.8})
        self.assertEqual(r["state"],"DONE")
        self.assertFalse(r["success"])
        self.assertGreater(r["true_remaining"],.05)
        self.assertIn("FSM DONE with insertion distance remaining",r["reasons"])

    def test_coast_assumption_can_complete_straight_insertion(self):
        r=self.run_case({"drive_coast_sec":.55})
        self.assertTrue(r["success"])
        self.assertIsNone(r["collision"])

    def test_fov_loss_runs_real_recovery_timeout(self):
        r=self.run_case({"perception":"fov","z":cfg.SAFETY_STANDOFF_Z_M+.5,
                         "loss_start":4,"loss_duration":5})
        self.assertEqual(r["state"],"FAILED")
        self.assertEqual(r["failure_reason"],"PnP recovery timeout")
        self.assertIn("STANDOFF_MOVE",[e["state"] for e in r["events"]])
        self.assertIn("RECOVER_VISUAL",[e["state"] for e in r["events"]])

    def test_uncommandable_alignment_fails_in_original_fsm(self):
        # This runner has no IMU; isolate the normal alignment rejection.
        r=self.run_case({"x":.3,"z":1.5,"yaw":-15}, {'COARSE_IMU_ENABLED':False})
        self.assertEqual(r["state"],"FAILED")
        self.assertIn("no commandable visibility-safe insertion rotation",r["failure_reason"])

    def test_far_start_executes_standoff_and_staging(self):
        r=self.run_case({"z":cfg.SAFETY_STANDOFF_Z_M+.5})
        states=[e["state"] for e in r["events"]]
        self.assertIn("STANDOFF_MOVE",states)
        self.assertIn("STAGING_PLAN",states)
        self.assertTrue(replay_trace(r["trace"])["passed"])

    def test_total_detection_loss_and_actuator_stall(self):
        lost=self.run_case({"dropout":1})
        self.assertEqual(lost["failure_reason"],"no pallet detected during 30 s right-rotation search")
        stalled=self.run_case({"drive_scale":0})
        self.assertEqual(stalled['state'],'DONE')
        self.assertFalse(stalled['success'])
        self.assertGreater(stalled['true_remaining'],1.)

    def test_rotation_endpoint_is_preserved_with_stop_coast(self):
        options=scenario_options({"rotation_coast_fraction":.3,"rotation_coast_sec":.6})
        world=World(options)
        p=world.plant
        for i in range(100):
            t=world.now
            world.advance(.02,[can_frame(0x2e3,[0x42,0,0,10,i%16,64,105,147],t),
                               can_frame(0x764,[0],t),
                               can_frame(0x1e3,[127,97,127,127,127,127,127,127],t)])
        before=p.heading
        world.advance(.6,[can_frame(0x1e3,[127]*8,world.now)])
        self.assertGreater(p.heading,before)
        self.assertAlmostEqual(p.heading,cfg.ROTATION_RESPONSE.total_angle(2.),places=8)

    def test_full_settle_guard_and_stability_window_are_not_skipped(self):
        r=self.run_case()
        events=r["events"]
        stop=next(e["t"] for e in events if e["state"]=="INSERT_SETTLE")
        done=next(e["t"] for e in events if e["state"]=="DONE")
        self.assertGreaterEqual(done-stop,cfg.STOP_MIN_SETTLE_SEC-1e-8)
        acquire=next(e['t'] for e in events if e['state']=='ACQUIRE_VERIFY')
        ready=next(e['t'] for e in events if e['state']=='READY_TO_INSERT')
        self.assertGreaterEqual(ready-acquire,cfg.STOP_MIN_SETTLE_SEC+cfg.STABLE_POSE_FRAMES/30-.04)

    def test_sensor_latency_uses_historical_positions(self):
        r=self.run_case({"latency":.2})
        moving=next(f for f in r["frames"] if f["command"]=="FWD" and f["truth"]["z"]<1.8)
        self.assertGreater(moving["observation"]["z"],moving["truth"]["z"])
        row=r['trace'][r['frames'].index(moving)]
        self.assertAlmostEqual(row['model_packet']['sim_time_s']-moving['observation']['measured'],.2,places=6)
        self.assertLess(moving['t']-row['model_packet']['sim_time_s'],1/30+1e-8)

    def test_seed_reproducibility_and_interleaved_sessions(self):
        options={"perception":"oracle","yaw_noise":.1,"position_noise":.001,"drive_scale_sd":.01}
        baseline=Session(options).run()
        a,b=Session(options),Session({"perception":"oracle","x":.3})
        with contextlib.redirect_stdout(io.StringIO()):
            while not a.done:
                a.step();b.step()
        self.assertEqual(a.report()["trace_hash"],baseline["trace_hash"])
        r=Session({**options,"seed":42}).run()
        self.assertNotEqual(r["trace_hash"],baseline["trace_hash"])

    def test_rotation_signs_and_camera_arc(self):
        p=Plant(scenario_options({"x":0,"z":2,"yaw":0}))
        p.move(10,0)
        self.assertAlmostEqual(p.yaw,-10)
        self.assertLess(p.x,0)
        self.assertAlmostEqual(math.hypot(p.x,p.z-cfg.CAMERA_TO_ROT_CENTER_Z_M),2-cfg.CAMERA_TO_ROT_CENTER_Z_M)
        p.move(-10,0)
        self.assertAlmostEqual(p.x,0)
        self.assertAlmostEqual(p.z,2)

    def test_removed_continuous_wall_monitor_cannot_reject_fsm_completion(self):
        # Previously failed only because the obsolete continuous wall monitor
        # disagreed with the FSM's nine-block insertion geometry.
        r=self.run_case(placement(3., .28125), {'COARSE_IMU_ENABLED':False,
                                             'FWD_PREDICTIVE_MAX_ADVANCE_M':0.})
        self.assertIsNone(r["collision"])
        self.assertEqual(r["state"],"DONE")
        self.assertTrue(r["success"])
        self.assertEqual(r["reasons"],[])

    def test_clock_config_and_source_are_restored(self):
        clock,status_clock=top.time,command_status.time
        original=cfg.FINAL_YAW_TOL_DEG
        with self.assertRaises(RuntimeError):
            with environment(3,{"FINAL_YAW_TOL_DEG":10}):
                self.assertEqual(top.time.monotonic(),1003)
                raise RuntimeError("test exit")
        self.assertIs(top.time,clock)
        self.assertIs(command_status.time,status_clock)
        self.assertEqual(cfg.FINAL_YAW_TOL_DEG,original)
        self.assertTrue(fingerprint()["disk_matches_loaded"])

    def test_coarse_disable_override_is_typed_and_restored(self):
        original = cfg.COARSE_IMU_ENABLED
        with self.assertRaisesRegex(RuntimeError, "exit"):
            with environment(0., {"COARSE_IMU_ENABLED": False}):
                self.assertIs(cfg.COARSE_IMU_ENABLED, False)
                raise RuntimeError("exit")
        self.assertIs(cfg.COARSE_IMU_ENABLED, original)
        for invalid in (0, 1, "false", None):
            with self.assertRaisesRegex(ValueError, "boolean"):
                with environment(0., {"COARSE_IMU_ENABLED": invalid}):
                    pass
            self.assertIs(cfg.COARSE_IMU_ENABLED, original)

    def test_zero_predictive_advance_override_is_restored(self):
        original = cfg.FWD_PREDICTIVE_MAX_ADVANCE_M
        with environment(0., {"FWD_PREDICTIVE_MAX_ADVANCE_M": 0.}):
            self.assertEqual(cfg.FWD_PREDICTIVE_MAX_ADVANCE_M, 0.)
        self.assertEqual(cfg.FWD_PREDICTIVE_MAX_ADVANCE_M, original)
        with self.assertRaises(ValueError):
            with environment(0., {"FWD_PREDICTIVE_MAX_ADVANCE_M": -.01}):
                pass
        self.assertEqual(cfg.FWD_PREDICTIVE_MAX_ADVANCE_M, original)

    def test_virtual_can_writes_without_physical_initialization(self):
        with patch.object(control,"can_init",side_effect=AssertionError("physical CAN init")), \
             patch.object(control,"_can_tx_worker",side_effect=AssertionError("physical CAN worker")), \
             patch.object(control,"_write",wraps=control._write) as writes:
            report=self.run_case()
        self.assertGreater(writes.call_count,100)
        self.assertEqual(writes.call_count,len(report["can_frames"]))
        self.assertFalse(control._CAN_ENABLED)
        self.assertIsNone(control._CAN_THREAD)

    def test_invalid_inputs_and_config_do_not_change_source(self):
        for options in ({"hz":0},{"latency":float("nan")},{"perception":"bad"},{"unknown":0}):
            with self.assertRaises(ValueError):
                Session(options)
        for overrides in ({"CAN_ENABLED":True},{"STABLE_POSE_FRAMES":2.5},{"IMAGE_EDGE_MARGIN_NORM":.8}):
            with self.assertRaises(ValueError):
                Session(overrides=overrides)


if __name__=="__main__":
    unittest.main()
