"""Real subprocess separation, deterministic CAN parity and failure lifecycle."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from controller_process import ControllerProcess,ControllerProcessError
from process_runtime import Session,replay_trace
from protocol_world import can_frame


class ProcessTests(unittest.TestCase):
    def fixture(self,mode='alternate',*args,timeout=2):
        return ControllerProcess(command=[sys.executable,'-u',str(ROOT/'tests/worker_fixture.py'),mode,*map(str,args)],timeout=timeout)

    def test_server_and_client_do_not_import_fsm_in_parent(self):
        code="import serve_v4,protocol_client,sys; from process_runtime import Session; s=Session(); assert s.controller.pid!=__import__('os').getpid(); assert not any(k.split('.')[0] in ('calib','torch','ultralytics','pyrealsense2') for k in sys.modules); s.close()"
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,capture_output=True)

    def test_actual_subprocess_can_and_states_match_reference(self):
        from v4_runtime import Session as Reference
        options=dict(inference_mode='continuous',perception='oracle',model_hz=13,model_interval_sd=.01,latency_sd=.02,x_noise=.005)
        with contextlib.redirect_stdout(io.StringIO()):
            old=Reference(options).run()
        session=Session(options)
        result=session.run()
        self.assertNotEqual(session.controller.pid,os.getpid())
        self.assertEqual(result['can_frames'],old['can_frames'])
        self.assertEqual([(r['state'],r['command']) for r in result['trace']],
                         [(r['state'],r['command']) for r in old['trace']])
        self.assertEqual(result['final'],old['final'])
        self.assertTrue(replay_trace(result['trace'],source_id=result['source_id'])['passed'])
        self.assertEqual(session.controller.process.poll(),0)
        self.assertEqual(result['provenance']['loaded_camera_inference_modules'],[])

    def test_source_diagram_is_used_without_camera_or_inference(self):
        session=Session()
        try:
            tree=ast.parse((ROOT.parent/'depth_cam/ui/diagram_v4.py').read_text(encoding='utf-8-sig'))
            flow=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='V4_FLOW' for t in n.targets))
            self.assertEqual(session.provenance['diagram']['phases'],
                             [dict(label=label,states=sorted(states)) for label,states in flow])
            frame=session.step()
            while not session.inputs[-1]['model_updated']:
                frame=session.step()
            self.assertEqual(frame['diagram']['phase_index'],0)
            self.assertEqual(frame['diagram']['activity'],'Searching for pallet')
        finally:
            session.close()

    def test_sessions_have_separate_processes_and_no_global_state_bleed(self):
        a,b=Session(),Session(overrides={'FINAL_YAW_TOL_DEG':10})
        try:
            self.assertNotEqual(a.controller.pid,b.controller.pid)
            self.assertEqual(b.config['FINAL_YAW_TOL_DEG'],10)
            self.assertNotEqual(a.config['FINAL_YAW_TOL_DEG'],10)
            a.step();b.step();a.step()
            self.assertGreater(a.now,b.now)
        finally:
            a.close();b.close()
        self.assertIsNotNone(a.controller.process.poll())
        self.assertIsNotNone(b.controller.process.poll())

    def test_lifecycle_does_not_require_done_state_name_or_private_fields(self):
        s=Session(controller_factory=self.fixture)
        r=s.run()
        self.assertTrue(r['finished'])
        self.assertEqual(r['lifecycle']['outcome'],'success')
        self.assertEqual(r['state'],'A_DIFFERENT_FINISHED_STATE')
        self.assertIsNone(r['fsm_remaining'])

    def test_crash_hang_and_malformed_responses_are_bounded_and_reaped(self):
        for mode in ('crash','hang','malformed','wrong_id'):
            with self.subTest(mode=mode):
                s=Session(controller_factory=lambda:self.fixture(mode,timeout=.5))
                start=time.monotonic()
                r=s.run()
                self.assertLess(time.monotonic()-start,4)
                self.assertEqual(r['lifecycle']['outcome'],'error')
                self.assertFalse(r['replayable'])
                self.assertIsNotNone(s.controller.process.poll())

    def test_lost_process_stops_by_can_watchdog_without_fabricated_stop(self):
        s=Session(dict(drive_coast_sec=.1))
        try:
            for _ in range(100):
                t=s.world.now
                s.world.advance(.02,[can_frame(0x2e3,[66,0,0,10,0,64,105,147],t),
                    can_frame(0x764,[0],t),can_frame(0x1e3,[127,127,67,127,127,127,127,127],t)])
            s.now=s.world.now
            count=len(s.world.can_log)
            s.controller.process.kill();s.controller.process.wait()
            frame=s.step()
            self.assertTrue(frame['terminal'])
            self.assertEqual(len(s.world.can_log),count)
            self.assertEqual(s.plant.drive,0)
            self.assertNotEqual(frame['bus_status'],'ready')
            self.assertEqual(frame['can']['data'][2],67)  # Last received frame, not a fake STOP.
        finally:
            s.close()

    def test_time_limit_sends_real_stop_and_is_replayable(self):
        s=Session(dict(perception='oracle',max_seconds=4))
        r=s.run()
        self.assertEqual(r['lifecycle']['outcome'],'timeout')
        self.assertEqual(s.plant.movement,[127]*8)
        self.assertEqual(s.plant.drive,0)
        self.assertTrue(replay_trace(r['trace'])['passed'])

    def test_vision_independent_insertion_ticks_without_new_model_results(self):
        r=Session(dict(perception='oracle',model_hz=5)).run()
        rows=[row for row in r['trace'] if row['state']=='INSERT_DRIVE']
        self.assertTrue(any(row['fsm_updated'] and not row['model_updated'] for row in rows))
        self.assertTrue(all(row['inputs']['offset_smooth'] is None for row in rows))
        self.assertTrue(all(row['advance_until']-row['t']<=.01000001 for row in rows))
        self.assertEqual(r['state'],'DONE')

    def test_bad_init_and_replay_source_mismatch_close_workers(self):
        with self.assertRaises(ControllerProcessError):
            Session(overrides={'CAN_ENABLED':True})
        r=Session(dict(max_seconds=1)).run()
        with self.assertRaises(ValueError):
            replay_trace(r['trace'],source_id='wrong')

    def test_new_process_loads_saved_code_and_existing_process_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'reload_adapter.py'
            code="class Driver:\n @staticmethod\n def info(): return dict(source_id='version A',config={},files={})\n def __init__(self,overrides=None): pass\n def telemetry(self): return {}\n def lifecycle(self): return {}\n"
            source.write_text(code)
            stamp=source.stat().st_mtime
            command=[sys.executable,'-u','-c',
                     "import sys; sys.path.insert(0,sys.argv.pop(1)); import fsm_worker; fsm_worker.main()",
                     directory,'--adapter','reload_adapter:Driver']
            a=ControllerProcess(command=command)
            try:
                self.assertEqual(a.request('init')['info']['source_id'],'version A')
                source.write_text(code.replace('version A','version B'))
                os.utime(source,(stamp,stamp))  # Same size/mtime must not reuse stale pyc.
                with ControllerProcess(command=command) as b:
                    self.assertEqual(b.request('init')['info']['source_id'],'version B')
                    self.assertEqual(a.request('info')['info']['source_id'],'version A')
            finally:
                a.close()


if __name__=='__main__':
    unittest.main()
