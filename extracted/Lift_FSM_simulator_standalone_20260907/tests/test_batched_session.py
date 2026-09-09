import sys
import tempfile
import time
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from process_runtime import Session
from batched_session import BatchedSession


class BatchParity(unittest.TestCase):
    def test_full_noisy_coarse_and_insertion_matches_every_tick(self):
        options=dict(x=0,z=4,yaw=15,latency=0,latency_sd=0,dropout=0,
            vertical_offset=.425,face_selection='fixed',drive_scale_sd=.188,
            rotation_scale_sd=.05,imu_scale=95/90,seed=20260908)
        with tempfile.TemporaryDirectory() as directory:
            start=time.perf_counter()
            old=Session(options,log_root=Path(directory)/'old').run()
            old_seconds=time.perf_counter()-start
            start=time.perf_counter();new_session=BatchedSession(options,log_root=Path(directory)/'batch')
            batches=0
            try:
                while not new_session.done:
                    new_session.step_batch(new_session.now+.1);batches+=1
                new=new_session.report()
                perf=new_session.phase_perf
                self.assertEqual(perf['ticks'],len(old['trace']))
                self.assertEqual(perf['fsm_calls'],perf['ticks'])
                self.assertGreater(perf['logging_ms'],0)
                self.assertGreaterEqual(perf['controller_ms'],perf['fsm_ms'])
                self.assertEqual(new_session.transport_perf['requests'],batches)
                for key in ('trace','can_frames','final','events','state','success'):
                    self.assertEqual(old[key],new[key],key)
                self.assertLess(batches,len(old['trace'])/2)
                print(f'Parity: {len(old["trace"])} ticks / {batches} batches; old {old_seconds:.2f}s, batch {time.perf_counter()-start:.2f}s')
            finally:new_session.close()
