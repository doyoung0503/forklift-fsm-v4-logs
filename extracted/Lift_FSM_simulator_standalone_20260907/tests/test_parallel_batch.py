import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parallel_batch import BatchJobs

class ParallelBatchTests(unittest.TestCase):
    def finish(self,jobs,ident):
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            s=jobs.status(ident)
            if s['status']!='running':return s
            time.sleep(.1)
        self.fail('batch timeout')

    def test_serial_parallel_reproducible_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs=BatchJobs(directory)
            cases=[dict(options=dict(x=0,z=4,yaw=15,seed=s,max_seconds=185,latency=0,dropout=0,drive_scale_sd=.188,imu_scale=95/90)) for s in [11,12]]
            serial=self.finish(jobs,jobs.start(dict(cases=cases,workers=1))['id'])
            parallel=self.finish(jobs,jobs.start(dict(cases=cases,workers=min(2,os.cpu_count() or 1)))['id'])
            self.assertEqual(parallel['status'],'completed')
            self.assertEqual(parallel['completed'],2)
            a=sorted(serial['results'],key=lambda r:r['case_id']);b=sorted(parallel['results'],key=lambda r:r['case_id'])
            for x,y in zip(a,b):
                self.assertEqual(x['trace_hash'],y['trace_hash'])
                self.assertEqual(x['final'],y['final'])
                self.assertEqual(y['logging']['status'],'disabled')
                self.assertNotIn('frames',y)
            self.assertEqual(len(set(r['controller_pid'] for r in b)),2)
            self.assertEqual(parallel['groups'][0]['completed'],2)
            self.assertEqual(jobs.status(parallel['id'],2)['results'],[])
            rows=Path(parallel['directory'],'results.jsonl').read_text(encoding='utf8').splitlines()
            self.assertEqual(len(rows),2)
            print('Serial seconds',serial['seconds'],'parallel seconds',parallel['seconds'],flush=True)

    def test_cancel_keeps_finished_cases_and_invalid_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs=BatchJobs(directory)
            with self.assertRaises(ValueError):jobs.start(dict(cases=[],workers=1))
            with self.assertRaises(ValueError):jobs.start(dict(cases=[dict(options={})],workers=0))
            c=dict(options=dict(max_seconds=1))
            job=jobs.start(dict(cases=[c]*20,workers=1))
            jobs.request('/api/batch/cancel',dict(id=job['id']))
            result=self.finish(jobs,job['id'])
            self.assertEqual(result['status'],'cancelled')
            self.assertLess(result['completed'],20)
