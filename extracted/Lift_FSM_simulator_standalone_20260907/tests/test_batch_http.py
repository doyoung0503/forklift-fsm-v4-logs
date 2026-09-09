import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request,urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import serve_v4
from parallel_batch import BatchJobs

class BatchHTTP(unittest.TestCase):
    def test_batch_routes_without_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            previous=serve_v4.JOBS;serve_v4.JOBS=BatchJobs(directory)
            server=ThreadingHTTPServer(('127.0.0.1',0),serve_v4.Handler)
            threading.Thread(target=server.serve_forever,daemon=True).start()
            def post(path,data):
                request=Request(f'http://127.0.0.1:{server.server_port}/api/batch/{path}',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
                with urlopen(request,timeout=10) as response:return json.load(response)
            try:
                job=post('start',dict(cases=[dict(options=dict(max_seconds=.1,seed=3))],workers=1))
                deadline=time.monotonic()+30
                while time.monotonic()<deadline:
                    result=post('status',dict(id=job['id']))
                    if result['status']!='running':break
                    time.sleep(.1)
                self.assertEqual(result['status'],'completed')
                self.assertEqual(result['completed'],1)
                self.assertEqual(result['results'][0]['options']['seed'],3)
                self.assertTrue(Path(job['directory'],'results.jsonl').exists())
            finally:
                server.shutdown();server.server_close();serve_v4.JOBS=previous
