import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.request import Request,urlopen
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import serve_v4 as api
from process_runtime import Session


class BackgroundClockTest(unittest.TestCase):
    def test_runs_without_browser_polls_and_pauses(self):
        with tempfile.TemporaryDirectory() as directory:
            session=Session(dict(max_seconds=3),log_root=Path(directory))
            with api.LOCK: api.SESSIONS['test']=session
            server=ThreadingHTTPServer(('127.0.0.1',0),api.Handler)
            threading.Thread(target=server.serve_forever,daemon=True).start()
            def post(route,**data):
                return json.load(urlopen(Request(f'http://127.0.0.1:{server.server_port}/api/{route}',
                    data=json.dumps(dict(id='test',**data)).encode(),headers={'Content-Type':'application/json'})))
            try:
                post('play')
                time.sleep(.6)  # No browser poll/render/ACK requests.
                post('pause')
                first=post('poll',cursor=0)
                self.assertGreater(session.now,.1)
                stopped=session.now
                time.sleep(.2)
                self.assertEqual(session.now,stopped)
                self.assertEqual(post('poll',cursor=first['cursor'])['frames'],[])
                post('play',rate=8)
                deadline=time.monotonic()+8
                while not session.done and time.monotonic()<deadline: time.sleep(.1)
                self.assertTrue(session.done)
                last=post('poll',cursor=first['cursor'])
                self.assertTrue(last['done'])
                self.assertTrue((session.logger.directory/'run_server_performance.jsonl').exists())
            finally:
                with api.LOCK:
                    api.SESSIONS.pop('test',None);session.close()
                server.shutdown();server.server_close()
