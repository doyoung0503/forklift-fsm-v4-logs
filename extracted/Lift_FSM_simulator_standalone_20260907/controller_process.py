"""FSM-agnostic, bounded JSON IPC and subprocess lifecycle management."""
import atexit
import collections
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tempfile
import weakref

HERE=Path(__file__).resolve().parent
PROTOCOL='lift.worker.v1'
ACTIVE=weakref.WeakSet()


class ControllerProcessError(RuntimeError):
    pass


class ControllerProcess:
    def __init__(self,*,command=None,timeout=10.):
        self.timeout=timeout
        self.lock=threading.RLock()
        self.responses=queue.Queue()
        self.stderr=collections.deque(maxlen=60)
        self.closed=False
        self.serial=0
        self.cache=tempfile.TemporaryDirectory(prefix='lift_fsm_cache_')
        env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1',
             'PYTHONPYCACHEPREFIX':self.cache.name}
        try:
            self.process=subprocess.Popen(command or [sys.executable,'-u','-B',
                str(HERE.parent/'depth_cam/main_rec_v4.py'),'--mode','simulation','--ipc'],
                cwd=HERE,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                text=True,encoding='utf-8',errors='replace',bufsize=1,env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except Exception:
            self.cache.cleanup()
            raise
        self.pid=self.process.pid
        self.reader=threading.Thread(target=self._read,daemon=True)
        self.log_reader=threading.Thread(target=self._logs,daemon=True)
        self.reader.start();self.log_reader.start()
        ACTIVE.add(self)

    def _read(self):
        try:
            while True:
                line=self.process.stdout.readline(4_000_001)
                if not line:
                    break
                if len(line)>4_000_000:
                    self.responses.put(ControllerProcessError('Worker response exceeds 4 MB'))
                    break
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    self.responses.put(ControllerProcessError('Worker emitted invalid JSON'))
                    break
        finally:
            self.responses.put(ControllerProcessError('Controller process disconnected'))

    def _logs(self):
        for line in self.process.stderr:
            self.stderr.append(line.rstrip()[-2000:])

    def request(self,op,**data):
        with self.lock:
            if self.closed:
                raise ControllerProcessError('Controller process is closed')
            self.serial+=1
            if op=='init':
                data.setdefault('show_window',False)
            try:
                self.process.stdin.write(json.dumps(dict(protocol=PROTOCOL,request_id=self.serial,op=op,**data),
                                                   ensure_ascii=False,allow_nan=False)+'\n')
                self.process.stdin.flush()
                response=self.responses.get(timeout=self.timeout)
                if isinstance(response,Exception):
                    raise response
                if not isinstance(response,dict) or response.get('protocol')!=PROTOCOL or response.get('request_id')!=self.serial:
                    raise ControllerProcessError('Worker response protocol or request ID mismatch')
                if not response.get('ok'):
                    raise ControllerProcessError(response.get('error','Worker rejected request'))
                return response['result']
            except (OSError,queue.Empty,KeyError,ValueError,ControllerProcessError) as exc:
                message='Controller response timeout' if isinstance(exc,queue.Empty) else str(exc)
                self.close(graceful=False)
                if self.stderr:
                    message+='\n'+'\n'.join(list(self.stderr)[-8:])
                raise ControllerProcessError(message) from exc

    def close(self,graceful=True):
        with self.lock:
            if self.closed:
                return
            self.closed=True
            try:
                if graceful and self.process.poll() is None:
                    self.serial+=1
                    self.process.stdin.write(json.dumps(dict(protocol=PROTOCOL,request_id=self.serial,op='close'))+'\n')
                    self.process.stdin.flush()
                self.process.stdin.close()
                self.process.wait(timeout=1.)
            except (OSError,subprocess.TimeoutExpired):
                if self.process.poll() is None:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    self.process.kill();self.process.wait(timeout=1.)
            finally:
                self.reader.join(timeout=1.);self.log_reader.join(timeout=1.)
                self.process.stdout.close();self.process.stderr.close()
                self.cache.cleanup()
                ACTIVE.discard(self)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()


@atexit.register
def close_controllers():
    for controller in list(ACTIVE):
        controller.close()
