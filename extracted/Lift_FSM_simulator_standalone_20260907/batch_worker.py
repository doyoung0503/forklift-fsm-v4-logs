"""Batch transport around the unchanged Session.step feedback loop."""
import json
import time
import sys
import tempfile
import traceback
import queue
import threading
from pathlib import Path
from process_runtime import Session
from batched_session import LocalController


def main():
    wire=sys.stdout;sys.stdout=sys.stderr
    session=None
    inbox=queue.Queue()
    reader_started=False
    def read_requests():
        try:
            for line in sys.stdin:inbox.put(line)
        finally:inbox.put(None)
    with tempfile.TemporaryDirectory(prefix='lift_batch_') as directory:
        try:
            while True:
                if session and session.controller.window:
                    pump_start=time.perf_counter()
                    pumped=session.controller.window.pump()
                    perf=session.controller.phase_perf
                    perf['outer_pump_executed']=perf.get('outer_pump_executed',0)+int(bool(pumped))
                    perf['outer_pump_ms']=perf.get('outer_pump_ms',0.)+(time.perf_counter()-pump_start)*1000
                    perf['outer_pump_calls']=perf.get('outer_pump_calls',0)+1
                if reader_started:
                    try:line=inbox.get(timeout=.02)
                    except queue.Empty:continue
                else:line=sys.stdin.readline() or None
                if line is None:break
                request=json.loads(line)
                try:
                    operation_start=time.perf_counter()
                    op=request['op']
                    if op=='close':break
                    if op=='create':
                        session=Session(request.get('options'),request.get('overrides'),capture=request.get('capture',True),
                            log_root=Path(request['log_root']),show_window=request.get('show_window',False),controller_factory=LocalController)
                        if request.get('show_window',False):
                            import cv2  # Load native DLLs before a pending stdin read.
                            threading.Thread(target=read_requests,daemon=True).start()
                            reader_started=True
                        start=0
                    elif op=='advance':
                        start=len(session.frames)
                        for _ in range(min(120,max(1,int(request.get('ticks',1))))):
                            if session.done:break
                            session.step()  # CAN -> world -> fresh sensors -> next FSM tick
                            if session.now>=request.get('target',float('inf')):break
                    elif op=='present':
                        result=session.present(request['frame'],request['presentation_id'],request.get('jpeg'),request.get('options'))
                    elif op=='report':
                        path=Path(directory)/'report.json'
                        path.write_text(json.dumps(session.report(),ensure_ascii=False),encoding='utf8')
                        result=dict(path=str(path))
                    else:raise ValueError(op)
                    if op in ('create','advance'):
                        result=dict(frames=session.frames[start:],now=session.now,done=session.done,status=session.status,
                            controller_wall_ms=session.controller_wall_ms,log_directory=session.logging_info()['directory'])
                        result['phase_perf']={**getattr(session,'phase_perf',{}),**session.controller.phase_perf}
                        result['worker_operation_ms']=(time.perf_counter()-operation_start)*1000
                        if op=='create':
                            result.update(initial=session.initial,provenance=session.provenance)
                    response=dict(protocol='lift.worker.v1',request_id=request['request_id'],ok=True,result=result)
                except Exception as error:
                    traceback.print_exc()
                    response=dict(protocol='lift.worker.v1',request_id=request['request_id'],ok=False,error=str(error))
                wire.write(json.dumps(response,ensure_ascii=False,allow_nan=False)+'\n');wire.flush()
        finally:
            if session:session.close()


if __name__=='__main__':main()
