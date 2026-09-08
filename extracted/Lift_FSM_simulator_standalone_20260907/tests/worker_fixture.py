"""Controlled IPC faults and alternate lifecycle for process-boundary tests."""
import json
import os
from pathlib import Path
import sys
import time

mode=sys.argv[1] if len(sys.argv)>1 else 'alternate'
for line in sys.stdin:
    request=json.loads(line)
    op=request['op']
    if mode=='hang' and op=='tick':
        time.sleep(30)
    if mode=='crash' and op=='tick':
        os._exit(23)
    if mode=='malformed' and op=='tick':
        print('not json',flush=True)
        continue
    if op in ('init','info'):
        marker=Path(sys.argv[2]).read_text() if len(sys.argv)>2 else 'fixture'
        result=dict(pid=os.getpid(),info=dict(source_id=marker,config={},controller='fixture',files={}),
                    telemetry=dict(state='A_DIFFERENT_FINISHED_STATE'),lifecycle=dict(finished=False,outcome='running'))
    elif op=='tick':
        result=dict(current_can=[],future_can=[],advance_until=request['t'],model_updated=True,
                    telemetry=dict(state='A_DIFFERENT_FINISHED_STATE'),
                    lifecycle=dict(finished=True,outcome='success',reason=None))
    elif op=='stop':
        result=dict(can_frames=[])
    elif op=='close':
        break
    else:
        result={}
    response=dict(protocol='lift.worker.v1',request_id=request['request_id'],ok=True,result=result)
    if mode=='wrong_id' and op=='tick':
        response['request_id']+=1
    print(json.dumps(response),flush=True)
