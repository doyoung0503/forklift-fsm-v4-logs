"""JSON-lines controller worker. stdout is reserved for lift.worker.v1 IPC."""
import argparse
import importlib
import json
import math
import os
import queue
import sys
import threading
import traceback

PROTOCOL='lift.worker.v1'


def main(argv=None,*,default_window=False,entrypoint='fsm_worker.py'):
    parser=argparse.ArgumentParser()
    parser.add_argument('--adapter',default='current_fsm_driver:CurrentFSMDriver')
    args=parser.parse_args(argv)
    wire=sys.stdout
    sys.stdout=sys.stderr  # Imports, original FSM prints and tracebacks cannot corrupt IPC.
    driver=None
    factory=None
    now=0.
    window=None
    show_window=False
    sync_display=False
    last_packet=None
    last_lifecycle=dict(finished=False,outcome='running',reason=None)
    inbox=queue.Queue()
    reader_started=False
    def read_requests():
        try:
            for line in sys.stdin:
                inbox.put(line)
        finally:
            inbox.put(None)
    try:
        module,name=args.adapter.split(':',1)
        factory=getattr(importlib.import_module(module),name)
        while True:
            if window is not None:
                window.pump()
                if window.closed and last_lifecycle.get('finished'):
                    break
            if reader_started:
                try:
                    line=inbox.get(timeout=.02)
                except queue.Empty:
                    continue
            else:
                line=sys.stdin.readline() or None
            if line is None:
                break
            request={}
            closing=False
            try:
                request=json.loads(line)
                if request.get('protocol')!=PROTOCOL:
                    raise ValueError('Unsupported worker protocol')
                op=request['op']
                if op=='info':
                    result=dict(info=factory.info(),pid=os.getpid())
                elif op=='init':
                    if driver is not None:
                        raise ValueError('Controller already initialised')
                    driver=factory(request.get('overrides'))
                    show_window=bool(request.get('show_window',default_window))
                    sync_display=bool(request.get('sync_display',False))
                    if show_window:
                        # Windows NumPy DLL startup can block behind a pending
                        # CRT stdin read. Load the native UI before that thread.
                        import cv2
                        threading.Thread(target=read_requests,daemon=True).start()
                        reader_started=True
                    result=dict(info=factory.info(),pid=os.getpid(),telemetry=driver.telemetry(),lifecycle=driver.lifecycle())
                elif op=='tick':
                    t,until=float(request['t']),float(request['until'])
                    if not math.isfinite(t) or not math.isfinite(until) or abs(t-now)>1e-8 or not t<=until<=t+1.00000001:
                        raise ValueError('Invalid or non-contiguous virtual time')
                    if show_window and not sync_display and window is None:
                        from fsm_window import FSMWindow
                        window=FSMWindow()
                    last_packet=request.get('model') or last_packet
                    if request.get('cancel',False) or (window is not None and window.closed):
                        stopped=driver.stop(t)
                        result=dict(current_can=stopped['can_frames'],future_can=[],advance_until=t,
                            model_updated=False,fsm_updated=False,inputs=None,
                            telemetry={**driver.telemetry(),'command':'STOP'},
                            lifecycle=dict(finished=True,outcome='cancelled',reason='simulation window closed'),cancel_requested=True)
                    else:
                        result=driver.tick(t,until,request.get('model'),bool(request.get('halt',False)))
                    now=result['advance_until']
                    last_lifecycle=result['lifecycle']
                    if window is not None and not sync_display:
                        window.render(driver,last_packet,t,last_lifecycle,result['current_can']+result['future_can'],request.get('scene'))
                        window.pump()
                    result['telemetry']['runtime_mode']='simulation'
                    result['telemetry']['window_open']=window is not None and not window.closed
                elif op=='present':
                    frame=request['frame']
                    if not isinstance(frame,dict) or not math.isfinite(float(frame['t'])):
                        raise ValueError('Invalid presentation frame')
                    presentation_error=None
                    try:
                        if show_window and window is None:
                            from fsm_window import FSMWindow
                            window=FSMWindow()
                        if window is not None and not window.closed:
                            window.render_snapshot(frame,request['options'],request['config'],driver.overrides,request.get('jpeg'))
                    except Exception as error:
                        # A display failure must not tear down the controller or
                        # produce an extra STOP/CAN frame through IPC error handling.
                        presentation_error=f'{type(error).__name__}: {error}'
                        if window is not None:
                            window.last_render_args=None
                    result=dict(presentation_id=request['presentation_id'],t=frame['t'],
                        model_sequence=(frame.get('model_packet') or {}).get('sequence'),
                        window_open=window is not None and not window.closed)
                    if presentation_error:
                        result['error']=presentation_error
                elif op=='stop':
                    t=float(request['t'])
                    if not math.isfinite(t) or abs(t-now)>1e-8:
                        raise ValueError('Invalid stop time')
                    result=driver.stop(t)
                elif op=='close':
                    result={}
                    closing=True
                else:
                    raise ValueError('Unknown worker operation')
                if op in ('info','init'):
                    result['info'].update(runtime_mode='simulation',entrypoint=entrypoint,
                        window_requested=show_window)
                    result['info']['loaded_camera_inference_modules']=[name for name in sys.modules if
                        name.split('.')[0] in ('torch','ultralytics','pyrealsense2','main_rec')]
                response=dict(protocol=PROTOCOL,request_id=request['request_id'],ok=True,result=result)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                response=dict(protocol=PROTOCOL,request_id=request.get('request_id'),ok=False,
                              error=f'{type(exc).__name__}: {exc}')
            wire.write(json.dumps(response,ensure_ascii=False,allow_nan=False)+'\n')
            wire.flush()
            if closing:
                break
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if window is not None:
            window.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
