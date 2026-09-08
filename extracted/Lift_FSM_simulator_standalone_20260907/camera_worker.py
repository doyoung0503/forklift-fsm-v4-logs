"""Single-owner OpenGL process; reuses the bounded JSON worker transport."""
import base64
import json
import sys
import traceback


def main():
    wire=sys.stdout;sys.stdout=sys.stderr
    from simulation_camera import SimulationCamera
    import cv2
    renderer=None
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line)
            if request.get('protocol')!='lift.worker.v1': raise ValueError('Invalid protocol')
            if request['op']=='close': break
            if request['op']!='render': raise ValueError('Unknown render operation')
            if renderer is None: renderer=SimulationCamera()
            image,meta=renderer.render(request['frame'],request['options'],request.get('config'))
            ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,88])
            if not ok: raise RuntimeError('JPEG encoding failed')
            result=dict(**meta,jpeg=base64.b64encode(encoded).decode('ascii'))
            response=dict(ok=True,result=result)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
        wire.write(json.dumps(dict(protocol='lift.worker.v1',request_id=request.get('request_id'),**response))+'\n')
        wire.flush()


if __name__=='__main__': main()
