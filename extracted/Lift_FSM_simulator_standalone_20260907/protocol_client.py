"""Example clients for the public CAN/model protocol, including the current FSM.

python protocol_client.py --controller demo --url http://127.0.0.1:8767
python protocol_client.py --controller current --url http://127.0.0.1:8767
"""
import argparse
import contextlib
import io
import json
import time
import urllib.request

from protocol_world import can_frame,model_step


class Client:
    def __init__(self,url):
        self.url=url.rstrip("/")

    def post(self,route,**data):
        request=urllib.request.Request(self.url+route,json.dumps(data).encode(),
                                      {"Content-Type":"application/json"})
        with urllib.request.urlopen(request,timeout=30) as response:
            return json.load(response)


class DemoController:
    """A different controller with no knowledge of CalibrationFSMV4 internals."""
    def __init__(self):
        self.sync=0

    def tick(self,packet,now,until):
        step=model_step(packet)
        moving=step["det_ok"] and step["dist_z"]>1.7
        frames=[]
        # 5 ms control, 10 ms movement, 200 ms heartbeat.
        first=round(now/.005)
        for index in range(first,int(until/.005)+1):
            t=index*.005
            if t<now-1e-9 or t>=until-1e-9:
                continue
            self.sync=(self.sync+1)&15
            frames.append(can_frame(0x2e3,[0x42,0,0,0x0a,self.sync,0x40,0x69,0x93],t))
            if index%40==0:
                frames.append(can_frame(0x764,[0],t))
            if index%2==0:
                frames.append(can_frame(0x1e3,[127,127,67 if moving else 127,127,127,127,127,127],t))
        return frames


class CurrentController:
    """Current FSM in its own child process; this client knows only worker IPC."""
    def __init__(self):
        from controller_process import ControllerProcess
        self.controller=ControllerProcess()
        try:
            result=self.controller.request('init')
            self.telemetry=result.get('telemetry',{})
        except Exception:
            self.controller.close()
            raise
        self.finished=False
        self.advance_until=0.

    def tick(self,packet,now,until):
        result=self.controller.request('tick',t=now,until=until,model=packet)
        self.finished=result.get('lifecycle',{}).get('finished',False)
        self.telemetry=result.get('telemetry',{})
        self.advance_until=result['advance_until']
        return result['current_can']+result['future_can']

    def close(self):
        self.controller.close()

    def stop(self,now):
        return self.controller.request('stop',t=now)['can_frames']


def run(url,controller="demo",seconds=15.,options=None,realtime=False,on_create=None):
    client=Client(url)
    response=client.post("/api/world/create",options=options or {"perception":"oracle"})
    sid,packet=response["id"],response["model"]
    if on_create:
        on_create(sid)
    driver=CurrentController() if controller=="current" else DemoController()
    now=0.
    started=time.monotonic()
    try:
        while now<seconds-1e-9:
            until=min(now+1/30,response["next_model_time_s"],seconds)
            frames=driver.tick(packet,now,until)
            if controller=='current':
                until=driver.advance_until
            response=client.post("/api/world/advance",id=sid,dt=until-now,can_frames=frames)
            packet=response["model"]
            now=response["sim_time_s"]
            if realtime:
                time.sleep(max(0.,started+now-time.monotonic()))
            if controller=='current' and driver.finished:
                break
        if controller=='current' and not driver.finished:
            # A client time limit is a normal stop, not loss of the CAN sender.
            client.post('/api/world/advance',id=sid,dt=0.,can_frames=driver.stop(now))
    finally:
        if controller=='current':
            driver.close()
    result=dict(id=sid,model=packet,diagnostics=client.post("/api/world/diagnostics",id=sid))
    if controller=="current":
        result["controller_state"]=driver.telemetry.get('state','RUNNING')
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8767")
    parser.add_argument("--controller",choices=("demo","current"),default="demo")
    parser.add_argument("--seconds",type=float,default=15.)
    parser.add_argument("--output")
    parser.add_argument("--options",help="Path to a JSON scenario options object")
    parser.add_argument("--realtime",action="store_true",help="Pace virtual time for live UI observation")
    args=parser.parse_args()
    from pathlib import Path
    options=json.loads(Path(args.options).read_text(encoding="utf-8-sig")) if args.options else None
    result=run(args.url,args.controller,args.seconds,options,args.realtime,
               on_create=lambda sid:print(f"World ID: {sid}",flush=True,file=__import__('sys').stderr))
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text,encoding="utf-8")
    else:
        print(text)
