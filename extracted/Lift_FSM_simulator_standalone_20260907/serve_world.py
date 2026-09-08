"""FSM-independent virtual lifter server. Standard library only; no calib import."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from uuid import uuid4

from protocol_world import World,PROFILE,DEFAULTS,MODEL_PROTOCOL,CAN_PROTOCOL


def protocol_info():
    sample=World({"perception":"oracle"}).observe()
    return dict(model_protocol=MODEL_PROTOCOL,can_protocol=CAN_PROTOCOL,
                model_sample=sample,vehicle_profile=PROFILE,scenario_defaults=DEFAULTS,
                capabilities=dict(planar_drive=True,rotation=True,combined_drive_steer=True,
                                  hydraulic_motion=False,can_receive_feedback=False),
                endpoints=["/api/world/create","/api/world/advance","/api/world/model",
                           "/api/world/diagnostics","/api/world/can-log"],
                clock="Advance virtual time explicitly. CAN t is simulation-relative seconds; model monotonic clock has epoch 1000.")


class WorldRegistry:
    def __init__(self):
        self.worlds={}
        self.lock=threading.RLock()

    def request(self,path,data):
        with self.lock:
            if path=="/api/world/create":
                world=World(data.get("options"))
                sid=uuid4().hex
                if len(self.worlds)>=16:
                    del self.worlds[next(iter(self.worlds))]
                self.worlds[sid]=world
                return dict(id=sid,model=world.observe(),sim_time_s=world.now,next_model_time_s=world.next_model_time)
            world=self.worlds[data["id"]]
            if path=="/api/world/advance":
                packet=world.advance(data["dt"],data.get("can_frames",[]))
                return dict(model=packet,sim_time_s=world.now,next_model_time_s=world.next_model_time)
            if path=="/api/world/model":
                return dict(model=world.observe(),sim_time_s=world.now,next_model_time_s=world.next_model_time)
            if path=="/api/world/diagnostics":
                return world.diagnostics()
            if path=="/api/world/can-log":
                return dict(frames=world.can_log)
            raise ValueError("Unknown world endpoint")


REGISTRY=WorldRegistry()


class WorldHandler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def reply(self,value,status=200):
        raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path=="/api/protocol":
            self.reply(protocol_info())
        else:
            self.reply({"error":"Use /api/protocol or the POST world endpoints"},404)

    def do_POST(self):
        origin=self.headers.get("Origin")
        if origin and origin!="http://"+self.headers.get("Host",""):
            self.reply({"error":"Same-origin requests only"},403)
            return
        try:
            length=int(self.headers.get("Content-Length",0))
            if not 0<length<=1_000_000:
                raise ValueError("Invalid request size")
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict):
                raise ValueError("JSON object required")
            self.reply(REGISTRY.request(self.path,data))
        except (KeyError,ValueError,TypeError) as error:
            self.reply({"error":str(error)},400)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=8767)
    args=parser.parse_args()
    server=ThreadingHTTPServer(("127.0.0.1",args.port),WorldHandler)
    print(f"Independent CAN/model simulation: http://127.0.0.1:{args.port}/api/protocol",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
