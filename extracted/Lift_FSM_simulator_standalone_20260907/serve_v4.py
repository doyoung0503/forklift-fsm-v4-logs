"""Local UI/API and reproducible headless validation for the current Python FSM."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import threading
import time
from datetime import datetime, timezone
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from process_runtime import HERE, LOCK, Session, clean, fingerprint, replay_trace, summarize
from controller_process import ControllerProcessError
from serve_world import REGISTRY,protocol_info
from camera_service import CAMERA
from batched_session import BatchedSession
from parallel_batch import JOBS

SESSIONS = {}


def background_clock():
    """Advance live sessions independently of browser RAF/visibility."""
    while True:
        with LOCK:
            for session in list(SESSIONS.values()):
                if not getattr(session,'playing',False) or session.done:
                    continue
                target=session.play_sim+(time.perf_counter()-session.play_wall)*session.play_rate
                if session.now < target:
                    try:
                        if hasattr(session,'step_batch'):
                            session.step_batch(target)
                        else:
                            session.step()
                        wall=time.perf_counter()
                        if session.logger and (session.done or wall-getattr(session,'perf_wall',0)>=1):
                            sample=dict(wall_utc=datetime.now(timezone.utc).isoformat(),
                                sim_time_s=session.now,state=session.status['state'],
                                clock_domain='server',server_elapsed_s=wall-session.play_wall,
                                sim_since_play_s=session.now-session.play_sim,
                                time_scale=session.play_rate,
                                phase_totals=getattr(session,'phase_perf',{}),
                                transport_totals=getattr(session,'transport_perf',{}))
                            with (session.logger.directory/'run_server_performance.jsonl').open('a',encoding='utf8') as handle:
                                handle.write(json.dumps(sample)+'\n')
                            session.perf_wall=wall
                    except Exception as error:
                        session.playing=False
                        session.background_error=str(error)
                if session.done:
                    session.playing=False
        time.sleep(.001)


threading.Thread(target=background_clock,daemon=True,name='simulation-clock').start()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def log_message(self, *args):
        pass

    def send_head(self):
        # Development UI must not keep an older JS bundle while a new worker
        # protocol is running, even for multiple saves in the same second.
        if 'If-Modified-Since' in self.headers:
            del self.headers['If-Modified-Since']
        return super().send_head()

    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()

    def reply(self, data, status=200):
        raw = json.dumps(clean(data), ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/api/info":
            try:
                self.reply(fingerprint())
            except ControllerProcessError as error:
                self.reply({'error':str(error)},500)
            return
        if route == "/api/protocol":
            self.reply(protocol_info())
            return
        if route == "/":
            self.path = "/v4.html"
        # Serve only this UI's static assets, not reports or Python source.
        elif route.startswith("/api/"):
            self.reply({"error":"Unknown route"},404)
            return
        resolved = (HERE / self.path.split("?")[0].lstrip("/")).resolve()
        image_asset = resolved == (HERE / 'assets/pallet_top.png').resolve()
        if not resolved.is_relative_to(HERE) or (not image_asset and resolved.suffix not in (".html",".css",".js",".csv")):
            self.send_error(404)
            return
        super().do_GET()

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and origin != "http://"+self.headers.get("Host", ""):
            self.reply({"error":"Same-origin requests only"},403)
            return
        try:
            length = int(self.headers.get("Content-Length",0))
            if not 0 < length <= 30_000_000:
                raise ValueError("Request size is outside the supported range")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data,dict):
                raise ValueError("Expected JSON object")
            route = urlsplit(self.path).path
            if route.startswith('/api/batch/'):
                self.reply(JOBS.request(route,data))
                return
            if route == '/api/camera':
                # Rendering neither advances the world nor samples model errors.
                # Keep it outside the FSM lock and in its own GPU-owning process.
                started=time.perf_counter()
                result=CAMERA.render(data['frame'],data['options'])
                result['server_camera_ms']=(time.perf_counter()-started)*1000
                self.reply(result)
                return
            if route.startswith("/api/world/"):
                self.reply(REGISTRY.request(route,data))
                return
            lock_start=time.perf_counter()
            with LOCK:
                lock_wait_ms=(time.perf_counter()-lock_start)*1000
                if route == "/api/session":
                    session = BatchedSession(data.get("options"), data.get("overrides"),show_window=data.get('show_window',True))
                    previous=SESSIONS.pop(data.get('replace_id'),None)
                    if previous is not None:
                        previous.close()
                    if len(SESSIONS) >= 8:
                        SESSIONS.pop(next(iter(SESSIONS))).close()
                    sid = uuid4().hex
                    SESSIONS[sid] = session
                    result = dict(id=sid, frame=session.initial,info=session.provenance)
                elif route == '/api/session/close':
                    session=SESSIONS.pop(data['id'])
                    session.close()
                    result=dict(closed=True)
                elif route == '/api/play':
                    session=SESSIONS[data['id']]
                    rate=float(data.get('rate',1))
                    if not 0 < rate <= 8:
                        raise ValueError('Playback rate must be in (0,8]')
                    session.play_sim=session.now
                    session.play_wall=time.perf_counter()
                    session.play_rate=rate
                    session.playing=not session.done
                    result=dict(playing=session.playing)
                elif route == '/api/pause':
                    session=SESSIONS[data['id']]
                    session.playing=False
                    result=dict(playing=False,t=session.now)
                elif route == '/api/poll':
                    session=SESSIONS[data['id']]
                    if session.logger and data.get('display_samples'):
                        with (session.logger.directory/'run_performance.jsonl').open('a',encoding='utf8') as handle:
                            for sample in data['display_samples'][:100]:
                                handle.write(json.dumps(dict(sample,wall_utc=datetime.now(timezone.utc).isoformat()),allow_nan=False)+'\n')
                    cursor=int(data.get('cursor',0))
                    if not 0 <= cursor <= len(session.frames):
                        raise ValueError('Invalid frame cursor')
                    frames=session.frames[cursor:cursor+600]
                    next_cursor=cursor+len(frames)
                    result=dict(frames=frames,cursor=next_cursor,server_lock_wait_ms=lock_wait_ms,
                                done=session.done and next_cursor==len(session.frames))
                    if getattr(session,'background_error',None):
                        raise ValueError(session.background_error)
                    if result['done']:
                        result['report']={k:v for k,v in session.report().items() if k not in ('frames','trace','can_frames')}
                elif route == '/api/present':
                    started=time.perf_counter()
                    result=SESSIONS[data['id']].present(data['frame'],data['presentation_id'],
                        data.get('jpeg'),data.get('options'))
                    result['server_lock_wait_ms']=lock_wait_ms
                    result['server_present_ms']=(time.perf_counter()-started)*1000
                elif route == '/api/performance':
                    session=SESSIONS[data['id']]
                    if session.logger is None:
                        raise ValueError('Run logging has not started')
                    record=dict(data['sample'],wall_utc=datetime.now(timezone.utc).isoformat())
                    # Independent of the main logger: the final display arrives
                    # after a completed run has closed its usual file handles.
                    with (session.logger.directory/'run_performance.jsonl').open('a',encoding='utf-8') as handle:
                        handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    result=dict(saved=True)
                elif route == "/api/step":
                    started=time.perf_counter()
                    session = SESSIONS[data["id"]]
                    if getattr(session,'playing',False):
                        raise ValueError('Pause server playback before manual stepping')
                    controller_before=session.controller_wall_ms
                    count = data.get("count",1)
                    if type(count) is not int or not 1 <= count <= 120:
                        raise ValueError("count must be an integer in [1,120]")
                    frames = []
                    duration=data.get("duration")
                    if duration is not None and (not isinstance(duration,(int,float)) or not 0<duration<=1):
                        raise ValueError("duration must be in (0,1] seconds")
                    deadline=session.now+duration if duration is not None else None
                    for _ in range(120 if duration is not None else count):
                        frames.append(session.step())
                        if session.done or (deadline is not None and session.now>=deadline):
                            break
                    result = dict(frames=frames,done=session.done)
                    result['server_step_ms']=(time.perf_counter()-started)*1000
                    result['controller_tick_ms']=session.controller_wall_ms-controller_before
                    if session.done:
                        result["report"] = {k:v for k,v in session.report().items() if k not in ("frames","trace","can_frames")}
                elif route == "/api/report":
                    result = SESSIONS[data["id"]].report()
                elif route == "/api/run":
                    session = Session(data.get("options"), data.get("overrides"),capture=bool(data.get("capture",False)),
                                      show_window=data.get('show_window',False))
                    result = session.run()
                    if session.show_window:
                        if len(SESSIONS)>=8:
                            SESSIONS.pop(next(iter(SESSIONS))).close()
                        sid=uuid4().hex
                        SESSIONS[sid]=session
                        result['window_session_id']=sid
                elif route == "/api/replay-check":
                    if not data.get('source_id') or not data.get('simulator_id'):
                        raise ValueError('FSM and simulator source IDs are required')
                    trace = data["trace"]
                    if not isinstance(trace,list) or not 1 <= len(trace) <= 72001:
                        raise ValueError("Invalid trace length")
                    result = replay_trace(trace,data.get("overrides"),data['source_id'],data['simulator_id'])
                else:
                    self.reply({"error":"Unknown route"},404)
                    return
            self.reply(result)
        except (ValueError,KeyError,TypeError) as exc:
            self.reply({"error":str(exc)},400)
        except Exception as exc:
            self.reply({"error":f"{type(exc).__name__}: {exc}"},500)


def validate(output):
    output.mkdir(parents=True,exist_ok=True)
    reports = []
    for perception, x,z,yaw in itertools.product(("oracle","fov"),(-.3,0.,.3),(1.8,2.,2.2),(-15.,0.,15.)):
        s = Session(dict(x=x,z=z,yaw=yaw,perception=perception))
        r = s.run()
        r["replay_check"] = replay_trace(r["trace"])
        if not r["replay_check"]["passed"]:
            raise AssertionError("Original FSM replay mismatch")
        r["replay_check"] = {k:v for k,v in r["replay_check"].items() if k != "mismatches"}
        r.pop("frames")
        r.pop("trace")
        r.pop("can_frames")
        reports.append(r)
        print(f'{len(reports)}/54 {perception} x={x:+.1f} z={z:.1f} yaw={yaw:+.0f}: {r["state"]}',flush=True)
    result = dict(provenance=fingerprint(),summary=summarize(reports),reports=reports)
    (output/"validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    with (output/"conditions.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["perception","x_m","z_m","yaw_deg","success","fsm_state","failure_state","reason","seconds","true_remaining_m","collision","replayed_frames"])
        for r in reports:
            o=r["options"]
            writer.writerow([o["perception"],o["x"],o["z"],o["yaw"],r["success"],r["state"],r["failure_state"],
                             " | ".join(r["reasons"]),r["elapsed"],r["true_remaining"],bool(r["collision"]),r["replay_check"]["frames"]])
    print(json.dumps(result["summary"],ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=8766)
    parser.add_argument("--no-browser",action="store_true")
    parser.add_argument("--validate",action="store_true")
    parser.add_argument("--output",type=Path,default=HERE/"validation_v4")
    args=parser.parse_args()
    if args.validate:
        validate(args.output)
    else:
        server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
        url=f"http://127.0.0.1:{args.port}/"
        print(f"Current Python FSM v4 simulator: {url} (virtual CAN ON; physical CAN OFF)",flush=True)
        if not args.no_browser:
            threading.Timer(.5,lambda:webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            CAMERA.close()
            for session in list(SESSIONS.values()):
                session.close()
            server.server_close()
