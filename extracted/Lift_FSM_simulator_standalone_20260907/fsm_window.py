"""Native result-only window using the real FSM's OpenCV diagram renderer."""
import os
import time
import base64
from types import SimpleNamespace


class FSMWindow:
    def __init__(self, *, offscreen=False):
        import cv2
        import numpy as np
        from ui.diagram_v4 import draw_fsm_v4_diagram_panel
        self.cv2,self.np,self.diagram=cv2,np,draw_fsm_v4_diagram_panel
        self.title=f'[SIMULATION] main_rec_v4.py | FSM PID {os.getpid()} | Virtual CAN'
        self.closed=False
        self.offscreen=offscreen
        self.last_image=None
        self.next_event_pump=0.
        self.last_draw=0.
        self.last_key=None
        self.camera_renderer=None
        from concurrent.futures import ThreadPoolExecutor
        self.camera_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='fsm-camera')
        self.camera_future=None
        self.camera_image=None
        self.camera_info=None
        self.camera_error=None
        self.camera_requested_time=None
        self.last_render_args=None
        if not offscreen:
            cv2.namedWindow(self.title,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title,1500,720)

    def render_snapshot(self,frame,options,config,overrides,jpeg=None):
        """Display an immutable recorded frame; never step or mutate the FSM."""
        image=None
        if jpeg:
            image=self.cv2.imdecode(self.np.frombuffer(base64.b64decode(jpeg,validate=True),dtype=self.np.uint8),self.cv2.IMREAD_COLOR)
            if image is None or image.shape!=(480,640,3):
                raise ValueError('Presentation camera image must be 640x480')
        values={**frame,'motion_target_status':frame.get('target'),
            'cmd_status':SimpleNamespace(code=frame['command']),
            '_visual_recovery_origin_state':(frame.get('diagram') or {}).get('recovery_origin')}
        proxy=SimpleNamespace(fsm=SimpleNamespace(**values),overrides=overrides,telemetry=lambda:frame)
        self.render(proxy,frame.get('model_packet'),frame['t'],frame.get('lifecycle') or {},
            [frame['can']],dict(truth=frame['truth'],options=options,config=config),
            synchronized=True,camera_image=image)
        self.last_render_args=None
        self.pump(force=True)  # Flush this image before acknowledging synchronized display.

    def render(self,driver,packet,now,lifecycle,can_frames,scene=None,*,synchronized=False,camera_image=None):
        if self.closed:
            return
        self.last_render_args=(driver,packet,now,lifecycle,can_frames,scene)
        from v4_runtime import environment
        with environment(now,driver.overrides):
            status=driver.telemetry()
        key=(status['state'],status['command'],lifecycle.get('outcome'))
        wall=time.monotonic()
        if not synchronized and key==self.last_key and wall-self.last_draw<1/30:
            return
        self.last_draw,self.last_key=wall,key
        cv,np=self.cv2,self.np
        from ui.text_rendering import draw_ui_text
        with environment(now,driver.overrides):
            diagram=self.diagram(driver.fsm,(720,580))
        panel=np.full((720,440,3),(25,28,31),dtype=np.uint8)
        step=(packet or {}).get('step',{})
        def fmt(value):
            return '--' if value is None else f'{float(value):.3f}'
        pose=step.get('offset_smooth') or [None,None,None]
        meta=step.get('vision_meta') or {}
        lines=[
            ('SIMULATION MODE',(90,220,145)),
            ('main_rec_v4.py  |  model results only',(235,237,240)),
            ('Camera OFF  |  Inference OFF  |  Physical CAN OFF',(160,175,190)),
            (f'Virtual time: {now:.3f} s',(235,237,240)),
            (f"State: {status['state']}",(90,220,145)),
            (f"Command: {status['command']}",(235,237,240)),
            (f"Detection: {'YES' if step.get('det_ok') else 'NO'}   Sequence: {(packet or {}).get('sequence','--')}",(235,237,240)),
            (f"X / Y / Z: {fmt(pose[0])} / {fmt(pose[1])} / {fmt(pose[2])} m",(235,237,240)),
            (f"Yaw: {fmt(step.get('yaw_smooth'))} deg",(235,237,240)),
            (f"Result FPS: {fmt((packet or {}).get('result_fps'))}",(235,237,240)),
            (f"Synthetic latency: {fmt(meta.get('inference_latency_sec'))} s",(235,237,240)),
            (f"FSM input: {status.get('input_mode','waiting')}",(90,220,255)),
            ('Clock mode: FSM steps without a pose during insertion',(160,175,190)),
            (f"Run: {lifecycle.get('outcome','running')}",(90,220,145)),
        ]
        if can_frames:
            movement=next((f for f in reversed(can_frames) if f['id']==0x1e3),None)
            if movement:
                self.last_can=' '.join(f'{v:02X}' for v in movement['data'])
        lines.append((f"CAN 0x1E3: {getattr(self,'last_can','--')}",(235,237,240)))
        reason=lifecycle.get('reason') or status.get('failure_reason')
        if reason:
            lines.append((str(reason),(100,120,255)))
        lines.extend([('Run / Pause / Step / Reset: simulator browser',(160,175,190)),
                      ('Window stays open when the run finishes.',(160,175,190)),
                      ('Esc / Q / close: cancel at the next simulation tick.',(160,175,190))])
        for index,(line,color) in enumerate(lines):
            draw_ui_text(panel,line,16,30+index*31,408,18,color)
        camera_panel=np.full((720,480,3),(25,28,31),dtype=np.uint8)
        if synchronized:
            if camera_image is not None:
                camera_panel[:360]=cv.resize(camera_image,(480,360),interpolation=cv.INTER_AREA)
            cv.putText(camera_panel,f'Synchronized frame | t={now:.3f}s',
                (16,515),cv.FONT_HERSHEY_SIMPLEX,.55,(220,230,240),1,cv.LINE_AA)
            if camera_image is None:
                cv.putText(camera_panel,'Camera image unavailable for this frame',(16,40),cv.FONT_HERSHEY_SIMPLEX,.5,(100,140,255),1)
        elif scene:
            try:
                frame=dict(t=now,truth=scene['truth'],model_packet=packet,**status)
                if self.camera_future is not None and self.camera_future.done():
                    self.camera_image,self.camera_info=self.camera_future.result()
                    self.camera_future=None
                if self.camera_future is None and self.camera_requested_time!=now:
                    self.camera_requested_time=now
                    self.camera_future=self.camera_executor.submit(self._render_camera,frame,scene)
                if self.camera_image is not None:
                    info=self.camera_info;camera_panel[:360]=cv.resize(self.camera_image,(480,360),interpolation=cv.INTER_AREA)
                    cv.putText(camera_panel,f'Synthetic RGB | t={info["t"]:.3f}s | {info["render_ms"]:.1f}ms',
                        (16,515),cv.FONT_HERSHEY_SIMPLEX,.55,(220,230,240),1,cv.LINE_AA)
                else:
                    cv.putText(camera_panel,'Loading 3D camera (FSM keeps running)',(16,40),cv.FONT_HERSHEY_SIMPLEX,.6,(220,230,240),1)
            except Exception as error:
                cv.putText(camera_panel,'Camera renderer unavailable (FSM continues)',(12,40),cv.FONT_HERSHEY_SIMPLEX,.5,(100,140,255),1)
                cv.putText(camera_panel,str(error)[:85],(12,70),cv.FONT_HERSHEY_SIMPLEX,.4,(100,140,255),1)
        else:
            cv.putText(camera_panel,'Waiting for simulator world pose',(20,40),cv.FONT_HERSHEY_SIMPLEX,.6,(220,230,240),1)
        self.last_image=np.hstack([camera_panel,panel,diagram])
        if not self.offscreen:
            cv.imshow(self.title,self.last_image)

    def _render_camera(self,frame,scene):
        # The same background thread owns the GL context for its whole life.
        if self.camera_renderer is None:
            from simulation_camera import SimulationCamera
            self.camera_renderer=SimulationCamera()
        return self.camera_renderer.render(frame,scene['options'],scene['config'])

    def pump(self, *, force=False):
        # Wall time only: virtual ticks and CAN/sensor scheduling are unchanged.
        # Tick and idle callers share one limit; explicit presentation must flush
        # imshow before the browser receives its matching-frame acknowledgment.
        if self.closed or self.offscreen:
            return False
        wall=time.perf_counter()
        if not force and wall < self.next_event_pump:
            return False
        self.next_event_pump=wall+1/30
        if self.camera_future is not None and self.camera_future.done() and self.last_render_args:
            self.render(*self.last_render_args)
        cv=self.cv2
        key=cv.waitKey(1)&255
        try:
            visible=cv.getWindowProperty(self.title,cv.WND_PROP_VISIBLE)>=1
        except cv.error:
            visible=False
        if key in (27,ord('q')) or not visible:
            self.close()
        return True

    def close(self):
        if not self.closed:
            self.closed=True
            self.camera_executor.shutdown(wait=False,cancel_futures=True)
            if self.offscreen:
                return
            try:
                self.cv2.destroyWindow(self.title)
                self.cv2.waitKey(1)
            except self.cv2.error:
                pass
