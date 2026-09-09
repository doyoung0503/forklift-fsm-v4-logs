"""Current control.py -> virtual CAN channel. This is a replaceable driver adapter.

The vehicle never imports this module. Real control.py frame factories and
_write boundaries are executed, but their channel writes go only to memory.
The discrete scheduler models the configured periods/bursts, not OS scheduling.
"""
from contextlib import contextmanager
from copy import deepcopy
import heapq
from types import SimpleNamespace

from protocol_world import can_frame, EPOCH


class WireFrame:
    def __init__(self,id_,data,flags=0):
        self.id=id_
        self.data=list(data)
        self.flags=flags


class CurrentCanTransport:
    def __init__(self,control):
        self.control=control
        self.ctx=control._BusCtx()
        self.ctx.ch=self
        self.templates=deepcopy(control.MOVEMENT_TEMPLATES)
        self.time=0.
        self.outbox=[]
        self.pending=False
        self.queue=[]
        self.serial=0
        self.last_sent="stop"
        for i in range(5):
            self.schedule(i*.005,"burst",("stop","driving_mode",i+1,5))
        self.next_ctrl=.025
        self.next_mov=.025
        self.next_hb=.225

    def write(self,frame):
        self.outbox.append(can_frame(int(frame.id),list(frame.data),self.time,
                                      bool(frame.flags & 0x4)))

    def writeSync(self,timeout):
        pass  # Memory channel writes are immediate.

    def schedule(self,t,kind,args):
        self.serial+=1
        heapq.heappush(self.queue,(t,self.serial,kind,args))

    @contextmanager
    def bind(self,now):
        c=self.control
        saved={k:getattr(c,k) for k in ("_CTX","Frame","time","_activate_command","MOVEMENT_TEMPLATES","_CAN_TX_OBSERVER")}
        if c._CAN_ENABLED or c._CAN_THREAD is not None:
            raise RuntimeError("Virtual adapter requires physical CAN disabled")
        try:
            self.time=now
            c._CTX=self.ctx
            c.Frame=WireFrame
            c.MOVEMENT_TEMPLATES=self.templates
            c._CAN_TX_OBSERVER=None
            c.time=SimpleNamespace(monotonic=lambda:EPOCH+self.time,
                                   monotonic_ns=lambda:round((EPOCH+self.time)*1e9))
            def activate(movement,control="driving_mode"):
                saved["_activate_command"](movement,control)
                self.pending=True
            c._activate_command=activate
            yield self
        finally:
            self.templates=c.MOVEMENT_TEMPLATES
            for key,value in saved.items():
                setattr(c,key,value)

    def pump(self,until):
        """Emit actual frame-factory output up to a virtual bus time."""
        c=self.control
        if self.pending:
            self.pending=False
            movement,mode=c._get_tx_state()
            n=c.DRIVE_ENTRY_BURST_N if movement!="stop" and self.last_sent=="stop" else 1
            for i in range(n):
                self.schedule(self.time+i*c.DRIVE_ENTRY_BURST_DT,"burst",(movement,mode,i+1,n))
            finish=self.time+(n*c.DRIVE_ENTRY_BURST_DT if n>1 else 0)
            self.next_ctrl=finish+c.CTRL_PERIOD
            self.next_mov=finish+c.MOV_PERIOD
            self.last_sent=movement
        while True:
            queued=self.queue[0][0] if self.queue else float("inf")
            deadline=c.timed_rotation_deadline()
            deadline=deadline-EPOCH if deadline is not None else float('inf')
            at=min(queued,self.next_ctrl,self.next_mov,self.next_hb,deadline)
            if at>until+1e-9:
                break
            self.time=at
            movement,mode=c._get_tx_state()
            timer=c.timed_rotation_status()
            timer_stop_due=timer.get('expired') and timer.get('stopped') is None
            if deadline==at or timer_stop_due:
                c._write_command_once(movement,mode,tx_source='timed_rotation_deadline')
                self.last_sent=movement
            elif queued==at:
                _,_,_,(movement,mode,index,count)=heapq.heappop(self.queue)
                c._write_command_once(movement,mode,tx_source="virtual_burst",burst_index=index,burst_count=count)
            else:
                movement,mode=c._get_tx_state()
                if self.next_ctrl<=at+1e-10:
                    c._write(c._mk_control(mode),"control")
                    self.next_ctrl=at+c.CTRL_PERIOD
                if self.next_mov<=at+1e-10:
                    c._write(c._mk_movement(movement),"movement",tx_source="virtual_periodic",movement=movement)
                    self.next_mov=at+c.MOV_PERIOD
                if self.next_hb<=at+1e-10:
                    c._write(c._mk_heartbeat(),"heartbeat")
                    self.next_hb=at+c.HEARTBEAT_PERIOD
        self.time=until
        frames=self.outbox
        self.outbox=[]
        return frames
