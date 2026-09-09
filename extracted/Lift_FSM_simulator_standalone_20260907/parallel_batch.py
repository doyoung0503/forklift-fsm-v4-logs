"""Headless experiments: one fresh spawned process per case, bounded concurrency."""
import concurrent.futures as cf
import multiprocessing as mp
import json
import os
import threading
import time
from pathlib import Path
from uuid import uuid4


def run_case(case):
    # Imports and mutable FSM globals are isolated from server and other cases.
    from process_runtime import Session
    from batched_session import LocalController
    start=time.perf_counter()
    session=Session(case['options'],case.get('overrides',{}),capture=False,
                    log_root=None,show_window=False,controller_factory=LocalController)
    try:
        report=session.run()
        for key in ('frames','trace','can_frames','events'):
            report.pop(key,None)
        report.update(case_id=case['case_id'],compute_seconds=time.perf_counter()-start)
        return report
    finally:
        session.close()


class BatchJobs:
    def __init__(self,root=None):
        self.root=Path(root or Path(__file__).parent/'batch_results')
        self.jobs={};self.lock=threading.RLock()

    def start(self,data):
        cases=data.get('cases')
        workers=data.get('workers',min(4,os.cpu_count() or 1))
        if type(workers) is not int or not 1<=workers<=min(32,os.cpu_count() or 1):
            raise ValueError('Invalid worker count (1..min(32, CPU count))')
        if not isinstance(cases,list) or not 1<=len(cases)<=1000:
            raise ValueError('Expected 1..1000 cases')
        if any(not isinstance(c,dict) or not isinstance(c.get('options'),dict) for c in cases):
            raise ValueError('Each case needs options')
        # JSON roundtrip makes the accepted request immutable and rejects NaN.
        cases=json.loads(json.dumps(cases,allow_nan=False))
        with self.lock:
            if any(j['status']=='running' for j in self.jobs.values()):
                raise ValueError('A batch is already running')
            ident=uuid4().hex;directory=self.root/ident;directory.mkdir(parents=True)
            for index,c in enumerate(cases):c['case_id']=index
            job=dict(id=ident,status='running',total=len(cases),workers=min(workers,len(cases)),
                     results=[],cancel=False,directory=str(directory),started=time.time(),error=None)
            self.jobs[ident]=job
            (directory/'manifest.json').write_text(json.dumps(dict(cases=cases,workers=job['workers']),indent=2),encoding='utf8')
            threading.Thread(target=self._run,args=(job,cases),daemon=True).start()
            return dict(id=ident,total=len(cases),workers=job['workers'],directory=str(directory))

    def _run(self,job,cases):
        pending={};index=0
        try:
            with cf.ProcessPoolExecutor(max_workers=job['workers'],mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
                while index<len(cases) or pending:
                    with self.lock:cancel=job['cancel']
                    while not cancel and index<len(cases) and len(pending)<job['workers']:
                        c=cases[index];pending[pool.submit(run_case,c)]=c;index+=1
                    if not pending:break
                    completed,_=cf.wait(pending,timeout=.2,return_when=cf.FIRST_COMPLETED)
                    for future in completed:
                        case=pending.pop(future)
                        try:result=future.result()
                        except Exception as error:
                            result=dict(case_id=case['case_id'],options=case['options'],overrides=case.get('overrides',{}),
                                state='ERROR',success=False,reasons=[str(error)],failure_reason=str(error))
                        with (Path(job['directory'])/'results.jsonl').open('a',encoding='utf8') as handle:
                            handle.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                        with self.lock:job['results'].append(result)
            with self.lock:job['status']='cancelled' if job['cancel'] else 'completed'
        except Exception as error:
            with self.lock:job.update(status='error',error=str(error))
        finally:
            with self.lock:job['ended']=time.time()
            (Path(job['directory'])/'summary.json').write_text(json.dumps(self.status(job['id']),indent=2,ensure_ascii=False),encoding='utf8')

    def status(self,ident,cursor=0):
        with self.lock:
            job=self.jobs[ident];rs=job['results']
            if type(cursor) is not int or not 0<=cursor<=len(rs):raise ValueError('Invalid cursor')
            groups={}
            for r in rs:
                o=r['options'];key=json.dumps({k:v for k,v in o.items() if k!='seed'},sort_keys=True)
                g=groups.setdefault(key,dict(x=o.get('x'),z=o.get('z'),yaw=o.get('yaw'),completed=0,passed=0,collisions=0,sim_seconds=0.,compute_seconds=0.,failures={}))
                g['completed']+=1;g['passed']+=int(r['success'])
                g['collisions']+=int(bool(r.get('collision')))
                g['sim_seconds']+=r.get('elapsed',0.)
                g['compute_seconds']+=r.get('compute_seconds',0.)
                for reason in r.get('reasons',[]):g['failures'][reason]=g['failures'].get(reason,0)+1
            return dict(id=ident,status=job['status'],error=job['error'],total=job['total'],completed=len(rs),
                workers=job['workers'],results=rs[cursor:],cursor=len(rs),groups=list(groups.values()),directory=job['directory'],
                seconds=job.get('ended',time.time())-job['started'])

    def request(self,route,data):
        if route.endswith('/start'):return self.start(data)
        if route.endswith('/status'):return self.status(data['id'],data.get('cursor',0))
        if route.endswith('/cancel'):
            with self.lock:self.jobs[data['id']]['cancel']=True
            return dict(cancelling=True)
        raise ValueError('Unknown batch route')


JOBS=BatchJobs()
