"""회전 중심(ICR) 동심원 적합 타당성 검사."""
import json, glob, os
import numpy as np, pandas as pd
from scipy.optimize import minimize

REC = r"c:\Users\dhshs\Downloads\로그수집\extracted\depth_cam\rec"

def cmds_of(p):
    out=[]
    for l in open(p,encoding='utf-8'):
        l=l.strip()
        if l:
            d=json.loads(l)
            if d.get('phase')=='cmd': out.append(d)
    return out

segs=[]
for cs in sorted(glob.glob(os.path.join(REC,"*_control_seq.jsonl"))):
    base=cs.replace("_control_seq.jsonl","")
    ps=base+"_pallet_state.csv"
    if not os.path.exists(ps): continue
    df=pd.read_csv(ps); df=df[df.det_ok==1].dropna(subset=['pos_x','pos_z'])
    cl=cmds_of(cs); tag=os.path.basename(base)[-6:]
    for i,r in enumerate(cl):
        if not str(r.get('cmd','')).startswith('ROT'): continue
        t0=r['t_mono']; t1=cl[i+1]['t_mono'] if i+1<len(cl) else None
        if t1 is None: continue
        d=df[(df.t_mono>=t0-1.0)&(df.t_mono<=t1+3.0)]
        if len(d)<12: continue
        segs.append(dict(tag=tag,step=r.get('step'),cmd=r['cmd'],t0=t0,t1=t1,
                         t=d.t_mono.values,x=d.pos_x.values,z=d.pos_z.values,
                         yaw=d.yaw_deg.values))
print("segments:",len(segs))

def resid(C, mirror):
    cx,cz=C; tot=[]
    for s in segs:
        sx = -cx if (mirror and s['cmd']=='ROT_RIGHT') else cx
        r=np.hypot(s['x']-sx, s['z']-cz)
        tot.append(r-r.mean())
    return np.concatenate(tot)

for mirror in (False,True):
    f=lambda C: float(np.sum(resid(C,mirror)**2))
    best=None
    for z0 in (-2,-1,-0.5,0,0.5,1,2):
        for x0 in (-1,0,1):
            r=minimize(f,[x0,z0],method='Nelder-Mead',options=dict(xatol=1e-4,fatol=1e-8,maxiter=4000))
            if best is None or r.fun<best.fun: best=r
    n=len(resid(best.x,mirror))
    print(f"mirror={mirror}  C=({best.x[0]:+.3f},{best.x[1]:+.3f})  rms={np.sqrt(best.fun/n)*1000:.1f} mm  n={n}")

# 고정 C=(0,-0.8) 대비
for C in [(0,-0.8),(0,0.0),(0,-1.9)]:
    r=resid(np.array(C),False); print(f"fixed C={C} rms={np.sqrt(np.mean(r**2))*1000:.1f} mm")
