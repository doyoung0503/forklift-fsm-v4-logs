"""회전 구간별 yaw / bearing / 회전중심각 비교 + 유효 회전중심 추정."""
import json, glob, os
import numpy as np, pandas as pd

REC = r"c:\Users\dhshs\Downloads\로그수집\extracted\depth_cam\rec"

def cmds_of(path):
    out=[]
    for line in open(path,encoding='utf-8'):
        line=line.strip()
        if not line: continue
        d=json.loads(line)
        if d.get('phase')=='cmd': out.append(d)
    return out

def robust(v):
    v=np.asarray(v,float); v=v[~np.isnan(v)]
    return np.nan if len(v)==0 else float(np.median(v))

rows=[]
for cs in sorted(glob.glob(os.path.join(REC,"*_control_seq.jsonl"))):
    base=cs.replace("_control_seq.jsonl","")
    ps=base+"_pallet_state.csv"
    if not os.path.exists(ps): continue
    df=pd.read_csv(ps); df=df[df.det_ok==1].dropna(subset=['yaw_deg','pos_x','pos_z'])
    cl=cmds_of(cs)
    tag=os.path.basename(base)[-6:]
    for i,r in enumerate(cl):
        if not str(r.get('cmd','')).startswith('ROT'): continue
        t0=r['t_mono']; t1=cl[i+1]['t_mono'] if i+1<len(cl) else None
        if t1 is None: continue
        pre=df[(df.t_mono>=t0-1.2)&(df.t_mono<=t0+0.4)]
        post=df[(df.t_mono>=t1+1.5)&(df.t_mono<=t1+4.0)]
        if len(pre)<4 or len(post)<4: continue
        sign = +1.0 if r['cmd']=='ROT_RIGHT' else -1.0
        y0,y1=robust(pre.yaw_deg),robust(post.yaw_deg)
        x0,z0=robust(pre.pos_x),robust(pre.pos_z)
        x1,z1=robust(post.pos_x),robust(post.pos_z)
        b0=np.degrees(np.arctan2(x0,z0)); b1=np.degrees(np.arctan2(x1,z1))
        def ang(C):
            a0=np.degrees(np.arctan2(x0-C[0],z0-C[1])); a1=np.degrees(np.arctan2(x1-C[0],z1-C[1]))
            return a1-a0
        rows.append(dict(tag=tag,step=r.get('step'),cmd=r['cmd'],sign=sign,
                         dur=round(t1-t0,3),n_pre=len(pre),n_post=len(post),
                         dyaw=y1-y0, dbear=b1-b0, dC08=ang((0.0,-0.8)),
                         x0=x0,z0=z0,x1=x1,z1=z1,y0=y0,y1=y1,
                         sd_pre=float(np.std(pre.yaw_deg)),sd_post=float(np.std(post.yaw_deg))))
t=pd.DataFrame(rows)
pd.set_option('display.width',200)
print(t[['tag','step','cmd','dur','dyaw','dbear','dC08','n_pre','n_post','sd_pre','sd_post']].to_string(index=False,float_format=lambda v:f"{v:8.2f}"))
print()
print("corr(dyaw,dbear)=",np.corrcoef(t.dyaw,t.dbear)[0,1])
# 유효 회전중심 z 를 스캔: dyaw 와 dC(z) 가 가장 잘 맞는 z
for zc in [-3,-2,-1.5,-1.0,-0.8,-0.5,-0.2,0.0,0.5,1.0]:
    d=[]
    for _,r in t.iterrows():
        a0=np.degrees(np.arctan2(r.x0,r.z0-zc)); a1=np.degrees(np.arctan2(r.x1,r.z1-zc))
        d.append(a1-a0)
    d=np.array(d)
    print(f"  zc={zc:+5.2f}  slope(dyaw~dC)={np.polyfit(d,t.dyaw,1)[0]:+6.3f}  rms(dyaw-dC)={np.sqrt(np.mean((t.dyaw-d)**2)):6.2f}")
