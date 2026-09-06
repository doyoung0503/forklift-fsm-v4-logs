"""빠른 확인: 회전 구간별 yaw / center bearing 시계열 요약."""
import json, glob, os
import numpy as np
import pandas as pd

REC = r"c:\Users\dhshs\Downloads\로그수집\extracted\depth_cam\rec"

def load_cmds(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("phase") in ("cmd", "begin", "end", "rotation_stop"):
            out.append(d)
    return out

for cs in sorted(glob.glob(os.path.join(REC, "*_control_seq.jsonl"))):
    base = cs.replace("_control_seq.jsonl", "")
    ps = base + "_pallet_state.csv"
    if not os.path.exists(ps):
        continue
    df = pd.read_csv(ps)
    recs = load_cmds(cs)
    tag = os.path.basename(base)
    print("="*100)
    print(tag, " frames:", len(df), " det_ok:", int(df["det_ok"].sum()))
    # rotation command windows: ROT_* cmd -> next STOP cmd
    cmds = [r for r in recs if r.get("phase") == "cmd"]
    for i, r in enumerate(cmds):
        if not str(r.get("cmd", "")).startswith("ROT"):
            continue
        t0 = r["t_mono"]
        t1 = None
        for r2 in cmds[i+1:]:
            t1 = r2["t_mono"]
            break
        seg = df[(df.t_mono >= t0 - 1.0) & (df.t_mono <= (t1 if t1 else t0+10) + 4.0)]
        seg = seg[seg.det_ok == 1]
        if len(seg) < 5:
            print(f"  step{r.get('step')} {r['cmd']} t0={t0:.3f} t1={t1} -> frames={len(seg)} (skip)")
            continue
        y = seg.yaw_deg.values
        print(f"  step{r.get('step')} {r['cmd']} t0={t0:.3f} dur={None if t1 is None else round(t1-t0,3)} "
              f"n={len(seg)} yaw[{y[0]:+.2f} -> {y[-1]:+.2f}] span={y.max()-y.min():.2f}")
