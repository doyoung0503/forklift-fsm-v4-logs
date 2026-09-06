import json, os, sys
import pandas as pd, numpy as np
REC = r"c:\Users\dhshs\Downloads\로그수집\extracted\depth_cam\rec"
tag, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
df = pd.read_csv(os.path.join(REC, tag + "_pallet_state.csv"))
seg = df[(df.t_mono >= t0 - 1.5) & (df.t_mono <= t1 + 6.0)]
for _, r in seg.iterrows():
    mark = ""
    if r.t_mono >= t0 and r.t_mono - 0.2 < t0: mark = " <== ROT"
    if r.t_mono >= t1 and r.t_mono - 0.2 < t1: mark = " <== STOP"
    yaw = f"{r.yaw_deg:+8.3f}" if r.det_ok == 1 and not pd.isna(r.yaw_deg) else "   ----"
    x = f"{r.pos_x:+7.3f}" if r.det_ok==1 and not pd.isna(r.pos_x) else "  ---"
    z = f"{r.pos_z:+7.3f}" if r.det_ok==1 and not pd.isna(r.pos_z) else "  ---"
    print(f"{r.t_mono-t0:+7.3f} {r.fsm_state:<16} det={int(r.det_ok)} yaw={yaw} x={x} z={z}{mark}")
