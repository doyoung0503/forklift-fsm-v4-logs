# -*- coding: utf-8 -*-
"""pose 소스별로 전체 파이프라인을 돌리고 결과를 비교한다.

  log         원본 pallet_state.csv (현장 4점 IPPE)
  recomp4     재추론 + 4점 IPPE   (전 프레임)
  recompmulti 재추론 + 다중 키포인트 PnP (전 프레임)

각 소스의 산출물은 out/<source>/ 에 들어간다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# "<source>" 또는 "<source>:<signal>"
COMBOS = sys.argv[1:] or ["log:bearing", "recomp4:bearing",
                          "recompmulti:bearing", "recompmulti:yaw"]
PY = sys.executable


def run(script, env):
    r = subprocess.run([PY, os.path.join(HERE, script)], env=env, cwd=HERE,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit("%s failed for %s/%s" % (script, env["ROT_POSE_SOURCE"], env["ROT_ANGLE_SIGNAL"]))
    return r.stdout


def main():
    rows = []
    for combo in COMBOS:
        src, _, sig = combo.partition(":")
        sig = sig or "bearing"
        name = "%s_%s" % (src, sig)
        out = os.path.join(HERE, "out", name)
        os.makedirs(out, exist_ok=True)
        env = dict(os.environ, ROT_POSE_SOURCE=src, ROT_ANGLE_SIGNAL=sig,
                   ROT_OUT=out, PYTHONIOENCODING="utf-8")
        print("=" * 78)
        print("[%s]" % name, flush=True)
        run("extract_rotation_dataset.py", env)
        print(run("fit_rotation_model.py", env).split("총회전각")[0], flush=True)
        run("report_rotation_model.py", env)

        with open(os.path.join(out, "rotation_model_fit.json"),
                  encoding="utf-8") as fh:
            rep = json.load(fh)
        seg = pd.read_csv(os.path.join(out, "segments.csv"))
        use = seg[seg.usable]
        p = rep["selected_params"]
        rows.append(dict(
            source=name, n_used=len(use), n_total=len(seg),
            bearing_noise_deg=float(use.bearing_noise_deg.median()),
            T_on=p["T_on"], T_off=p["T_off"], alpha=p.get("alpha"),
            beta=p.get("beta"), w_max=p["w_max"],
            traj_rms_deg=rep["fit_rms_deg"],
            total_rms_deg=rep["total_angle_prediction"]["rms_deg"],
            loo_rms_deg=rep.get("loo", {}).get("rms_deg"),
            delay_median=rep["startup_delay_measured"]["median"],
            delay_sd=rep["startup_delay_measured"]["sd"],
            coast_median=float(use.coast_deg.median()),
            settle_p90=float(np.nanpercentile(
                use.settle_time_after_stop_s.dropna(), 90)),
        ))

    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(HERE, "out", "source_comparison.csv"), index=False,
               encoding="utf-8-sig")
    print("=" * 78)
    print(cmp.to_string(index=False, float_format=lambda v: "%8.3f" % v))


if __name__ == "__main__":
    main()
