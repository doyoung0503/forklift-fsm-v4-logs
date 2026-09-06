# -*- coding: utf-8 -*-
"""토크(조이스틱 편향) 30 회전 응답 곡선 적합.

모델 A (1차 지연):  omega' = (u(t)*w_max - omega) / tau
모델 B (등가속/등감속): u=1 이면 alpha 로 가속(w_max 포화), u=0 이면 beta 로 감속
공통: u(t) = 1  for  T_on <= t < T_cmd + T_off,  else 0
      (T_on = 회전명령 시동지연, T_off = STOP 명령 시동지연)

t 는 회전명령(cmd) 송신 시점 기준. 좌/우는 동일 특성으로 가정하고 부호로만 구분,
관측 회전각은 방향 정규화되어 항상 증가 방향이다.
"""
from __future__ import annotations

import io
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("ROT_OUT", os.path.join(HERE, "out"))
exec(io.open(os.path.join(HERE, "rot_common.py"), encoding="utf-8").read())

FIT_LO, FIT_HI = -0.5, 2.5          # 적합 구간: cmd-0.5s ~ STOP+2.5s
SIM_DT = 0.005


# ---------------------------------------------------------------- simulators
# 두 모델 모두 구간별 해석해가 있으므로 수치적분하지 않는다 (LOO 교차검증이
# 수백 번 돌기 때문에 속도가 중요하다).  drive 구간 = [T_on, hold + T_off].
def _theta_omega(model, p, t, drive_end):
    """t (배열, 회전명령 기준) 에서의 (theta, omega). t<0 은 0 으로 잘린다."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, None)
    T_on = p["T_on"]
    D = max(0.0, drive_end - T_on)          # 실제 구동 시간
    td = np.clip(t - T_on, 0.0, D)          # 구동구간 경과
    ta = np.clip(t - drive_end, 0.0, None)  # 구동 종료 후 경과

    if model == "A":
        tau, wm = max(1e-6, p["tau"]), p["w_max"]
        w_drive = wm * (1.0 - np.exp(-td / tau))
        th_drive = wm * (td - tau * (1.0 - np.exp(-td / tau)))
        w_end = wm * (1.0 - np.exp(-D / tau))
        w = np.where(t <= drive_end, w_drive, w_end * np.exp(-ta / tau))
        theta = np.where(t <= drive_end, th_drive,
                         th_drive + w_end * tau * (1.0 - np.exp(-ta / tau)))
        return theta, w

    a, b, wm = p["alpha"], p["beta"], p["w_max"]
    t_sat = wm / a
    w_drive = np.minimum(wm, a * td)
    th_drive = np.where(td <= t_sat, 0.5 * a * td ** 2,
                        0.5 * wm * t_sat + wm * (td - t_sat))
    w_end = min(wm, a * D)
    t_stop = w_end / b
    tc = np.clip(ta, 0.0, t_stop)
    w = np.where(t <= drive_end, w_drive, np.maximum(0.0, w_end - b * ta))
    theta = np.where(t <= drive_end, th_drive,
                     th_drive + w_end * tc - 0.5 * b * tc ** 2)
    return theta, w


def simulate(model, p, t_cmd_duration, t_end, dt=SIM_DT):
    """호환용 격자 출력 (t, omega, theta)."""
    n = int(np.ceil(t_end / dt)) + 1
    t = np.arange(n) * dt
    theta, w = _theta_omega(model, p, t, t_cmd_duration + p["T_off"])
    return t, w, theta


def predict(model, p, t_cmd_duration, t_query):
    theta, w = _theta_omega(model, p, t_query, t_cmd_duration + p["T_off"])
    return theta, w


def total_angle(model, p, t_cmd_duration):
    theta, _ = _theta_omega(model, p, np.array([1e6]),
                            t_cmd_duration + p["T_off"])
    return float(theta[0])


PARAMS = {
    "A": ["T_on", "T_off", "tau", "w_max"],
    "B": ["T_on", "T_off", "alpha", "beta", "w_max"],
}
INIT = {
    "A": dict(T_on=1.00, T_off=0.45, tau=0.50, w_max=26.0),
    "B": dict(T_on=1.00, T_off=0.45, alpha=40.0, beta=45.0, w_max=26.0),
}
BOUNDS = {
    "T_on": (0.3, 2.0), "T_off": (0.0, 1.2), "tau": (0.05, 2.0),
    "w_max": (8.0, 60.0), "alpha": (5.0, 200.0), "beta": (5.0, 300.0),
}

# 수송지연은 구동계 성질이라 관측 도메인(bearing/yaw)과 무관하다.
# yaw 는 bearing 보다 7배 잡음이 커서 상승 시작점을 못 잡고 T_on 이 늦게
# 치우치며 w_max 와 맞바꿔진다. 그래서 bearing 적합에서 얻은 지연을 고정할 수
# 있게 해 둔다 (ROT_FIX_TON / ROT_FIX_TOFF).
_FIX = {}
for _k, _env in (("T_on", "ROT_FIX_TON"), ("T_off", "ROT_FIX_TOFF")):
    _v = os.environ.get(_env)
    if _v:
        _FIX[_k] = float(_v)
        BOUNDS[_k] = (float(_v) - 1e-6, float(_v) + 1e-6)
        INIT["A"][_k] = float(_v)
        INIT["B"][_k] = float(_v)


def fit(model, segs, per_segment_delay=False):
    names = PARAMS[model]
    x0 = [INIT[model][n] for n in names]
    lo = [BOUNDS[n][0] for n in names]
    hi = [BOUNDS[n][1] for n in names]
    if per_segment_delay:
        x0 = x0 + [INIT[model]["T_on"]] * len(segs)
        lo = lo + [BOUNDS["T_on"][0]] * len(segs)
        hi = hi + [BOUNDS["T_on"][1]] * len(segs)

    def unpack(x, k):
        p = {n: float(v) for n, v in zip(names, x[:len(names)])}
        if per_segment_delay:
            p["T_on"] = float(x[len(names) + k])
        return p

    def resid(x):
        out = []
        for k, s in enumerate(segs):
            p = unpack(x, k)
            tt = s.t - s.t_cmd
            m = (tt >= FIT_LO) & (tt <= s.cmd_duration + FIT_HI)
            th, _ = predict(model, p, s.cmd_duration, tt[m])
            out.append(s.angle[m] - th)
        return np.concatenate(out)

    best_r = None
    starts = [x0]
    if not per_segment_delay and "T_on" not in _FIX:
        for d_on in (0.85, 1.15):
            for d_off in (0.20, 0.70):
                alt = list(x0)
                alt[names.index("T_on")] = d_on
                alt[names.index("T_off")] = d_off
                starts.append(alt)
    for s0 in starts:
        r = least_squares(resid, s0, bounds=(lo, hi), method="trf",
                          diff_step=0.03, xtol=1e-12, ftol=1e-12,
                          max_nfev=20000)
        if best_r is None or r.cost < best_r.cost:
            best_r = r
    r = best_r
    res = r.fun
    params = {n: float(v) for n, v in zip(names, r.x[:len(names)])}
    extra = r.x[len(names):] if per_segment_delay else None
    return r, params, extra, float(np.sqrt(np.mean(res ** 2))), len(res)


# --------------------------------------------------------------- coast model
def coast_closed_form(model, p, w0, extra_hold=None):
    """STOP 시점 각속도 w0 일 때 이후 추가 회전량 [deg] 과 정지시간 [s]."""
    T_off = p["T_off"] if extra_hold is None else extra_hold
    if model == "A":
        tau, wm = p["tau"], p["w_max"]
        w1 = wm + (w0 - wm) * np.exp(-T_off / tau)
        d1 = wm * T_off + (w0 - wm) * tau * (1.0 - np.exp(-T_off / tau))
        return d1 + w1 * tau, T_off + 3.0 * tau, w1
    a, b, wm = p["alpha"], p["beta"], p["w_max"]
    w1 = min(wm, w0 + a * T_off)
    d1 = 0.5 * (w0 + w1) * T_off
    return d1 + w1 * w1 / (2.0 * b), T_off + w1 / b, w1


def main():
    segs_all = load_segments()
    seg_df = pd.read_csv(os.path.join(OUT, "segments.csv"))
    keep = set(seg_df[seg_df.usable].seg_id.tolist())
    segs = [s for k, s in enumerate(segs_all) if k in keep]
    ids = [k for k in range(len(segs_all)) if k in keep]
    print("fit segments: %d / %d" % (len(segs), len(segs_all)))

    report = {"n_segments_total": len(segs_all), "n_segments_used": len(segs),
              "pose_source": POSE_SOURCE, "angle_signal": ANGLE_SIGNAL,
              "fixed_params": dict(_FIX)}
    fits = {}
    for model in ("A", "B"):
        r, p, _, rms, n = fit(model, segs)
        fits[model] = dict(params=p, rms=rms, n=n)
        print("model %s: rms=%.3f deg  n=%d  %s" % (
            model, rms, n, "  ".join("%s=%.4f" % (k, v) for k, v in p.items())))

    best = min(fits, key=lambda m: fits[m]["rms"])
    p = fits[best]["params"]
    print("-> selected model %s" % best)

    # 세그먼트별 시동지연 자유화 -> 지연 분포
    r2, p2, delays, rms2, _ = fit(best, segs, per_segment_delay=True)
    delays = np.asarray(delays, float)
    print("per-segment T_on fit: rms=%.3f deg  mean=%.3f median=%.3f sd=%.3f "
          "min=%.3f max=%.3f" % (rms2, delays.mean(), np.median(delays),
                                 delays.std(ddof=1), delays.min(), delays.max()))

    onset = seg_df[seg_df.usable].startup_delay_s.values
    report["startup_delay_measured"] = dict(
        n=int(len(onset)), mean=float(onset.mean()),
        median=float(np.median(onset)), sd=float(onset.std(ddof=1)),
        p10=float(np.percentile(onset, 10)), p90=float(np.percentile(onset, 90)),
        min=float(onset.min()), max=float(onset.max()))
    report["startup_delay_fitted"] = dict(
        mean=float(delays.mean()), median=float(np.median(delays)),
        sd=float(delays.std(ddof=1)), min=float(delays.min()),
        max=float(delays.max()),
        per_segment={int(i): float(d) for i, d in zip(ids, delays)})
    report["models"] = fits
    report["selected_model"] = best
    report["selected_params"] = p
    report["fit_rms_deg"] = fits[best]["rms"]
    report["per_segment_delay_fit"] = dict(params=p2, rms=rms2)

    # ---- 예측 검증: 명령유지시간 -> 총 회전각
    rows = []
    for k, s in zip(ids, segs):
        row = seg_df[seg_df.seg_id == k].iloc[0]
        t, w, th = simulate(best, p, s.cmd_duration, s.cmd_duration + 6.0)
        rows.append(dict(
            seg_id=k, tag=s.tag, cmd=s.cmd, cmd_duration_s=s.cmd_duration,
            measured_total_deg=row.final_angle_deg,
            predicted_total_deg=float(th[-1]),
            measured_stop_angle=row.angle_at_stop_deg,
            predicted_stop_angle=float(np.interp(s.cmd_duration, t, th)),
            measured_stop_rate=row.rate_at_stop_deg_s,
            predicted_stop_rate=float(np.interp(s.cmd_duration, t, w)),
            measured_coast=row.coast_deg,
            predicted_coast=float(th[-1] - np.interp(s.cmd_duration, t, th))))
    val = pd.DataFrame(rows)
    val["total_err"] = val.predicted_total_deg - val.measured_total_deg
    val.to_csv(os.path.join(OUT, "validation.csv"), index=False,
               encoding="utf-8-sig")
    err = val.total_err.values
    report["total_angle_prediction"] = dict(
        rms_deg=float(np.sqrt(np.mean(err ** 2))),
        mae_deg=float(np.mean(np.abs(err))),
        max_abs_deg=float(np.max(np.abs(err))))
    print("\n총회전각 예측 검증")
    print(val[["tag", "cmd", "cmd_duration_s", "measured_total_deg",
               "predicted_total_deg", "total_err", "measured_coast",
               "predicted_coast"]].to_string(
        index=False, float_format=lambda v: "%8.2f" % v))
    print("RMS=%.2f deg  MAE=%.2f deg" % (
        report["total_angle_prediction"]["rms_deg"],
        report["total_angle_prediction"]["mae_deg"]))

    # ---- 좌/우 대칭성 검정
    for cmd in ("ROT_LEFT", "ROT_RIGHT"):
        sub = val[val.cmd == cmd]
        if len(sub):
            report.setdefault("direction_check", {})[cmd] = dict(
                n=int(len(sub)), bias_deg=float(sub.total_err.mean()),
                rms_deg=float(np.sqrt(np.mean(sub.total_err.values ** 2))))
    print("\n좌/우 대칭성: " + json.dumps(report.get("direction_check", {}),
                                     ensure_ascii=False))

    with io.open(os.path.join(OUT, "rotation_model_fit.json"), "w",
                 encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\nsaved -> " + os.path.join(OUT, "rotation_model_fit.json"))
    return report, best, p, segs, ids


if __name__ == "__main__":
    main()
