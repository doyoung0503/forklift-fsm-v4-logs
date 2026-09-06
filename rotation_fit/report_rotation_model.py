# -*- coding: utf-8 -*-
"""적합 결과 리포트: 각도별 도달시간 표, 교차검증, 그림 6종, 마크다운 요약."""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("ROT_OUT", os.path.join(HERE, "out"))
sys.path.insert(0, os.path.join(
    os.path.dirname(HERE), "extracted", "depth_cam", "calib", "fsm_v4"))

exec(io.open(os.path.join(HERE, "rot_common.py"), encoding="utf-8").read())
_fit_src = io.open(os.path.join(HERE, "fit_rotation_model.py"),
                   encoding="utf-8").read().split("def main()")[0]
_fit_src = _fit_src.replace(
    'exec(io.open(os.path.join(HERE, "rot_common.py"), encoding="utf-8").read())', "")
exec(_fit_src)

from rotation_model import RotationResponse  # noqa: E402

MILESTONES = np.round(np.arange(0.5, 20.0001, 0.5), 3)
PLOT_KW = dict(dpi=120, bbox_inches="tight")


def response_from_params(p, delay_sd=0.0, resid_sd=0.0, source=""):
    return RotationResponse(
        startup_delay_sec=p["T_on"], stop_delay_sec=p["T_off"],
        accel_deg_s2=p["alpha"], decel_deg_s2=p["beta"],
        max_rate_deg_s=p["w_max"], startup_delay_sd_sec=delay_sd,
        residual_sd_deg=resid_sd, source=source)


# --------------------------------------------------------------------- table
def milestone_table(ms, seg_df, R):
    """각도별 도달시간 요약 (명령 기준 / 온셋 기준)."""
    ms = ms[ms.usable].copy()
    rows = []
    for A in MILESTONES:
        sub = ms[np.isclose(ms.angle_deg, A)]
        if len(sub) == 0:
            continue
        during = sub[sub.during_command]
        tc = sub.t_from_cmd_s.values.astype(float)
        to = sub.t_from_onset_s.dropna().values.astype(float)
        # 적합 곡선에서의 예측 도달시간 (명령 유지 중)
        pred_cmd = np.nan
        if R.total_angle(8.0) >= A:
            lo, hi = 0.0, 8.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if R.angle_at(mid) < A:
                    lo = mid
                else:
                    hi = mid
            pred_cmd = 0.5 * (lo + hi)
        rows.append(dict(
            angle_deg=A, n=len(sub), n_during_command=int(len(during)),
            t_cmd_mean=tc.mean(), t_cmd_median=float(np.median(tc)),
            t_cmd_sd=float(tc.std(ddof=1)) if len(tc) > 1 else np.nan,
            t_onset_mean=to.mean() if len(to) else np.nan,
            t_onset_median=float(np.median(to)) if len(to) else np.nan,
            t_onset_sd=float(to.std(ddof=1)) if len(to) > 1 else np.nan,
            t_onset_model=(pred_cmd - R.startup_delay_sec
                           if np.isfinite(pred_cmd) else np.nan),
            t_cmd_model=pred_cmd))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- LOO
def loo_validation(segs, ids, seg_df, model):
    rows = []
    for i in range(len(segs)):
        train = [s for j, s in enumerate(segs) if j != i]
        _, p, _, _, _ = fit(model, train)
        s = segs[i]
        t, w, th = simulate(model, p, s.cmd_duration, s.cmd_duration + 6.0)
        row = seg_df[seg_df.seg_id == ids[i]].iloc[0]
        rows.append(dict(seg_id=ids[i], tag=s.tag, cmd=s.cmd,
                         cmd_duration_s=s.cmd_duration,
                         measured_total_deg=row.final_angle_deg,
                         loo_pred_total_deg=float(th[-1]),
                         err=float(th[-1]) - row.final_angle_deg))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- plots
def plot_response(segs, seg_df, ids, R, model, p):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for s, sid in zip(segs, ids):
        d = seg_df[seg_df.seg_id == sid].iloc[0]
        tt = s.t - s.t_cmd
        m = (tt >= -0.5) & (tt <= s.cmd_duration + 2.5)
        axes[0].plot(tt[m], s.angle[m], ".", ms=3, alpha=0.55,
                     color="#1f77b4" if s.cmd == "ROT_RIGHT" else "#d62728")
        on = d.startup_delay_s
        dur = tt[m] - on
        keep = (dur >= 0) & (tt[m] <= s.cmd_duration)
        axes[1].plot(dur[keep], s.angle[m][keep], ".", ms=4, alpha=0.6,
                     color="#1f77b4" if s.cmd == "ROT_RIGHT" else "#d62728")
    grid = np.linspace(0, 3.2, 400)
    axes[0].plot(grid, [R.angle_at(g) for g in grid], "k-", lw=2,
                 label="fitted (command held)")
    axes[1].plot(grid, [R.angle_at(g + R.startup_delay_sec) for g in grid],
                 "k-", lw=2, label="fitted")
    axes[0].axvline(R.startup_delay_sec, color="g", ls="--",
                    label="T_on = %.2f s" % R.startup_delay_sec)
    axes[0].set_xlabel("time since ROT command [s]")
    axes[1].set_xlabel("time since motion onset [s]")
    for ax in axes:
        ax.set_ylabel("rotation [deg]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(-1, 22)
    axes[0].set_title("all samples (incl. post-STOP coast)")
    axes[1].set_title("command-held phase only, aligned on onset")
    fig.suptitle("rotation response curve (deflection 30, L/R pooled)")
    fig.savefig(os.path.join(OUT, "02_response_curve.png"), **PLOT_KW)
    plt.close(fig)


def plot_milestones(mt, R):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, lbl in ((axes[0], "t_cmd", "from ROT command"),
                         (axes[1], "t_onset", "from motion onset")):
        m = mt[np.isfinite(mt[key + "_mean"])]
        ax.errorbar(m[key + "_median"], m.angle_deg,
                    xerr=m[key + "_sd"].fillna(0), fmt="o", ms=4, lw=1,
                    capsize=2, color="#1f77b4", label="log median +/- sd")
        mm = mt[np.isfinite(mt[key + "_model"])]
        ax.plot(mm[key + "_model"], mm.angle_deg, "k-", lw=2,
                label="fitted curve")
        ax.set_xlabel("time [s] (%s)" % lbl)
        ax.set_ylabel("rotation [deg]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("angle milestones: reach time per 0.5 deg step")
    fig.savefig(os.path.join(OUT, "03_milestones.png"), **PLOT_KW)
    plt.close(fig)


def plot_coast(seg_df, R):
    d = seg_df[seg_df.usable]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    w = np.linspace(0, max(16.0, R.max_rate_deg_s * 1.2), 200)
    axes[0].plot(w, [R.coast(x)[0] for x in w], "k-", lw=2, label="model")
    axes[0].plot(d.rate_at_stop_deg_s, d.coast_deg, "o", ms=6,
                 color="#d62728", label="measured (quadratic-fit stop rate)")
    axes[0].plot(d.rate_at_stop_linfit_deg_s, d.coast_deg, "s", ms=5,
                 mfc="none", color="#1f77b4", label="measured (0.45 s linear fit)")
    axes[0].set_xlabel("rotation rate at STOP command [deg/s]")
    axes[0].set_ylabel("rotation added after STOP [deg]")
    axes[0].set_title("inertia / stop-delay curve")
    axes[1].plot(w, [R.coast(x)[1] for x in w], "k-", lw=2, label="model")
    axes[1].plot(d.rate_at_stop_deg_s, d.settle_time_after_stop_s, "o", ms=6,
                 color="#d62728", label="measured")
    axes[1].set_xlabel("rotation rate at STOP command [deg/s]")
    axes[1].set_ylabel("STOP -> fully stopped [s]")
    axes[1].set_title("settle time")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "04_coast.png"), **PLOT_KW)
    plt.close(fig)


def plot_delay(seg_df, report):
    d = seg_df[seg_df.usable].startup_delay_s.values
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(d, bins=np.arange(0.7, 1.45, 0.05), color="#1f77b4", alpha=0.8,
            edgecolor="k")
    ax.axvline(np.median(d), color="r", lw=2,
               label="median %.3f s" % np.median(d))
    ax.axvline(d.mean(), color="orange", lw=2, ls="--",
               label="mean %.3f s (sd %.3f)" % (d.mean(), d.std(ddof=1)))
    ax.axvline(1.516, color="k", lw=1.5, ls=":",
               label="config ROT_STARTUP_DELAY_SEC = 1.516")
    ax.set_xlabel("startup delay: ROT command -> motion onset [s]")
    ax.set_ylabel("segments")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "05_startup_delay.png"), **PLOT_KW)
    plt.close(fig)


def plot_control_curve(R):
    tgt = np.linspace(1.0, 20.0, 200)
    hold = np.array([R.command_seconds(t) for t in tgt])
    sd = np.array([R.angle_uncertainty_deg(h) for h in hold])
    stop_a = np.array([R.angle_at(h) for h in hold])
    coast = tgt - stop_a
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(tgt, hold, lw=2)
    axes[0].axhline(R.startup_delay_sec, color="g", ls="--",
                    label="T_on = %.2f s" % R.startup_delay_sec)
    axes[0].set_xlabel("target rotation [deg]")
    axes[0].set_ylabel("ROT command hold [s]")
    axes[0].set_title("target -> command time")
    axes[1].plot(tgt, stop_a, lw=2, label="angle when STOP is written")
    axes[1].plot(tgt, coast, lw=2, label="coast after STOP")
    axes[1].plot(tgt, tgt, "k--", lw=1, label="target")
    axes[1].set_xlabel("target rotation [deg]")
    axes[1].set_ylabel("[deg]")
    axes[1].set_title("pre-emptive STOP split")
    axes[2].fill_between(tgt, tgt - sd, tgt + sd, alpha=0.3, color="#1f77b4",
                         label="+/-1 sigma from startup-delay jitter")
    axes[2].plot(tgt, tgt, "k-", lw=2)
    axes[2].set_xlabel("target rotation [deg]")
    axes[2].set_ylabel("expected achieved [deg]")
    axes[2].set_title("open-loop accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "06_control_curve.png"), **PLOT_KW)
    plt.close(fig)


def plot_fit_check(val, loo):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 22], [0, 22], "k--", lw=1)
    ax.plot(val.measured_total_deg, val.predicted_total_deg, "o", ms=7,
            label="in-sample (RMS %.2f deg)"
            % np.sqrt(np.mean(val.total_err ** 2)))
    ax.plot(loo.measured_total_deg, loo.loo_pred_total_deg, "s", ms=6,
            mfc="none", label="leave-one-out (RMS %.2f deg)"
            % np.sqrt(np.mean(loo.err ** 2)))
    ax.set_xlabel("measured total rotation [deg]")
    ax.set_ylabel("predicted total rotation [deg]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("total rotation prediction")
    fig.savefig(os.path.join(OUT, "07_fit_check.png"), **PLOT_KW)
    plt.close(fig)


def main():
    seg_df = pd.read_csv(os.path.join(OUT, "segments.csv"))
    ms = pd.read_csv(os.path.join(OUT, "milestones.csv"))
    val = pd.read_csv(os.path.join(OUT, "validation.csv"))
    with io.open(os.path.join(OUT, "rotation_model_fit.json"),
                 encoding="utf-8") as fh:
        report = json.load(fh)

    model = report["selected_model"]
    p = report["selected_params"]
    R = response_from_params(
        p, delay_sd=report["startup_delay_measured"]["sd"],
        resid_sd=report["total_angle_prediction"]["rms_deg"],
        source=report.get("pose_source", ""))

    segs_all = load_segments()
    keep = set(seg_df[seg_df.usable].seg_id.tolist())
    segs = [s for k, s in enumerate(segs_all) if k in keep]
    ids = [k for k in range(len(segs_all)) if k in keep]

    mt = milestone_table(ms, seg_df, R)
    mt.to_csv(os.path.join(OUT, "milestone_table.csv"), index=False,
              encoding="utf-8-sig")

    loo = loo_validation(segs, ids, seg_df, model)
    loo.to_csv(os.path.join(OUT, "loo_validation.csv"), index=False,
               encoding="utf-8-sig")
    report["loo"] = dict(rms_deg=float(np.sqrt(np.mean(loo.err ** 2))),
                         mae_deg=float(np.mean(np.abs(loo.err))),
                         max_abs_deg=float(np.max(np.abs(loo.err))))

    plot_response(segs, seg_df, ids, R, model, p)
    plot_milestones(mt, R)
    plot_coast(seg_df, R)
    plot_delay(seg_df, report)
    plot_control_curve(R)
    plot_fit_check(val, loo)

    ctl = pd.DataFrame([dict(
        target_deg=t, hold_sec=R.command_seconds(t),
        stop_angle_deg=R.angle_at(R.command_seconds(t)),
        stop_rate_deg_s=R.rate_at(R.command_seconds(t)),
        coast_deg=t - R.angle_at(R.command_seconds(t)),
        settle_sec=R.settle_sec(R.command_seconds(t)),
        sd_deg=R.angle_uncertainty_deg(R.command_seconds(t)))
        for t in np.arange(2.0, 20.5, 0.5)])
    ctl.to_csv(os.path.join(OUT, "control_table.csv"), index=False,
               encoding="utf-8-sig")

    with io.open(os.path.join(OUT, "rotation_model_fit.json"), "w",
                 encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("LOO: rms=%.2f mae=%.2f max=%.2f deg" % (
        report["loo"]["rms_deg"], report["loo"]["mae_deg"],
        report["loo"]["max_abs_deg"]))
    print(mt[["angle_deg", "n", "t_cmd_median", "t_cmd_sd", "t_onset_median",
              "t_onset_sd", "t_onset_model"]].head(45).to_string(
        index=False, float_format=lambda v: "%7.3f" % v))
    print("\nfigures + tables -> " + OUT)


if __name__ == "__main__":
    main()
