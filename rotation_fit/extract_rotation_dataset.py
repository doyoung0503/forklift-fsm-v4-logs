"""회전 로그 -> 적합용 데이터셋.

산출물 (out/):
  segments.csv        세그먼트별 요약 (지연/정지속도/관성회전/정지시간)
  milestones.csv      각도별 도달시간 (0.5 deg 간격, 명령기준/온셋기준 두 종류)
  traces.csv          세그먼트별 정규화 시계열 (곡선 적합 입력)
"""
from __future__ import annotations

import io
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
exec(io.open(os.path.join(HERE, "rot_common.py"), encoding="utf-8").read())

OUT = os.environ.get("ROT_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)

MILESTONES = np.round(np.arange(0.5, 20.0001, 0.5), 3)
SETTLE_LO, SETTLE_HI = 2.0, 3.5      # STOP 이후 정지 판정 창 [s]
ONSET_MIN_THRESHOLD_DEG = 0.25


def onset_time(seg: "Segment"):
    """명령 후 실제 회전이 시작된 시각 (선형 역추정). 실패 시 None."""
    thr = max(ONSET_MIN_THRESHOLD_DEG, 4.0 * seg.ref_noise)
    m = seg.t >= seg.t_cmd
    t, a = seg.t[m], seg.angle[m]
    if len(t) < 5:
        return None, thr
    # 잡음 스파이크 한 점으로 온셋이 잡히지 않도록, 임계 통과 후 회전이
    # 실제로 이어지는지(0.5 s 안에 2*thr 도달 + 그 사이 임계 아래로 안 떨어짐)
    # 확인한다. 재추론 데이터는 표본이 2배라 이 확인이 없으면 오검출이 난다.
    idx = None
    for i in range(1, len(t) - 1):
        if a[i] < thr:
            continue
        w = (t >= t[i]) & (t <= t[i] + 0.5)
        if int(w.sum()) < 3:
            continue
        if np.max(a[w]) >= 2.0 * thr and np.min(a[w]) >= 0.5 * thr:
            idx = i
            break
    if idx is None or idx == 0:
        return None, thr
    t_cross = float(np.interp(thr, [a[idx - 1], a[idx]], [t[idx - 1], t[idx]])) \
        if a[idx] > a[idx - 1] else float(t[idx])
    w = (t >= t_cross - 0.20) & (t <= t_cross + 0.45)
    if int(w.sum()) >= 3:
        slope, intercept = np.polyfit(t[w], a[w], 1)
        if slope > 0.5:
            t0 = -intercept / slope
            if seg.t_cmd - 0.3 <= t0 <= t_cross + 0.05:
                return float(t0), thr
    return t_cross, thr


def stop_rate(seg: "Segment", t_on):
    """STOP 시점의 순간 회전속도. 가속 중이므로 국소 2차 적합의 미분을 쓴다."""
    lo = seg.t_stop - 0.60 if t_on is None else max(t_on, seg.t_stop - 0.60)
    m = (seg.t >= lo) & (seg.t <= seg.t_stop + 1e-9)
    if int(m.sum()) >= 4:
        c = np.polyfit(seg.t[m] - seg.t_stop, seg.angle[m], 2)
        return float(c[1])            # d/dt at t_stop
    if int(m.sum()) >= 3:
        return float(np.polyfit(seg.t[m] - seg.t_stop, seg.angle[m], 1)[0])
    return None


def stall_gap(seg: "Segment", t_on, peak_rate):
    """온셋~STOP 구간에서 회전이 멈춘 최장 구간 길이 [s]."""
    if t_on is None or not np.isfinite(peak_rate) or peak_rate <= 0:
        return 0.0
    m = (seg.t >= t_on) & (seg.t <= seg.t_stop)
    t, a = seg.t[m], seg.angle[m]
    if len(t) < 4:
        return 0.0
    rate = np.diff(a) / np.diff(t)
    slow = rate < 0.15 * peak_rate
    best = run = 0.0
    for i, is_slow in enumerate(slow):
        run = run + (t[i + 1] - t[i]) if is_slow else 0.0
        best = max(best, run)
    return float(best)


def settle_stats(seg: "Segment"):
    """STOP 이후 최종각 / 완전정지 시각."""
    m = (seg.t >= seg.t_stop + SETTLE_LO) & (seg.t <= seg.t_stop + SETTLE_HI)
    if int(m.sum()) < 4:
        m = seg.t >= seg.t_stop + 1.2
        if int(m.sum()) < 4:
            return None, None, None
    final = float(np.median(seg.angle[m]))
    band = max(0.20, 4.0 * seg.ref_noise)
    post = seg.t >= seg.t_stop
    tp, ap = seg.t[post], seg.angle[post]
    # 이후 표본의 95% 가 밴드 안이면 정지로 본다. all() 로 하면 한참 뒤의
    # 단일 이상치 하나가 정지시각을 몇 초씩 밀어낸다.
    t_settled = None
    for i in range(len(tp)):
        rest = np.abs(ap[i:] - final) <= band
        if len(rest) >= 4 and float(np.mean(rest)) >= 0.95:
            t_settled = float(tp[i])
            break
    peak = float(np.max(ap)) if len(ap) else np.nan
    return final, t_settled, peak


def main():
    segs = load_segments()
    seg_rows, ms_rows, tr_rows = [], [], []
    for k, s in enumerate(segs):
        t_on, thr = onset_time(s)
        final, t_settled, peak = settle_stats(s)
        a_stop = value_at(s.t, s.angle, s.t_stop)
        w_stop = stop_rate(s, t_on)
        w_stop_lin = local_rate(s.t, s.angle, s.t_stop, window=0.45)
        # 명령 유지 중 최대 회전속도 (온셋 이후 ~ STOP)
        w_peak = np.nan
        if t_on is not None:
            m = (s.t >= t_on) & (s.t <= s.t_stop + 0.35)
            if int(m.sum()) >= 3:
                d = np.diff(s.angle[m]) / np.diff(s.t[m])
                w_peak = float(np.max(d)) if len(d) else np.nan

        pre = s.angle[(s.t >= s.t_cmd - 1.0) & (s.t <= s.t_cmd)]
        baseline_ok = bool(len(pre) >= 4
                           and np.max(np.abs(pre)) <= max(0.40, 5 * s.ref_noise))
        move_time = None if t_on is None else s.t_stop - t_on
        onset_ok = bool(move_time is not None and move_time >= 0.15)
        gap = stall_gap(s, t_on, w_peak)
        stall_ok = bool(gap <= 0.8)
        big_enough = bool((final or 0.0) >= 2.0)
        usable = bool(baseline_ok and onset_ok and stall_ok and big_enough)
        reason = ""
        if not baseline_ok:
            reason = "명령 직전 1s 정지 미확인(이전 동작 잔류)"
        elif t_on is None:
            reason = "온셋 검출 실패"
        elif not onset_ok:
            reason = "온셋이 STOP 이후(명령 유지시간 < 시동지연)"
        elif not stall_ok:
            reason = f"명령 유지 중 회전 정체 {gap:.1f}s"
        elif not big_enough:
            reason = "총 회전량 < 2 deg"

        seg_rows.append(dict(
            seg_id=k, tag=s.tag, step=s.step, cmd=s.cmd, kind=s.kind,
            sign=s.sign, range_m=float(np.median(s.rng)),
            bearing_ref_deg=s.ref_bearing, bearing_noise_deg=s.ref_noise,
            cmd_duration_s=s.cmd_duration,
            startup_delay_s=(None if t_on is None else t_on - s.t_cmd),
            onset_threshold_deg=thr,
            angle_at_stop_deg=a_stop,
            rate_at_stop_deg_s=w_stop,
            rate_at_stop_linfit_deg_s=w_stop_lin,
            stall_gap_s=gap,
            peak_rate_deg_s=w_peak,
            move_time_before_stop_s=(None if t_on is None else s.t_stop - t_on),
            final_angle_deg=final,
            peak_angle_deg=peak,
            coast_deg=(None if final is None else final - a_stop),
            overshoot_deg=(None if (final is None or peak is None) else peak - final),
            settle_time_after_stop_s=(None if t_settled is None else t_settled - s.t_stop),
            log_stop_rate_deg_s=s.diag.get("stop_measured_rate_deg_s"),
            log_post_stop_deg=s.end.get("post_stop_rotation_deg"),
            usable=usable, exclude_reason=reason,
        ))

        for A in MILESTONES:
            tc = interp_cross(s.t[s.t >= s.t_cmd], s.angle[s.t >= s.t_cmd], A)
            if tc is None:
                continue
            ms_rows.append(dict(
                seg_id=k, tag=s.tag, step=s.step, cmd=s.cmd, angle_deg=float(A),
                t_from_cmd_s=tc - s.t_cmd,
                t_from_onset_s=(None if t_on is None else tc - t_on),
                during_command=bool(tc <= s.t_stop),
                usable=usable,
            ))

        for ti, ai in zip(s.t, s.angle):
            tr_rows.append(dict(
                seg_id=k, tag=s.tag, cmd=s.cmd,
                t_from_cmd_s=ti - s.t_cmd,
                t_from_onset_s=(None if t_on is None else ti - t_on),
                angle_deg=ai, during_command=bool(ti <= s.t_stop),
                usable=usable,
            ))

    seg_df = pd.DataFrame(seg_rows)
    ms_df = pd.DataFrame(ms_rows)
    tr_df = pd.DataFrame(tr_rows)
    seg_df.to_csv(os.path.join(OUT, "segments.csv"), index=False, encoding="utf-8-sig")
    ms_df.to_csv(os.path.join(OUT, "milestones.csv"), index=False, encoding="utf-8-sig")
    tr_df.to_csv(os.path.join(OUT, "traces.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    cols = ["tag", "step", "cmd", "range_m", "cmd_duration_s", "startup_delay_s",
            "move_time_before_stop_s", "angle_at_stop_deg", "rate_at_stop_deg_s",
            "peak_rate_deg_s", "coast_deg", "final_angle_deg", "overshoot_deg",
            "settle_time_after_stop_s", "usable", "exclude_reason"]
    print(seg_df[cols].to_string(index=False,
          float_format=lambda v: f"{v:8.2f}"))
    print(f"\nsaved -> {OUT}")
    return seg_df, ms_df, tr_df


if __name__ == "__main__":
    main()
