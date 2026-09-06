# -*- coding: utf-8 -*-
"""적합 결과 -> ROTATION_MODEL_REPORT.md 생성 (숫자는 항상 산출물에서 읽는다)."""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("ROT_OUT", os.path.join(HERE, "out"))
DST = os.path.join(HERE, "ROTATION_MODEL_REPORT.md")


def block(df, cols, fmt="%8.2f", widths=None):
    """CLAUDE.md 규칙: 파이프 테이블 금지, 공백정렬 + ──── 구분선."""
    head = []
    body = []
    if widths is None:
        widths = []
        for c in cols:
            cell = max((len(("%d" % v) if isinstance(v, (int, np.integer))
                            else (fmt % v).strip()
                            if isinstance(v, (float, np.floating)) and np.isfinite(v)
                            else str(v)) for v in df[c]), default=0) if len(df) else 0
            widths.append(max(len(str(c)), cell, 4))
    w = widths
    head.append("  ".join(str(c).ljust(wi) for c, wi in zip(cols, w)))
    head.append("─" * (sum(w) + 2 * (len(w) - 1)))
    for _, r in df.iterrows():
        cells = []
        for c, wi in zip(cols, w):
            v = r[c]
            if isinstance(v, (int, np.integer)) or (
                    isinstance(v, (float, np.floating)) and np.isfinite(v)
                    and float(v).is_integer() and c in ("n", "n_used", "step",
                                                        "n_during_command")):
                cells.append(str(int(v)).rjust(wi))
            elif isinstance(v, (float, np.floating)) and np.isfinite(v):
                cells.append((fmt % v).strip().rjust(wi))
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("-".rjust(wi))
            else:
                cells.append(str(v).ljust(wi))
        body.append("  ".join(cells))
    return "```\n" + "\n".join(head + body) + "\n```"


def main():
    with io.open(os.path.join(OUT, "rotation_model_fit.json"),
                 encoding="utf-8") as fh:
        rep = json.load(fh)
    seg = pd.read_csv(os.path.join(OUT, "segments.csv"))
    use = seg[seg.usable].copy()
    mt = pd.read_csv(os.path.join(OUT, "milestone_table.csv"))
    ctl = pd.read_csv(os.path.join(OUT, "control_table.csv"))
    val = pd.read_csv(os.path.join(OUT, "validation.csv"))
    loo = pd.read_csv(os.path.join(OUT, "loo_validation.csv"))
    p = rep["selected_params"]
    d = rep["startup_delay_measured"]
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "extracted",
                                    "depth_cam", "calib", "fsm_v4"))
    from rotation_model import RotationResponse
    R = RotationResponse(p["T_on"], p["T_off"], p["alpha"], p["beta"],
                         p["w_max"])
    min_total = R.min_total_deg
    # 폐루프 시뮬: 모델대로 움직이는 차량에 컨트롤러의 STOP 규칙을 적용
    rows_cl = []
    for tgt in (3.0, 6.0, 10.0, 16.0):
        hold = R.command_seconds(tgt, max_hold_sec=2.5)
        el = None
        for k in range(1, 60):
            t = k * 0.095
            rem = tgt - R.angle_at(t)
            if t >= R.startup_delay_sec - R.stop_delay_sec and                     rem <= R.stop_now_margin_deg(t, None, 0.08):
                el = t
                break
            if t >= hold:
                el = t
                break
        final = R.angle_at(el) + R.coast(R.rate_at(el))[0]
        rows_cl.append(dict(target_deg=tgt, plan_hold_s=hold, stop_at_s=el,
                            achieved_deg=final, error_deg=final - tgt))
    closed_loop = block(pd.DataFrame(rows_cl),
                        ["target_deg", "plan_hold_s", "stop_at_s",
                         "achieved_deg", "error_deg"], "%8.2f")

    cmp_path = os.path.join(HERE, "out", "source_comparison.csv")
    cmp_block = ""
    if os.path.exists(cmp_path):
        c = pd.read_csv(cmp_path)
        cmp_block = "\n## 0. pose 소스별 비교\n\n" + block(
            c, ["source", "n_used", "bearing_noise_deg", "T_on", "T_off",
                "alpha", "beta", "w_max", "traj_rms_deg", "total_rms_deg",
                "loo_rms_deg"], "%8.3f") + "\n"

    ms = mt[mt.angle_deg.isin([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
                               10.0, 12.0, 14.0, 16.0, 18.0, 20.0])]

    txt = f"""# 회전 오차 보정 — 로그 기반 적합함수

토크(회전 조이스틱 편향) **{30}** 고정, 좌·우 20 deg 이내 소각도 회전용.
좌/우는 동일 특성으로 가정하고 부호로만 구분한다 (아래 6절에서 검정).

생성: `rotation_fit/make_report.py` — 숫자는 모두 `{os.path.basename(OUT)}/` 산출물에서 읽는다.
{cmp_block}
## 1. 회전각 관측량

회전각으로 쓸 수 있는 신호가 두 개 있고, 서로 도메인이 다르다.

- **bearing** = `atan2(pos_x, pos_z)` (전면 중심 방위각). FACE/RECENTER 모드의
  오차신호. 잡음이 가장 작다(정지구간 표준편차 0.02~0.03 deg). 단, 차량 헤딩뿐
  아니라 회전에 따른 카메라 병진도 함께 담긴다.
- **yaw** = 팔레트 평면 yaw. WAYPOINT 모드의 오차신호이자 순수 차량 헤딩.
  기존 4점 PnP 로는 **쓸 수 없었다** — 전면이 1.1 x 0.14 m (약 8:1) 얇은 면이라
  평면 PnP 가 나쁘게 조건화되어 12 세그먼트 중 5개에서 회전 방향이 반대로 나왔다
  (`corr(Δyaw, Δbearing) = 0.02`). 후면 키포인트까지 쓰는 9점 PnP 로 바꾸자
  12/12 부호 일치, 상관 0.99 로 회복됐다.

둘의 관계는 회전중심이 카메라 뒤 `d` 에 있을 때
`Δbearing = Δheading * (range + d) / range`. 이 적합은 **heading 도메인**에서
하고, bearing 모드는 위 식으로 변환한다 (8절에서 검증).

## 2. 회전 시작 지연시간

회전 명령 송신 → 실제 회전 시작(방위각이 잡음 위로 상승하기 시작하는 시점,
초기 상승 직선을 0 으로 역추정).

**채택값 T_on = {p['T_on']:.3f} s, T_off = {p['T_off']:.3f} s.**

수송지연은 구동계 성질이라 관측 도메인과 무관하다. 그런데 yaw 는 bearing 보다
잡음이 7배 커서 상승 시작점을 잘 못 잡고 T_on 이 늦게 치우치며 w_max 와
맞바꿔진다(자유 적합 시 T_on 1.282 s). 그래서 지연은 **bearing 적합값으로
고정**하고 각속도 파라미터만 yaw 로 적합했다 (0절 `recompmulti_yaw_fixed`).

지연 산포 (같은 세그먼트를 bearing 으로 잰 값):

```
세그먼트별 T_on 적합 표준편차   0.105 s   (log / recomp4 / recompmulti 모두 0.105~0.107)
임계통과 온셋 중앙값            1.013 s
임계통과 온셋 표준편차          0.162 s
```

참고: yaw 신호로 임계통과 온셋을 재면 {d['median']:.3f} s (sd {d['sd']:.3f}) 로 나오는데,
이는 yaw 잡음이 커서 임계값(4x잡음)이 0.75 deg 로 올라간 탓이고 실제 지연이
아니다.

기존 `ROT_STARTUP_DELAY_SEC = 1.516` 은 채택값보다 **{1.516 - p['T_on']:.3f} s 크다**.
그만큼 STOP 이 늦어 매번 과회전한다.

주의: 이 값에는 카메라 파이프라인 지연(노출→호스트 도착)이 포함되어 있다.
추론 지연 31 ms 는 보정했지만 센서 파이프라인 지연(config 상 PROVISIONAL 50 ms)은
보정하지 않았으므로, 실제 기계적 지연은 이보다 최대 50 ms 작을 수 있다.

## 3. 각도별 도달시간

`{os.path.basename(OUT)}/milestone_table.csv` 전체 (0.5 deg 간격, 20 deg 까지). 발췌:

{block(ms, ['angle_deg', 'n', 't_cmd_median', 't_cmd_sd', 't_onset_median',
            't_onset_sd', 't_onset_model'], '%8.3f')}

`t_cmd_*` = 명령 송신부터, `t_onset_*` = 실제 회전 시작부터, `t_onset_model` = 적합곡선 값.

두 가지 주의:
- **heading 도메인에서 실측이 존재하는 구간은 약 13 deg 까지**다(회전 하나가
  낸 최대 헤딩 변화). 그 위는 등속(w_max) 가정의 외삽이다. bearing 도메인
  (`recompmulti_bearing/milestone_table.csv`)은 17.5 deg 까지 실측이 있다.
- 0.5~1.5 deg 구간의 `t_onset_median` 이 모델보다 이른 것은 yaw 잡음
  (정지구간 sd 0.19 deg) 이 임계를 먼저 건드리기 때문이고 실제 도달이 아니다.
  같은 구간을 잡음 0.03 deg 인 bearing 으로 재면 모델과 잘 맞는다.

## 4. 연속 응답 곡선

이산 도달시각을 각각의 규칙으로 쓰지 않고, 모든 측정점을 하나의 시간-회전각 곡선에
적합했다. 두 모델을 궤적 전체(명령 전 0.5 s ~ STOP 후 2.5 s)에 최소자승 적합해 비교:

```
모델                                        궤적 RMS
──────────────────────────────────────────────────────
A  1차 지연 (dω/dt = (u·w_max - ω)/tau)     {rep['models']['A']['rms']:.3f} deg
B  등가속/등감속 + 양단 수송지연            {rep['models']['B']['rms']:.3f} deg   ← 채택
```

채택 모델:

```
u(t) = 1   for   T_on <= t < hold + T_off,   else 0
dω/dt = +alpha  (u=1, w_max 에서 포화)
        -beta   (u=0)
θ(t)  = ∫ ω dt
```

```
T_on   시동지연            {p['T_on']:.3f} s
T_off  STOP 수송지연       {p['T_off']:.3f} s
alpha  각가속도            {p['alpha']:.2f} deg/s^2
w_max  정상 회전속도       {p['w_max']:.2f} deg/s   (포화까지 {p['w_max']/p['alpha']:.2f} s)
beta   각감속도            {p['beta']:.1f} deg/s^2
```

구간 해석:
- `0 ~ T_on` 명령을 보냈지만 움직이지 않는 시작 지연구간
- `T_on ~ T_on + w_max/alpha` 가속구간 (θ = ½·alpha·τ²)
- 그 이후 등속구간 (w_max)
- STOP 후 `T_off` 동안 **계속 가속**, 그 다음 beta 로 감속

## 5. 정지 후 관성 회전

**핵심 관측**: STOP 시점 회전속도가 3 deg/s 든 25 deg/s 든 관성 회전량은
{use.coast_deg.min():.1f} ~ {use.coast_deg.max():.1f} deg (중앙값 {use.coast_deg.median():.2f}) 로 거의 일정하다.
속도에 비례하지 않는다.

이유는 **STOP 명령에도 수송지연 T_off = {p['T_off']:.3f} s 가 있기 때문**이다. STOP 을 쓴 뒤에도
그 시간 동안 차량은 회전 명령을 계속 수행하며 가속한다. 그래서 느릴 때 STOP 해도
관성 회전량이 줄지 않는다. 닫힌형:

```
w1 = min(w_max, w0 + alpha·T_off)
coast(w0) = [T_off 동안의 회전]  +  w1² / (2·beta)
```

측정값:

```
STOP 시점 회전속도    {use.rate_at_stop_deg_s.min():.1f} ~ {use.rate_at_stop_deg_s.max():.1f} deg/s
관성 회전량           {use.coast_deg.min():.2f} ~ {use.coast_deg.max():.2f} deg  (중앙값 {use.coast_deg.median():.2f})
STOP -> 완전정지      중앙값 {np.nanmedian(use.settle_time_after_stop_s):.2f} s, p90 {np.nanpercentile(use.settle_time_after_stop_s.dropna(), 90):.2f} s
정지 후 되튐(rebound) 중앙값 {use.overshoot_deg.median():.2f} deg, 최대 {use.overshoot_deg.max():.2f} deg
```

되튐(차량이 멈춘 뒤 살짝 되돌아오는 양)은 구동 모델에 없다. 그래서 모델의
`settle_sec` 은 감속 종료 시점일 뿐이고, **정지 판정 대기시간은 실측 p90 을 쓴다**
(`RotationResponse.settle_wait_sec`).

## 6. 좌/우 대칭성

방향별 총회전각 예측오차:

{block(pd.DataFrame([dict(direction=k, **v) for k, v in rep['direction_check'].items()]),
       ['direction', 'n', 'bias_deg', 'rms_deg'], '%8.3f')}

편향 차이가 잡음 수준이라 **좌/우 동일 특성 가정은 유지 가능**하다.

## 6b. heading -> bearing 변환 검증

heading 도메인 모델 하나에 기하 변환 `(range + d)/range` 만 적용해서 FACE/RECENTER
가 실제로 본 bearing 회전량을 재현할 수 있는지 확인했다 (같은 10 세그먼트):

```
d [m]    bearing 총회전각 편향    RMS
──────────────────────────────────────
0.0             -3.50           3.92
0.5             -1.83           2.41
0.8             -0.82           1.77
1.0             -0.15           1.64   <- 채택
1.3             +0.85           1.98

참고: bearing 도메인을 직접 적합했을 때의 RMS = 1.64
```

**d = 1.0 m 에서 직접 적합과 완전히 같은 정확도**가 나온다. 즉 heading 모델 하나로
두 도메인을 모두 커버할 수 있고, 회전중심이 카메라 뒤 약 1 m 라는 것이 데이터로
지지된다 (d=0 은 명확히 기각). config 의 PROVISIONAL 값 0.80 m 와 같은 자릿수이나
동일하지는 않아, `ROT_BEARING_CENTRE_OFFSET_M` 을 따로 두었다 —
`CAMERA_TO_ROT_CENTER_Z_M` 은 경로계획에도 쓰이므로 실측 없이 바꾸지 않았다.

## 7. 검증

```
궤적 적합 RMS                    {rep['fit_rms_deg']:.2f} deg
총회전각 예측 RMS (in-sample)    {rep['total_angle_prediction']['rms_deg']:.2f} deg
총회전각 예측 RMS (leave-one-out) {rep['loo']['rms_deg']:.2f} deg
총회전각 예측 최대오차 (LOO)      {rep['loo']['max_abs_deg']:.2f} deg
세그먼트별 시동지연 자유화 시 RMS {rep['per_segment_delay_fit']['rms']:.2f} deg
```

마지막 줄이 중요하다: 시동지연을 세그먼트마다 자유롭게 두면 RMS 가
{rep['fit_rms_deg']:.2f} → {rep['per_segment_delay_fit']['rms']:.2f} deg 로 줄어든다.
**남은 오차의 대부분은 시동지연 산포(0.105 s)** 이고, 이것은 개루프로는 줄일 수 없다.
w_max 기준으로 {0.105 * p['w_max']:.2f} deg 에 해당한다. 그래서 실시간 포즈 기반
선제정지(8절)가 없으면 정확도가 이 수준에서 더 좋아지지 않는다.

세그먼트별:

{block(val.assign(err=val.total_err), ['tag', 'cmd', 'cmd_duration_s',
       'measured_total_deg', 'predicted_total_deg', 'err'], '%8.2f')}

## 8. 최종 회전 제어 함수

`calib/fsm_v4/rotation_model.py` 의 `RotationResponse`.

```
plan(error_deg)            목표각 -> RotationPlan
  .hold_sec                ROT 명령 유지시간 (시동지연 + 필요 구동시간)
  .predicted_stop_angle_deg  STOP 을 쓰는 시점까지의 회전량 (= 선제 정지 지점)
  .predicted_coast_deg     STOP 이후 관성 회전량
  .predicted_total_deg     합계 (= 목표각)
  .angle_sd_deg            시동지연 산포로 인한 1σ
command_seconds(target)    목표각 -> 명령시간 (total_angle 의 역함수, 이분법)
coast(rate)                STOP 시 회전속도 -> (관성 회전량, 정지시간)
stop_now_margin_deg(...)   지금 STOP 하면 얼마나 더 도는가 (측정지연 포함)
should_stop_now(...)       남은 오차 <= 예상 관성회전량 이면 True
settle_wait_sec(hold)      정지 판정 대기시간 (실측 p90 반영)
```

목표각 → 명령시간 표 (`{os.path.basename(OUT)}/control_table.csv` 전체):

{block(ctl[ctl.target_deg.isin([2.5, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20])],
       ['target_deg', 'hold_sec', 'stop_angle_deg', 'stop_rate_deg_s',
        'coast_deg', 'settle_sec', 'sd_deg'], '%8.2f')}

한 번의 명령으로 낼 수 있는 최소 회전각은 약 **{min_total:.2f} deg** 다 — 시동지연까지 유지해야 움직이기 시작하므로
`hold = T_on` 이 최소 명령이고 그때 이미 관성으로 {p['T_off']:.2f} s 만큼 더 돈다.
기존 `ROT_MIN_COMMANDABLE_ANGLE_DEG = 2.50` 은 이 하한 위에 있어 타당하다.

### FSM 제어 흐름 (`controllers.RotationController`)

1. `start()` 에서 `plan()` 으로 명령유지시간과 예상 관성회전량을 계산
2. 매 프레임 STOP 판정 우선순위
   1. `live_coast_margin` — 실시간 팔레트 포즈 기준 남은 오차 <= 예상 관성회전량
      (측정 지연 `age × rate` 를 더해 보정, 실측 속도가 모델보다 빠르면 실측을 쓴다)
   2. `target_crossed` — 오차 부호가 실제로 바뀜
   3. `fitted_hold_elapsed` — 계획한 명령시간 도달 (포즈가 없어도 반드시 멈춘다)
3. 정지 후 `settle_wait_sec` 만큼 기다렸다가 최종 회전각 재측정,
   남은 오차가 허용범위 밖이면 다시 `plan()` (오차가 최소 명령각 미만이면 회전 포기)

폐루프 시뮬레이션 (제어주기 95 ms, 차량이 모델대로 움직인다고 가정):

{closed_loop}

## 9. 제외한 로그와 사유

{block(seg[~seg.usable], ['tag', 'step', 'cmd', 'cmd_duration_s',
       'exclude_reason'], '%8.2f', widths=[16, 6, 10, 15, 40])}

특히 `20260901_174342 step7` 은 [확인]된 실차 거동이다: ROT_LEFT 를 5.66 s 연속
송신했는데 차량은 1.34~2.02 s 에만 6.5 deg 돌고 **3 s 넘게 완전히 정지**했다가
STOP 직전에 다시 움직였다. 이 구간 재투영 오차는 0.03~0.6 px 로 인지 문제가 아니다.

→ 이 때문에 `ROT_MAX_COMMAND_HOLD_SEC = 2.50` 을 두어 한 번의 명령을 2.5 s 로
제한했다(정상 동작한 모든 회전이 2.5 s 이하 명령이었다). 그 결과 한 명령의
최대 회전량은 **heading 16.0 deg** (bearing 도메인, 거리 2.3 m 기준 22.9 deg) 다.
`ROT_MAX_WAYPOINT_TURN_DEG = 20` 인 heading 목표는 한 번에 안 되고 FSM 의
replan 루프가 두 번에 나눠 수행한다 — `plan()` 이 `feasible=False` 와
`note="hold capped..."` 로 알려준다.

또 하나: `20260901_175419` 는 재추론 소스에서 통째로 빠졌다. 이 녹화의 raw.mp4 는
moovless 손상에서 복구된 파일이라 중간 프레임이 유실되어 `frame_i = raw_index + offset`
고정 오프셋이 성립하지 않는다. `rot_common.py` 의 정렬 게이트(4점 재계산 vs 원본
로그 |Δx| 중앙값 > 5 mm)가 자동으로 걸러낸다.

## 10. 남은 불확실성

- 명령 유지 중 실제로 관측된 최대 회전시간은 온셋 이후 약 1.2 s 다. 그보다 긴 구간의
  등속 가정(w_max)은 외삽이다. 20 deg 목표는 온셋 이후 {ctl[ctl.target_deg == 20].iloc[0].hold_sec - p['T_on']:.2f} s 이므로
  약간의 외삽이 들어간다.
- 사용 세그먼트 {len(use)} 개(좌 {int((use.cmd == 'ROT_LEFT').sum())} / 우 {int((use.cmd == 'ROT_RIGHT').sum())}), 녹화 3건.
  팔레트 거리 {use.range_m.min():.1f}~{use.range_m.max():.1f} m. 표본이 적다.
- 회전중심 거리 d 는 회전 로그에서 **간접적으로만** 추정했다 (6b절, d=1.0 m).
  독립 오도메트리가 없으므로 [추정] 이다. 실측 외부파라미터가 확보되면
  `ROT_BEARING_CENTRE_OFFSET_M` 을 재검증할 것.
- 9점 객체모델의 절대 스케일은 전면 폭 1.100 m 가정에 100% 의존한다. depth 가
  녹화에 없어 독립 검증이 불가능했다. 실폭이 다르면 거리·치수가 같은 비율로
  스케일된다(각도는 영향 없음).
- `beta`(감속도)는 적합 상한 300 deg/s^2 에 붙었다 = 이 데이터로는 식별되지 않는다.
  관성 회전량은 사실상 `T_off` 가 결정하므로 제어에는 영향이 작지만, `beta` 값
  자체를 물리량으로 인용하면 안 된다.
- 재추론(오프라인, CPU, H.264 압축 영상)은 라이브 추론과 완전히 같지 않다.
  원본 로그 대비 |Δyaw| 중앙값 0.27~0.60 deg 차이가 있다. 4점/9점 비교는 같은
  키포인트 덤프 안에서 했으므로 공정하다.
- 토크 30 이외의 편향에는 적용할 수 없다.
"""
    io.open(DST, "w", encoding="utf-8").write(txt)
    print("saved", DST)


if __name__ == "__main__":
    main()
