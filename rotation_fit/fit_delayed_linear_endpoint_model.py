"""Offline CAN-duration -> settled C4 rotation fit; never deploys to hardware."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def fit_delayed_linear(times, angles, weights=None):
    """Global least squares for k*max(T-d,0), k,d>=0.

    Between successive observed times the active set is fixed. The optimum
    is either its unconstrained straight-line regression or an interval edge.
    """
    t, y = np.asarray(times, float), np.asarray(angles, float)
    w = np.ones_like(t) if weights is None else np.asarray(weights, float)
    if (t.ndim != 1 or not len(t) or t.shape != y.shape or t.shape != w.shape
            or not np.all(np.isfinite([t, y, w])) or np.any(t < 0)
            or np.any(y < 0) or np.any(w <= 0) or t.max() <= 0):
        raise ValueError('Expected nonnegative finite observations and positive weights/times')
    edges = np.unique(np.r_[0., t])
    candidates = list(edges)
    for lo, hi in zip(edges[:-1], edges[1:]):
        active = t > (lo + hi) / 2
        if len(np.unique(t[active])) < 2:
            continue
        a = np.column_stack([t[active], np.ones(active.sum())])
        k, b = np.linalg.lstsq(a * np.sqrt(w[active, None]),
                               y[active] * np.sqrt(w[active]), rcond=None)[0]
        if k > 0 and lo <= -b / k <= hi:
            candidates.append(float(-b / k))
    best = None
    for d in candidates:
        q = np.maximum(t - d, 0)
        denom = np.sum(w * q * q)
        k = max(0., float(np.sum(w * q * y) / denom)) if denom else 0.
        sse = float(np.sum(w * (k * q - y)**2))
        candidate = (sse, float(d), k)
        if best is None or candidate < best:
            best = candidate
    return dict(slope_deg_s=best[2], effective_delay_s=best[1], weighted_sse=best[0])


def predict(model, times):
    return model['slope_deg_s'] * np.maximum(np.asarray(times) - model['effective_delay_s'], 0.)


def inverse(model, angle, max_duration):
    if not np.isfinite(angle) or angle < 0:
        raise ValueError('Expected a finite nonnegative angle')
    if angle == 0:
        return 0.
    if model['slope_deg_s'] <= 0:
        raise ValueError('Zero slope: positive rotation is not identifiable')
    duration = model['effective_delay_s'] + angle / model['slope_deg_s']
    if duration > max_duration:
        raise ValueError('Target outside fitted duration support')
    return duration


def metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean())**2)
    return dict(n=len(y), rmse_deg=float(np.sqrt(np.mean(e**2))),
                mae_deg=float(np.mean(abs(e))), bias_deg=float(e.mean()),
                r2=float(1 - np.sum(e**2) / ss) if ss > 0 else None,
                p90_absolute_error_deg=float(np.percentile(abs(e), 90)))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(source, output, exclude_at_or_above=5.):
    with source.open(encoding='utf-8-sig', newline='') as f:
        original = list(csv.DictReader(f))
    if not original or any(r['accepted'] != 'True' for r in original):
        raise ValueError('Expected accepted settled endpoint pairs')
    if len({r['model_hash'] for r in original}) != 1:
        raise ValueError('Mixed perception models')
    rows = [dict(r) for r in original if float(r['command_duration_s']) < exclude_at_or_above]
    excluded = [dict(r, exclusion='command_duration_s >= user threshold') for r in original
                if float(r['command_duration_s']) >= exclude_at_or_above]
    t = np.array([float(r['command_duration_s']) for r in rows])
    y = np.array([float(r['rotation_magnitude_deg']) for r in rows])
    g = np.array([r['recording'] for r in rows])
    names = np.unique(g)
    if len(names) < 4:
        raise ValueError('Need at least four recording groups')
    model = fit_delayed_linear(t, y)
    fitted = predict(model, t)
    held = np.full(len(t), np.nan)
    outside = np.zeros(len(t), bool)
    folds = []
    for name in names:
        train, test = g != name, g == name
        local = fit_delayed_linear(t[train], y[train])
        held[test] = predict(local, t[test])
        outside[test] = (t[test] < t[train].min()) | (t[test] > t[train].max())
        folds.append(dict(recording=name, **local, **metrics(y[test], held[test]),
                          outside_training_duration_range=int(outside[test].sum())))
    group_rmse = float(np.sqrt(np.mean([f['rmse_deg']**2 for f in folds])))
    weights = np.array([1 / np.sum(g == name) for name in g])
    balanced = fit_delayed_linear(t, y, weights)
    rng = np.random.default_rng(20260906)
    boot = []
    for _ in range(500):
        indices = np.concatenate([np.flatnonzero(g == name) for name in rng.choice(names, len(names))])
        bm = fit_delayed_linear(t[indices], y[indices])
        boot.append([bm['slope_deg_s'], bm['effective_delay_s']])
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    report = dict(model_type='delayed_linear_settled_endpoint', parameters=model,
                  function='theta_deg(T_s) = slope_deg_s * max(T_s - effective_delay_s, 0)',
                  inverse='T_s = effective_delay_s + theta_deg/slope_deg_s for positive theta; zero -> zero',
                  source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  perception_model_hash=original[0]['model_hash'], original_count=len(original),
                  exclusion_threshold_s=exclude_at_or_above, excluded_count=len(excluded),
                  included_count=len(rows), recording_count=len(names),
                  weighting='Ordinary least squares: each command pair has equal weight; LEFT/RIGHT pooled',
                  measured_duration_range_s=[float(t.min()), float(t.max())],
                  in_sample=metrics(y, fitted), leave_one_recording_out=metrics(y, held),
                  equal_recording_loo_rmse_deg=group_rmse, reference_rmse_gate_deg=3.,
                  reference_rmse_gate_passed=group_rmse <= 3.,
                  loo_outside_training_range_count=int(outside.sum()),
                  equal_recording_weight_fit_sensitivity=balanced,
                  recording_bootstrap_runs=500,
                  recording_bootstrap_95pct_parameter_interval=dict(slope_deg_s=ci[:, 0].tolist(),
                                                                 effective_delay_s=ci[:, 1].tolist()),
                  deployment_performed=False, safe_for_control=False,
                  limitations=[
                      'Angles are settled C4/PnP estimates, not independently measured physical ground truth.',
                      'Effective delay is a fitted endpoint intercept, not measured physical motion onset.',
                      'Endpoint rotation includes inertia after STOP; slope is not independently measured peak speed.',
                      'Cross-validation holds out a recording and refits BOTH parameters; it does not retrain C4.',
                      'Predictions outside each training duration range are flagged, not clipped, and included in errors.',
                      'Validation is conditional on endpoint quality selection and the user-selected duration cutoff.',
                      'Do not extrapolate the fitted function beyond the measured time range or auto-deploy.'])
    output.mkdir(parents=True, exist_ok=False)
    for i, r in enumerate(rows):
        r.update(fitted_deg=float(fitted[i]), residual_deg=float(fitted[i] - y[i]),
                 loo_predicted_deg=float(held[i]), loo_residual_deg=float(held[i] - y[i]),
                 loo_outside_training_duration_range=bool(outside[i]))
    write_csv(output / 'predictions.csv', rows)
    write_csv(output / 'excluded_points.csv', excluded)
    write_csv(output / 'recording_cross_validation.csv', folds)
    (output / 'model.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    targets = [a for a in (1, 3, 5, 8, 10, 12, 15) if a <= predict(model, t.max())]
    write_csv(output / 'diagnostic_inverse.csv', [dict(angle_deg=a, command_duration_s=inverse(model, a, t.max())) for a in targets])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))
    colors = plt.get_cmap('tab20')
    for i, name in enumerate(names):
        mask = g == name
        label = name.replace('forklift_v4_recording_', '')
        axes[0].scatter(t[mask], y[mask], s=32, color=colors(i), label=label)
    grid = np.linspace(0, t.max() + .12, 300)
    axes[0].plot(grid, predict(model, grid), 'k-', lw=2,
                 label=f"{model['slope_deg_s']:.3f} * max(T - {model['effective_delay_s']:.3f}, 0)")
    axes[0].axvline(model['effective_delay_s'], color='gray', ls=':')
    axes[0].set(xlabel='Actual CAN command duration (s)', ylabel='Settled C4 rotation magnitude (deg)',
                title=f'{len(rows)} pairs; commands >= {exclude_at_or_above:g}s excluded')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[1].scatter(y, fitted, label='In-sample', s=32, alpha=.7)
    axes[1].scatter(y, held, label='Leave-one-recording-out', marker='x', s=32)
    limit = max(y.max(), held.max()) + 1
    axes[1].plot([0, limit], [0, limit], 'k--')
    axes[1].set(xlabel='Observed rotation (deg)', ylabel='Predicted rotation (deg)',
                title=f"LOO RMSE {report['leave_one_recording_out']['rmse_deg']:.2f} deg; R2 {report['leave_one_recording_out']['r2']:.3f}")
    axes[1].legend(fontsize=8)
    axes[2].scatter(t, held - y, s=35)
    axes[2].axhline(0, color='black')
    axes[2].axhline(3, color='gray', ls=':')
    axes[2].axhline(-3, color='gray', ls=':')
    axes[2].set(xlabel='Actual CAN command duration (s)', ylabel='LOO prediction minus observation (deg)',
                title=f'Recording-balanced LOO RMSE {group_rmse:.2f} deg')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'fit_validation.png', dpi=160)
    plt.close(fig)
    k, d = model['slope_deg_s'], model['effective_delay_s']
    m, cv = report['in_sample'], report['leave_one_recording_out']
    excluded_lines = '\n'.join(f"- {r['recording']}, step {r['step']}: {float(r['command_duration_s']):.3f}s / {float(r['rotation_magnitude_deg']):.3f}°" for r in excluded)
    review = f'''# {exclude_at_or_above:g}초 이상 제외: 정지시간 + 선형 회전량 적합

기존 {len(original)}개 endpoint 측정점에서 사용자 지정 조건 T >= {exclude_at_or_above:g}s만 추가 제외했다. 원본 CSV는 변경하지 않았다.
입력 {len(original)}점 → 제외 {len(excluded)}점 → 적합 {len(rows)}점 / {len(names)}개 녹화. 좌우 통합, 토크 30 로그.

## 함수

θ(T) = {k:.6f} × max(T − {d:.6f}, 0) [deg]

양의 목표각도에 대한 역함수: T(θ) = {d:.6f} + θ / {k:.6f} [s]. θ=0이면 명령시간 0.
측정 명령시간 범위: {t.min():.3f}–{t.max():.3f}s. 5초 미만 전체가 검증된 것은 아니다.
기울기·절편을 동시에 추정한 최소제곱 회귀이며 각 명령 측정점에 동일 가중치를 적용했다.
정지시간 전 구간의 작은 관측값도 제외하지 않고 오차에 포함했다.

## 검증

| 방식 | RMSE (°) | MAE (°) | R² |
|---|---:|---:|---:|
| 동일 로그 적합 | {m['rmse_deg']:.4f} | {m['mae_deg']:.4f} | {m['r2']:.4f} |
| 녹화 전체를 하나씩 제외한 교차검증 | {cv['rmse_deg']:.4f} | {cv['mae_deg']:.4f} | {cv['r2']:.4f} |

녹화별 동일 가중 교차검증 RMSE: {group_rmse:.4f}°. 기존 참고 기준 3°: {'통과' if group_rmse <= 3 else '미통과'}.
각 fold에서 정지시간과 기울기를 모두 다시 추정했다. 학습 시간범위 밖 {outside.sum()}점도 제외하거나 클리핑하지 않고 검증에 포함했다.
이 수치는 현재 선택된 로그에 대한 응답함수 검증이며 독립 실측 및 새로운 환경 검증이 아니다.

## 제외점

{excluded_lines}

## 해석과 제한

- 정지시간 {d:.4f}s는 최종 회전량의 절편에서 얻은 **유효 지연시간**이다. 실제 움직임이 시작된 프레임을 직접 측정한 값과 같다고 할 수 없다.
- θ는 정지 후 안정 구간과 명령 전 안정 구간의 C4 각도 차이다. 관성이 포함되므로 이 기울기를 실제 최대 회전속도라고 단정하지 않는다.
- 기존 물리 구간 모델과는 예측 대상·포함 로그가 달라 과거 RMSE와 직접 비교하여 개선율을 주장하지 않는다.
- 원시 프레임 재추론/각도 재계산 없이 기존 endpoint를 사용했다. 이전 인식 또는 면 전환 오차가 남아 있을 수 있다.
- model.json은 오프라인 진단 결과이며 제어 설정과 실제 장비에는 적용하지 않았다.

![적합 및 검증](fit_validation.png)
'''
    (output / 'REVIEW.md').write_text(review, encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-at-or-above', type=float, default=5.)
    args = parser.parse_args()
    main(args.source, args.output, args.exclude_at_or_above)
