"""Compare three response models using only the user's 15 selected recordings.

Physical fitting uses motion traces; endpoint models use stable endpoints.
All models are evaluated on the same endpoint pairs, holding out entire videos.
No deployment and no new image inference.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from rotation_fit.rotation_log_fit import (
    FitSettings, load_frame_series, load_command_windows, analyse_segment,
    _apply_direction_sign_audit, _fit_once, _driven_angle, _json_safe, write_segments_csv)
from rotation_fit.fit_delayed_linear_endpoint_model import fit_delayed_linear, predict, metrics
from rotation_fit.fit_settled_endpoint_model import fit_knots


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    import csv
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    import csv
    if rows:
        with path.open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(_json_safe(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def fit_physical(raw_segments, settings):
    # Sign screening must be learned from training videos only, not held-out data.
    segments = copy.deepcopy(raw_segments)
    signs = _apply_direction_sign_audit(segments)
    delay, phase, inertia, model = _fit_once(segments, settings)
    usable = [s for s in segments if s.drive_usable]
    maximum = max((float(s.drive_time_from_onset_s.max()) for s in usable), default=0.)
    lower_tau = max(.05, min(.15, .25 * maximum))
    upper_tau = max(lower_tau + .05, 1.25 * maximum)
    boundary = bool(np.isfinite(phase.accel_duration_sec) and phase.accel_duration_sec <= lower_tau + .001)
    report = dict(dead_time_sec=delay, phase_fit=asdict(phase), inertia_fit=asdict(inertia),
                  segment_count=len(segments), drive_usable=sum(s.drive_usable for s in segments),
                  inertia_usable=sum(s.inertia_usable for s in segments), direction_sign_audit=signs,
                  acceleration_duration_bounds_s=[lower_tau, upper_tau],
                  acceleration_at_lower_bound=boundary,
                  warning=('Acceleration duration is at optimizer lower bound; acceleration and its duration are not independently resolved.' if boundary else ''),
                  parameters=model.to_dict() if model else None,
                  deployment_performed=False, safe_for_control=False)
    return model, report, segments


def main(run, recordings, output):
    endpoint_report = read_json(run / 'selected_endpoints/endpoint_extraction_report.json')
    names = endpoint_report['selected_recordings']
    if len(names) != 15:
        raise ValueError('Expected exactly 15 selected recordings')
    rows = [r for r in read_csv(run / 'selected_endpoints/accepted_command_pairs.csv')
            if float(r['command_duration_s']) < 5]
    if any(r['recording'] not in names or r['model_hash'] != endpoint_report['model_hash'] for r in rows):
        raise ValueError('Endpoint selection or model mismatch')
    settings = FitSettings()
    raw, excluded_long, source_hashes = [], [], []
    for name in names:
        path = run / 'inference' / f'{name}_new_pose.csv'
        frames = load_frame_series(recordings / f'{name}_inference_timing.csv', path,
                                   angle_column='yaw_deg', valid_columns=['pose_ok'],
                                   confidence_column='confidence', min_confidence=.4,
                                   model_id_column='model_hash')
        if set(frames.model_ids) != {endpoint_report['model_hash']}:
            raise ValueError('Trace model mismatch')
        source_hashes.append(dict(recording=name, pose_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for window in load_command_windows(recordings / f'{name}_control_seq.jsonl', settings):
            segment = analyse_segment(window, frames, settings)
            if segment.command_duration_s >= 5:
                excluded_long.append(dict(recording=name, step=segment.step, hold_sec=segment.command_duration_s))
            else:
                raw.append(segment)
    output.mkdir(parents=True, exist_ok=False)
    physical, physical_report, fitted_segments = fit_physical(raw, settings)
    save_json(output / 'physical_model.json', physical_report)
    write_segments_csv(output / 'physical_segments.csv', fitted_segments)
    write_csv(output / 'excluded_long_commands.csv', excluded_long)
    t = np.array([float(r['command_duration_s']) for r in rows])
    y = np.array([float(r['rotation_magnitude_deg']) for r in rows])
    g = np.array([r['recording'] for r in rows])
    linear = fit_delayed_linear(t, y)
    knots = fit_knots(t, y, g)
    save_json(output / 'selected_linear_model.json', dict(parameters=linear, selected_recordings=names,
              function='theta(T)=k*max(T-d,0)', deployment_performed=False))
    save_json(output / 'selected_isotonic_model.json', dict(knots=knots, selected_recordings=names,
              function='theta(T)=linear interpolation of knots; zero anchor; no out-of-range use', deployment_performed=False))
    train_predictions = dict(linear=predict(linear, t), isotonic=np.interp(t, *knots),
                             physical=np.array([physical.total_angle(v) for v in t]) if physical else np.full(len(t), np.nan))
    held = {method: np.full(len(t), np.nan) for method in train_predictions}
    folds = []
    for name in np.unique(g):
        train, test = g != name, g == name
        pmodel, preport, _ = fit_physical([s for s in raw if s.recording != name], settings)
        lmodel = fit_delayed_linear(t[train], y[train])
        local_knots = fit_knots(t[train], y[train], g[train])
        held['linear'][test] = predict(lmodel, t[test])
        # Same endpoint method as previous selected-fit validation (boundary clamping).
        held['isotonic'][test] = np.interp(t[test], *local_knots)
        if pmodel:
            held['physical'][test] = [pmodel.total_angle(v) for v in t[test]]
        folds.append(dict(recording=name, test_count=int(test.sum()), physical=preport,
                          endpoint_outside_training_time_range=int(np.sum((t[test] < t[train].min()) | (t[test] > t[train].max()))),
                          linear_parameters=lmodel))
        print(f'validated {name}', flush=True)
    scores = {}
    for method in held:
        valid = np.isfinite(held[method])
        in_valid = np.isfinite(train_predictions[method])
        by_recording = {name: metrics(y[(g == name) & valid], held[method][(g == name) & valid])
                        for name in np.unique(g) if np.any((g == name) & valid)}
        scores[method] = dict(in_sample=metrics(y[in_valid], train_predictions[method][in_valid]) if in_valid.any() else None,
                             loo=metrics(y[valid], held[method][valid]) if valid.any() else None,
                             evaluated_points=int(valid.sum()), total_test_points=len(t),
                             equal_recording_loo_rmse_deg=float(np.sqrt(np.mean([m['rmse_deg']**2 for m in by_recording.values()]))) if by_recording else None,
                             rmse_by_recording=by_recording, complete_coverage=bool(valid.all()))
    for i, row in enumerate(rows):
        for method in held:
            row[method + '_fitted_deg'] = float(train_predictions[method][i])
            row[method + '_loo_deg'] = float(held[method][i])
        fold_model = next(f['physical']['parameters'] for f in folds if f['recording'] == row['recording'])
        support = fold_model['fit_metadata'] if fold_model else None
        row['physical_loo_outside_trace_fitted_hold_range'] = (
            not support['fitted_min_hold_sec'] <= t[i] <= support['fitted_max_hold_sec']
            if support else None)
    write_csv(output / 'same_endpoint_predictions.csv', rows)
    report = dict(selected_recordings=names, settings=asdict(settings), model_hash=endpoint_report['model_hash'],
                  source_hashes=source_hashes, endpoint_points=len(rows), endpoint_recordings=len(np.unique(g)),
                  physical_training_command_count=len(raw), excluded_commands_at_or_above_5s=excluded_long,
                  comparison='Same held-out endpoint targets and selected 15-video pool. Physical model learns usable motion traces; endpoint models learn stable pairs. No held-out video in any training or direction-sign screening.',
                  scores=scores, folds=folds, physical_fit=physical_report,
                  limitations=['Endpoint selection is fixed from prior analysis; validation is conditional on that selection.',
                               'Physical inertia targets STOP+2s; shared evaluation uses stable tails up to STOP+2.7s.',
                               'Physical and linear extrapolations are included without clipping; isotonic boundary-clamps as in existing validation.',
                               'No independent physical yaw truth. Cannot infer an inertia time profile from one post-STOP endpoint.',
                               'Lower-bound acceleration duration is not proof of a measured short acceleration phase.'],
                  deployment_performed=False, safe_for_control=False)
    save_json(output / 'comparison.json', report)
    grid = np.linspace(0, max(t), 500)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0, 0].scatter(t, y, s=25, alpha=.6, label=f'Same endpoints (n={len(t)})')
    axes[0, 0].plot(grid, predict(linear, grid), label='Delayed linear')
    axes[0, 0].plot(grid, np.interp(grid, *knots), label='Selected isotonic')
    if physical:
        axes[0, 0].plot(grid, [physical.total_angle(v) for v in grid], label='Physical phases', color='#d04b35')
    axes[0, 0].set(xlabel='CAN command duration (s)', ylabel='Final rotation (deg)', title='Three models: selected 15 videos only')
    axes[0, 0].legend()
    for method in held:
        axes[0, 1].scatter(y, held[method], s=26, alpha=.65, label=method)
    axes[0, 1].plot([0, y.max()], [0, y.max()], 'k--')
    axes[0, 1].set(xlabel='Observed endpoint rotation (deg)', ylabel='Held-out prediction (deg)', title='Identical leave-one-video-out test targets')
    axes[0, 1].legend()
    for seg in fitted_segments:
        if seg.drive_usable:
            axes[1, 0].scatter(seg.drive_time_from_onset_s, seg.drive_angle_from_onset_deg, s=5, alpha=.3)
        if seg.inertia_usable:
            axes[1, 1].scatter(seg.stop_rate_observed_deg_s, seg.inertia_rotation_deg, s=25, alpha=.7)
    phase, inertia = physical_report['phase_fit'], physical_report['inertia_fit']
    if physical:
        q = np.linspace(0, max(s.drive_time_from_onset_s.max() for s in fitted_segments if s.drive_usable), 300)
        axes[1, 0].plot(q, _driven_angle(q, phase['accel_deg_s2'], phase['accel_duration_sec']), 'k-', lw=2)
        axes[1, 0].axvline(phase['accel_duration_sec'], color='gray', ls='--')
        speed = np.linspace(0, inertia['speed_max_deg_s'], 200)
        axes[1, 1].plot(speed, inertia['intercept_deg'] + inertia['slope_sec']*speed, 'k-')
    axes[1, 0].set(xlabel='Elapsed from each detected onset (s)', ylabel='Command-held rotation (deg)', title='Constant acceleration -> cruise fit')
    axes[1, 1].set(xlabel='Observed pre-STOP angular speed (deg/s)', ylabel='Observed rotation to STOP+2s (deg)', title='Linear inertia fit (speed -> extra angle)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'three_methods.png', dpi=150)
    plt.close(fig)
    lines = ['# 선택 15개 영상: 물리 구간 모델과 두 endpoint 모델 비교', '',
             f'선택 영상 15개만 사용. 5초 이상 명령 {len(excluded_long)}개 제외. 물리 분석 명령 {len(raw)}개, 구동 사용 {physical_report["drive_usable"]}개, 관성 사용 {physical_report["inertia_usable"]}개.',
             f'비교는 동일한 안정 구간 측정점 {len(rows)}개 / {len(np.unique(g))}개 녹화에서 수행. 선택 영상 선형회귀도 이 데이터로 새로 적합했다(앞선 전체 48점 선형회귀와 다름).', '',
             '## 물리 함수', '',
             f'정지시간 d = {physical_report["dead_time_sec"]:.6f}s',
             f'등가속 시간 τ = {phase["accel_duration_sec"]:.6f}s, 가속도 a = {phase["accel_deg_s2"]:.6f}deg/s²',
             f'최대 속도 v = aτ = {phase["max_rate_deg_s"]:.6f}deg/s',
             f'관성 추가각도 I(ω) = {inertia["intercept_deg"]:.6f} + {inertia["slope_sec"]:.6f}ω [deg] (ω>0; 정지 시 0)', '',
             '명령시간 T에서 u=max(T-d,0), ω(T)=a·min(u,τ).',
             '명령 중 회전량 D(T)=0.5·a·u² (u≤τ), 그 이후 D(T)=aτ·(u−τ/2).',
             '최종 회전량 Θ(T)=D(T)+I(ω(T)). 역함수는 물리 모델의 command_duration()으로 계산할 수 있으나 진단용이다.', '',
             '지연은 명령 뒤 최초 움직임의 프레임 구간에서 추정하고, 구동은 움직임 시작 이후 STOP 직전까지의 프레임 로그로 적합했다. 관성은 실제 STOP 직전 속도와 STOP+2초 추가각도로 적합했다.',
             '관성 회귀에는 예측 속도가 아닌 로그에서 관측한 속도를 쓰고, 명령시간만으로 최종각도를 예측할 때는 구동 모델이 예측한 종료 속도를 쓴다.', '',
             '## 동일 관측점 교차검증', '',
             '| 방법 | 적합 RMSE | 적합 R² | 교차검증 RMSE | 교차검증 R² | 녹화 동일가중 RMSE | 예측점 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method, score in scores.items():
        if score['in_sample'] and score['loo']:
            m, cv = score['in_sample'], score['loo']
            lines.append(f"| {method} | {m['rmse_deg']:.3f}° | {m['r2']:.3f} | {cv['rmse_deg']:.3f}° | {cv['r2']:.3f} | {score['equal_recording_loo_rmse_deg']:.3f}° | {score['evaluated_points']}/{len(t)} |")
        else:
            lines.append(f'| {method} | 추정 불가 | — | — | — | — | {score["evaluated_points"]}/{len(t)} |')
    lines += ['', '각 fold에서 물리 파라미터와 좌우 부호 판별도 학습 영상만으로 다시 계산했다. 테스트 영상의 중간 속도/회전량을 입력해 예측하지 않았다.', '',
              '## 식별 가능성과 제한', '',
              physical_report['warning'] or '가속 시간의 하한 고착은 검출되지 않음. 이것만으로 물리 파라미터의 정확성이 입증되지는 않음.',
              '- 물리 모델은 가속·관성 추정을 위해 중간 프레임 로그가 필요하다. endpoint가 안정해도 중간 프레임이 안정하다는 보장은 없다.',
              '- 구동/관성의 자체 적합 오차와 최종 회전량의 통합 예측 오차를 구분해야 한다.',
              '- STOP+2초 관성과 STOP+2.7초 이내 안정 구간 사이의 차이도 최종 오차에 포함된다.',
              '- 같은 영상/평가점이지만 훈련 입력은 다름: 물리 모델은 해당 15개 영상의 5초 미만 명령 추적 로그, 나머지는 안정 endpoint만 사용.',
              '- 모든 값은 Cleanlabel 영상 추정치에 대한 일치도. 실측 정답 검증이나 장비 제어 적용은 하지 않았다.', '',
              '![세 모델 비교](three_methods.png)', '',
              '[물리 함수](physical_model.json) · [구간별 실제 시각과 제외 사유](physical_segments.csv) · [동일점 예측 CSV](same_endpoint_predictions.csv) · [전체 비교 및 fold 결과](comparison.json)', '']
    with (output / 'REVIEW.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(json.dumps(_json_safe(dict(physical=physical_report, scores=scores)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--recordings', type=Path, default=Path('rotation_fit/out/command_clips'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.recordings, args.output)
