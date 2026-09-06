"""Reconstruct capture-time response from recording metadata AND logged plans."""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'extracted/depth_cam'))
    from calib.fsm_v4.rotation_model import RotationResponse
    from rotation_fit.fit_delayed_linear_endpoint_model import predict, metrics
    run = root / 'rotation_fit/out/cleanlabel_command_clips_20260906'
    recordings = root / 'rotation_fit/out/command_clips'
    final = read_json(run / 'linear_under5s/model.json')
    with (run / 'linear_under5s/predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        endpoints = list(csv.DictReader(f))
    audit, models, events = [], {}, {}
    for path in sorted(recordings.glob('*_meta.json')):
        name = path.name.removesuffix('_meta.json')
        meta = read_json(path)
        with (recordings / f'{name}_control_seq.jsonl').open(encoding='utf-8-sig') as f:
            begins = [json.loads(line) for line in f if line.strip()]
        begins = [r for r in begins if r.get('phase') == 'begin' and 'fitted_hold_sec' in r.get('params', {})]
        events[name] = {str(r['step']): r for r in begins}
        row = dict(recording=name, fitted_enabled=meta.get('v4_rotation_use_fitted_response'),
                   fitted_plan_count=len(begins), capture_model_known=False,
                   coast_reduction_deg=meta.get('v4_rotation_coast_heuristic_reduction_deg'),
                   correction_evidence='not recorded', max_logged_plan_residual_deg=None,
                   adaptive_enabled=meta.get('v4_rotation_adaptive_overshoot_enabled'),
                   bearing_centre_offset_m=meta.get('v4_rotation_bearing_centre_offset_m'))
        if meta.get('v4_rotation_use_fitted_response') is True:
            response = RotationResponse(*[meta[f'v4_rotation_response_{key}'] for key in
                        ['startup_delay_sec', 'stop_delay_sec', 'accel_deg_s2', 'decel_deg_s2', 'max_rate_deg_s']])
            errors = {}
            for correction in [0., 1.07]:
                residuals = []
                for event in begins:
                    p = event['params']
                    gain = float(p['fitted_max_rate_deg_s']) / response.max_rate_deg_s
                    hold = float(p['fitted_hold_sec'])
                    # Gain is captured in each plan, including actual bearing-domain scaling.
                    predicted = response.angle_at(hold)*gain + max(0., response.coast_after_hold(hold)*gain-correction)
                    residuals.append(abs(predicted-float(p['fitted_total_deg'])))
                if residuals:
                    errors[correction] = max(residuals)
            correction = row['coast_reduction_deg']
            if correction is None and errors:
                candidate = min(errors, key=errors.get)
                if errors[candidate] < 1e-5:
                    correction = candidate
                    row['correction_evidence'] = 'Reproduces every recorded plan total to <1e-5deg; metadata field absent'
            elif correction is not None:
                row['correction_evidence'] = 'Recorded metadata, checked against recorded plan totals'
            if correction is not None and errors.get(correction, float('inf')) < 1e-5:
                row.update(capture_model_known=True, coast_reduction_deg=correction,
                           max_logged_plan_residual_deg=errors[correction])
                models[name] = (response, correction)
        audit.append(row)
    paired = []
    for r in endpoints:
        name, step = r['recording'], r['step']
        event = events[name].get(step)
        if name not in models or event is None:
            continue
        old, correction = models[name]
        gain = float(event['params']['fitted_max_rate_deg_s']) / old.max_rate_deg_s
        t = float(r['command_duration_s'])
        before = old.angle_at(t) + max(0., old.coast_after_hold(t)-correction/gain)
        paired.append(dict(recording=name, step=step, actual_can_duration_s=t,
                           observed_cleanlabel_rotation_deg=float(r['rotation_magnitude_deg']),
                           capture_response_predicted_heading_deg=before,
                           selected_linear_predicted_heading_deg=float(predict(final['parameters'], t)),
                           logged_plan_hold_sec=event['params']['fitted_hold_sec'],
                           logged_plan_total_deg=event['params']['fitted_total_deg'],
                           logged_domain=event['params']['fitted_domain'], bearing_gain=gain,
                           capture_coast_reduction_in_logged_domain_deg=correction))
    sample = next(iter(models.values()))[0]
    grid = np.linspace(0, 2.531, 400)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter([float(r['command_duration_s']) for r in endpoints],
               [float(r['rotation_magnitude_deg']) for r in endpoints], s=32, color='#a3acb5', alpha=.6, label='현재 Cleanlabel 측정점 48개')
    ax.plot(grid, predict(final['parameters'], grid), color='#057a9d', lw=2.5, label='현재 선택한 선형회귀 (48점)')
    ax.plot(grid, [sample.adjusted_total_angle(float(t), 0.) for t in grid], color='#9b6baa', lw=2,
            label='수집 당시: 추가 관성 차감 없음 (9/3 ~ 9/4 오전)')
    ax.plot(grid, [sample.adjusted_total_angle(float(t), 1.07) for t in grid], color='#dd8243', lw=2, ls='--',
            label='수집 당시: 관성 1.07° 차감 (9/4 오후, heading 기준)')
    ax.set(xlabel='실제 CAN 명령 유지시간 (초)', ylabel='최종 회전량 (도, heading)',
           title='녹화 당시 실제 함수 설정과 현재 선형회귀 비교\n9/1 영상 4개는 설정 기록 부족으로 과거 함수 확정 제외')
    ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.tight_layout()
    output = run / 'capture_time_comparison'
    output.mkdir(exist_ok=False)
    fig.savefig(output / 'capture_vs_current.png', dpi=160)
    plt.close(fig)
    for filename, rows in [('recording_function_audit.csv', audit), ('same_command_comparison.csv', paired)]:
        with (output / filename).open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    y = np.array([r['observed_cleanlabel_rotation_deg'] for r in paired])
    scores = {method: metrics(y, np.array([r[column] for r in paired])) for method, column in
              [('capture_time_function', 'capture_response_predicted_heading_deg'), ('selected_linear', 'selected_linear_predicted_heading_deg')]}
    result = dict(recording_count=len(audit), verified_capture_functions=len(models),
                  recording_audit=audit, paired_endpoint_count=len(paired), descriptive_scores=scores,
                  heading_examples=[dict(hold_s=t, capture_no_reduction_deg=sample.adjusted_total_angle(t, 0),
                                         capture_reduced_deg=sample.adjusted_total_angle(t, 1.07),
                                         selected_linear_deg=float(predict(final['parameters'], t))) for t in [1.5, 2., 2.5]],
                  target_time_examples=[dict(angle_deg=a,
                                            capture_no_reduction_s=sample.command_seconds(a, max_hold_sec=2.5, coast_reduction_deg=0),
                                            capture_reduced_s=sample.command_seconds(a, max_hold_sec=2.5, coast_reduction_deg=1.07),
                                            selected_linear_s=final['parameters']['effective_delay_s']+a/final['parameters']['slope_deg_s']) for a in [3., 5., 10.]],
                  limitations=['Missing fitted metadata is unknown, not proof the fit was disabled.',
                               'Plan durations and actual CAN durations differ due to live STOP and loop timing.',
                               'Descriptive errors are against new Cleanlabel estimates on its fitting cohort, not unbiased generalization or physical truth.',
                               '3 original videos enabled adaptive timing; each logged plan was used to verify the base response.',
                               'Graph shows heading functions; bearing actions require each logged gain and correction-domain conversion.'])
    (output / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'recording_audit'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
