"""Number all 48 actual endpoint observations for the selected FSM response."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / 'rotation_fit/out/cleanlabel_command_clips_20260906'
    with (run / 'linear_under5s/predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    model = json.loads((run / 'linear_under5s/model.json').read_text(encoding='utf-8'))
    active = json.loads((root / 'extracted/depth_cam/calib/fsm_v4/rotation_endpoint.selected.json').read_text(encoding='utf-8'))
    if len(rows) != 48 or active['parameters'] != {k: model['parameters'][k] for k in active['parameters']}:
        raise ValueError('Selected FSM model must match exactly the 48-point fit')
    output = run / 'final_selected'
    output.mkdir(exist_ok=True)
    t = np.array([float(r['command_duration_s']) for r in rows])
    y = np.array([float(r['rotation_magnitude_deg']) for r in rows])
    for i, row in enumerate(rows, 1):
        row['reference_point'] = i
    with (output / 'reference_points_48.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['reference_point'] + [k for k in rows[0] if k != 'reference_point'])
        writer.writeheader(); writer.writerows(rows)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 8))
    for direction, color, marker, label in [('LEFT', '#227ab0', 'o', '좌회전'), ('RIGHT', '#d8752d', '^', '우회전')]:
        mask = np.array([r['direction'] == direction for r in rows])
        ax.scatter(t[mask], y[mask], color=color, marker=marker, s=52, alpha=.82, label=f'{label} {mask.sum()}점')
    k, d = model['parameters']['slope_deg_s'], model['parameters']['effective_delay_s']
    grid = np.linspace(0, t.max(), 400)
    ax.plot(grid, k*np.maximum(grid-d, 0), color='#242424', lw=2.3,
            label=f'최종 함수: θ(T) = {k:.4f} × max(T - {d:.4f}, 0)')
    ax.axvline(d, color='gray', ls=':', label=f'회귀상 유효 지연 {d:.4f}초')
    ax.set(xlabel='실제 CAN 명령 유지시간 T (초)', ylabel='정지 후 총 회전량 |Δyaw| (도, Cleanlabel 추정)',
           title='최종 회전보정 함수와 실제 기준점 48개\n각 점은 실제 CAN 명령시간과 명령 전후 안정 구간의 회전량', xlim=(-.06, 2.72), ylim=(-.6, 21.8))
    ax.grid(alpha=.2); ax.legend(loc='upper left')
    fig.text(.5, .02, '토크 30 · 좌우 통합 · 5초 이상 제외 · 적합 RMSE 1.69° / R² 0.889 · 녹화별 교차검증 RMSE 1.98°', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(output / 'reference_points_48.png', dpi=160)
    fig.savefig(output / 'reference_points_48.svg')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
