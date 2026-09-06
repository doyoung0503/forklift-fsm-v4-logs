"""Show the selected endpoint function, previous FSM fallback and diagnostic phase fit."""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'extracted/depth_cam'))
    from calib.fsm_v4 import config as cfg
    from rotation_fit.piecewise_rotation_model import LogRotationModel
    run = root / 'rotation_fit/out/cleanlabel_command_clips_20260906'
    diagnostic = json.loads((run / 'physical_selected_comparison/physical_model.json').read_text(encoding='utf-8'))
    physical = LogRotationModel.from_dict(diagnostic['parameters'])
    final = cfg.ROTATION_RESPONSE
    old = cfg.ROTATION_RESPONSE_FALLBACK
    old_total = lambda t: old.adjusted_total_angle(t, cfg.ROT_COAST_HEURISTIC_REDUCTION_DEG)
    with (run / 'linear_under5s/predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    output = run / 'final_selected'
    grid = np.linspace(0, 2.5, 400)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter([float(r['command_duration_s']) for r in rows],
                    [float(r['rotation_magnitude_deg']) for r in rows], s=27, alpha=.55,
                    color='#9ba5ad', label='실제 Cleanlabel 기준점 48개')
    for fn, color, label in [(final.total_angle, '#087ea4', '최종 선택: 정지시간 + 선형회귀'),
                              (old_total, '#8064a2', '기존 FSM 기본값: 관성 1.07도 보정 포함'),
                              (physical.total_angle, '#d75c43', '직전 물리 구간 적합: 진단용, 미적용')]:
        axes[0].plot(grid, [fn(float(t)) for t in grid], color=color, lw=2, label=label)
    axes[0].set(xlabel='CAN 명령시간 (초)', ylabel='예측 최종 회전량 (도)', title='같은 명령시간에 대한 회전량 예측')
    angles = np.linspace(2.5, 14, 200)
    new_holds = np.array([final.command_seconds(float(a)) for a in angles])
    old_holds = np.array([old.command_seconds(float(a), max_hold_sec=2.5,
                                            coast_reduction_deg=cfg.ROT_COAST_HEURISTIC_REDUCTION_DEG) for a in angles])
    axes[1].plot(angles, 1000*(new_holds-old_holds), color='#087ea4', lw=2)
    axes[1].axhline(0, color='gray', ls='--')
    axes[1].set(xlabel='목표 회전량 (도)', ylabel='새 함수 명령시간 - 기존 FSM 명령시간 (ms)',
                title='기존 FSM 대비 명령시간 변화\n양수: 새 함수가 더 오래 명령')
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'previous_vs_final.png', dpi=160)
    plt.close(fig)
    result = dict(
        time_predictions=[dict(command_s=t, final_deg=final.total_angle(t),
                               old_fsm_deg=old_total(t), diagnostic_physical_deg=physical.total_angle(t))
                          for t in [1., 1.5, 2., 2.5]],
        target_holds=[dict(target_deg=a, final_s=final.command_seconds(a),
                           old_fsm_s=old.command_seconds(a, max_hold_sec=2.5, coast_reduction_deg=1.07))
                      for a in [3., 5., 10., 14.]],
        interpretations=['Old FSM fallback and recent diagnostic physical fit are distinct models.',
                         'Comparison is of deterministic function shapes, not a new controlled accuracy benchmark.',
                         'All curves refer to final settled angle, not instantaneous angle during movement.'])
    (output / 'previous_vs_final.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
