"""Fit a conservative proposal, validate it with the same FSM, export evidence."""
import csv
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from calibrate_lateral import Experiment

OUT = Path('verification/lateral_no_predictive_20260909')


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(OUT))
    parser.add_argument('--policy-output', default='verification/lateral_current_policy_validation_20260909')
    parser.add_argument('--validation-output', default=None)
    args = parser.parse_args()
    OUT = Path(args.output)
    bounds = json.loads((OUT/'boundaries.json').read_text(encoding='utf8'))
    if len(bounds) != 10 or any('conservative_success_m' not in r for r in bounds):
        raise RuntimeError('Resolve incomplete or unguarded boundaries before fitting')
    provenance = json.loads((OUT/'provenance.json').read_text(encoding='utf8'))
    stage = provenance['config']['STAGING_DISTANCE_M']
    tolerance = provenance['config']['FINAL_LATERAL_TOL_M']
    distances = sorted({r['distance_m'] for r in bounds})
    limits = [min(r['conservative_success_m'] for r in bounds if r['distance_m']==d) for d in distances]
    approach = np.array(distances)-stage
    excess = np.array(limits)-tolerance
    k_ls = float(approach@excess/(approach@approach))
    k_envelope = float(min(excess/approach))
    eta = .8  # Engineering choice, NOT inferred real-world reliability.
    k_safe = math.floor(eta*k_envelope*1000)/1000
    search = Experiment(OUT, disable_predictive=True)
    experiment = Experiment(args.validation_output or OUT, disable_predictive=True)
    validation = []
    for distance in np.arange(3., 5.001, .25):
        limit = tolerance+k_safe*(distance-stage)
        for side in (1, -1):
            row = experiment.trial(float(distance), side*limit)
            validation.append(dict(distance_m=float(distance), lateral_m=side*limit,
                                   facing=True, success=row['success'], reasons=row['reasons']))
    # Test interior values at measured distances and a second initial heading
    # at three distances, without confusing those with the calibrated facing case.
    for distance in distances:
        limit = tolerance+k_safe*(distance-stage)
        for side in (1, -1):
            row = experiment.trial(distance, side*.5*limit)
            validation.append(dict(distance_m=distance, lateral_m=side*.5*limit,
                                   facing=True, success=row['success'], reasons=row['reasons']))
    for distance in (3., 4., 5.):
        limit = tolerance+k_safe*(distance-stage)
        for side in (1, -1):
            row = experiment.trial(distance, side*limit, facing=False)
            validation.append(dict(distance_m=distance, lateral_m=side*limit,
                                   facing=False, success=row['success'], reasons=row['reasons']))
    # A proposed trigger must also work with the unchanged production policy.
    # These remain separate from the no-predictive ideal-reference experiment.
    baseline = Experiment(args.policy_output)
    current_validation = []
    for distance in distances:
        limit = tolerance+k_safe*(distance-stage)
        for side in (1, -1):
            row = baseline.trial(distance, side*limit)
            current_validation.append(dict(distance_m=distance, lateral_m=side*limit,
                                           success=row['success'], reasons=row['reasons']))
    if {r['source_id'] for r in search.rows + experiment.rows + baseline.rows} != {provenance['source_id']}:
        raise RuntimeError('Calibration and validation FSM versions differ')
    fit = dict(distance_definition='pallet-normal rotation-centre distance',
               domain_m=[3.,5.], lateral_tolerance_m=tolerance, staging_distance_m=stage,
               reserve_m=0., least_squares_k=k_ls, lower_envelope_k=k_envelope,
               engineering_retention=eta, proposed_k=k_safe,
               effective_angle_deg=math.degrees(math.atan(k_safe)),
               formula='L_limit = lateral_tolerance_m + proposed_k * (D - staging_distance_m)',
               production_controller_changed=False, validation=validation,
               calibration_source_id=provenance['source_id'],
               search_trials=len(search.rows),
               validation_output=str(experiment.output),
               policy_output=str(baseline.output),
               current_policy_validation=current_validation,
               current_policy_validation_passed=all(r['success'] for r in current_validation),
               facing_validation_passed=all(r['success'] for r in validation if r['facing']),
               all_validation_passed=all(r['success'] for r in validation))
    (OUT/'fit.json').write_text(json.dumps(fit, indent=2), encoding='utf8')
    all_rows = list({(r['distance_m'], r['lateral_m'], r['facing']): r
                     for r in search.rows + experiment.rows}.values())
    with (OUT/'trials.csv').open('w', newline='', encoding='utf-8-sig') as file:
        fields=['distance_m','lateral_m','facing','success','state','elapsed','reasons','trace_hash']
        writer=csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fields})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    xx=np.linspace(3,5,200)
    for side, color, label in ((1,'#2563eb','Right placement'),(-1,'#a855f7','Left placement')):
        rows=[r for r in bounds if r['side']==side]
        ax.errorbar([r['distance_m'] for r in rows], [r['conservative_success_m'] for r in rows],
                    yerr=[r['conservative_failure_m']-r['conservative_success_m'] for r in rows],
                    fmt='o',color=color,capsize=5,label=label)
    ax.plot(xx,tolerance+k_ls*(xx-stage),'--',color='#9ca3af',label='Least-squares approximation')
    ax.plot(xx,tolerance+k_envelope*(xx-stage),':',color='#d97706',label='Sampled lower envelope')
    ax.plot(xx,tolerance+k_safe*(xx-stage),color='#15803d',label='Conservative proposal')
    ax.set(xlabel='Pallet-normal pivot distance D (m)',ylabel='Absolute lateral (m)',
           title='No predictive stop: audited inner boundaries')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    facing_rows=[r for r in all_rows if r['facing']]
    for ok, marker, color in ((True,'o','#15803d'),(False,'x','#dc2626')):
        rows=[r for r in facing_rows if r['success']==ok]
        bx.scatter([r['distance_m'] for r in rows],[abs(r['lateral_m']) for r in rows],
                   marker=marker,color=color,s=22,alpha=.65,label='Success' if ok else 'Failure')
    bx.plot(xx,tolerance+k_safe*(xx-stage),color='#15803d')
    bx.set(xlabel='Pallet-normal pivot distance D (m)',ylabel='Absolute lateral (m)',
           title='All facing-start samples (both sides)')
    bx.legend();bx.grid(alpha=.2)
    fig.savefig(OUT/'lateral_threshold.png',dpi=180)
    fig.savefig(OUT/'lateral_threshold.svg')
    print(json.dumps(fit,indent=2),flush=True)


if __name__=='__main__':
    main()
