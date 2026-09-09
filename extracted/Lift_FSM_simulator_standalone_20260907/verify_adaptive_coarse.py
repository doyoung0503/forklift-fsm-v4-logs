"""Exercise the applied threshold through the real FSM and virtual CAN."""
import json
from pathlib import Path

from calibrate_lateral import placement
from process_runtime import Session
from v4_runtime import cfg
from calib.fsm_v4.coarse import coarse_lateral_limit_m


def main():
    output = Path('verification/adaptive_coarse_applied_20260909')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for distance in (3., 3.5, 4., 4.5, 5.):
        limit = coarse_lateral_limit_m(distance)
        for side in (1., -1.):
            for fraction in (.98, 1.05):
                lateral = side * fraction * limit
                report = Session(placement(distance, lateral), {},
                                 capture=True, log_root=None).run()
                if report['controller_error']:
                    raise RuntimeError(report['controller_error'])
                coarse = any(f['state'].startswith('COARSE_') for f in report['frames'])
                changes = []
                for frame in report['frames']:
                    if not changes or frame['state'] != changes[-1]['state']:
                        changes.append({k: frame[k] for k in ('t', 'state', 'truth', 'lines')})
                row = dict(distance_m=distance, lateral_m=lateral, limit_m=limit,
                           expected_coarse=fraction > 1., observed_coarse=coarse,
                           routing_passed=coarse == (fraction > 1.), changes=changes,
                           **{k: report[k] for k in ('success', 'state', 'reasons', 'collision',
                              'elapsed', 'final', 'options', 'overrides', 'source_id',
                              'simulator_id', 'trace_hash', 'provenance')})
                rows.append(row)
                (output/f'case_{len(rows):02d}.json').write_text(
                    json.dumps(row, indent=2), encoding='utf8')
                (output/'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
                print(json.dumps({k: row[k] for k in ('distance_m', 'lateral_m',
                    'expected_coarse', 'observed_coarse', 'success', 'reasons')}), flush=True)
    if not all(r['routing_passed'] and r['success'] for r in rows):
        raise RuntimeError('Applied threshold regression failed; inspect saved evidence')


if __name__ == '__main__':
    main()
