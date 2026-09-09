"""Reproducible zero-added-error lateral reachability experiment, virtual CAN only.

Distance is pallet-normal rotation-centre distance, NOT camera Z or range.
Binary search locates a sampled success/failure boundary; interior probes audit
its monotonicity assumption. Disconnected successes are never a safe envelope.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from process_runtime import Session, SIMULATOR_HASHES, SIMULATOR_ID, HERE


def placement(distance, lateral, facing=True):
    return dict(placement_mode="world", pallet_x=0., pallet_z=distance,
                pallet_heading=180., forklift_x=lateral, forklift_z=0.,
                forklift_heading=math.degrees(math.atan2(-lateral, distance)) if facing else 0.,
                latency=0., latency_sd=0., dropout=0., model_hz=30., hz=30.,
                model_interval_sd=0., drive_scale=1., rotation_scale=1.,
                drive_scale_sd=0., rotation_scale_sd=0., drive_delay=0.,
                drive_coast_sec=0., rotation_coast_fraction=0.,
                imu_scale=1., imu_bias_deg_s=0., imu_noise_deg_s=0.,
                yaw_noise=0., position_noise=0., x_noise=0., y_noise=0., z_noise=0.,
                yaw_bias=0., x_bias=0., y_bias=0., z_bias=0.,
                vertical_offset=.425, face_selection="fixed", fork_width=.115,
                max_seconds=185., seed=20260909)


class Experiment:
    def __init__(self, output, disable_predictive=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.rows = []
        self.overrides = {'COARSE_IMU_ENABLED': False}
        if disable_predictive:
            self.overrides['FWD_PREDICTIVE_MAX_ADVANCE_M'] = 0.
        self.allowed_simulator_ids = {SIMULATOR_ID}
        prior = self.output/'provenance.json'
        if prior.exists():
            previous = json.loads(prior.read_text(encoding='utf8'))
            if previous['simulator_id'] != SIMULATOR_ID:
                # Known concurrent browser-only update: Session source is byte
                # identical after removing two newly fingerprinted, unused files.
                changed = {k for k,v in previous['simulator_files'].items()
                           if SIMULATOR_HASHES.get(k) != v}
                added = set(SIMULATOR_HASHES)-set(previous['simulator_files'])
                normalized = (HERE/'process_runtime.py').read_bytes().replace(
                    b",'batched_session.py','batch_worker.py'", b'')
                if not (changed <= {'process_runtime.py','serve_v4.py'}
                        and added <= {'batched_session.py','batch_worker.py'}
                        and hashlib.sha256(normalized).hexdigest() == previous['simulator_files']['process_runtime.py']):
                    raise RuntimeError('Simulation behaviour may have changed; use a new output folder')
                self.allowed_simulator_ids.add(previous['simulator_id'])
                bridge = dict(old_simulator_id=previous['simulator_id'], new_simulator_id=SIMULATOR_ID,
                    changed_files=sorted(changed), added_manifest_files=sorted(added),
                    session_source_byte_equivalent_after_manifest_removal=True,
                    reason='Direct Session does not import serve_v4, batched_session, or batch_worker',
                    new_simulator_files=SIMULATOR_HASHES)
                (self.output/'manifest_bridge.json').write_text(json.dumps(bridge,indent=2),encoding='utf8')
        for path in sorted(self.output.glob("trial_*.json")):
            row = json.loads(path.read_text(encoding="utf8"))
            if row['overrides'] != self.overrides:
                raise ValueError('Different experiment overrides; use another output folder')
            if row['simulator_id'] not in self.allowed_simulator_ids:
                raise ValueError('Unrecognized cached simulator version')
            self.rows.append(row)
            self.cache[(row['distance_m'], row['lateral_m'], row['facing'])] = row

    def trial(self, distance, lateral, facing=True):
        lateral = round(lateral, 7)
        key = (distance, lateral, facing)
        if key in self.cache:
            return self.cache[key]
        start = time.perf_counter()
        session = Session(placement(distance, lateral, facing),
                          self.overrides, capture=True, log_root=None)
        report = session.run()
        if report['controller_error']:
            raise RuntimeError(report['controller_error'])
        changes = []
        for frame in report['frames']:
            if not changes or changes[-1]['state'] != frame['state']:
                changes.append({k: frame.get(k) for k in ('t', 'state', 'truth', 'lines')})
        assert not any(c['state'].startswith('COARSE_') for c in changes)
        row = dict(distance_m=distance, lateral_m=lateral, facing=facing,
                   wall_seconds=time.perf_counter()-start, initial=session.initial,
                   changes=changes, **{k: report[k] for k in (
                       'success', 'state', 'elapsed', 'reasons', 'failure_reason',
                       'failure_state', 'collision', 'final', 'options', 'overrides',
                       'source_id', 'simulator_id', 'trace_hash', 'true_remaining')})
        if self.rows:
            assert row['source_id'] == self.rows[0]['source_id'], 'FSM changed; use a new output folder'
            assert row['simulator_id'] in self.allowed_simulator_ids, 'Simulator changed; use a new output folder'
        else:
            (self.output/'provenance.json').write_text(
                json.dumps(report['provenance'], indent=2, ensure_ascii=False), encoding='utf8')
        self.rows.append(row)
        self.cache[key] = row
        (self.output/f'trial_{len(self.rows):04d}.json').write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding='utf8')
        print(json.dumps({k: row[k] for k in ('distance_m', 'lateral_m', 'facing',
              'success', 'state', 'elapsed', 'reasons', 'wall_seconds')}), flush=True)
        return row

    def boundary(self, distance, side, tolerance=.01):
        if not self.trial(distance, 0.)['success']:
            return dict(distance_m=distance, side=side, status='zero_lateral_failed')
        lo, hi = 0., .1
        while self.trial(distance, side*hi)['success']:
            lo = hi
            hi *= 2
            if hi > 6.4 + 1e-9:
                return dict(distance_m=distance, side=side, status='unbracketed', success_m=lo)
        initial_failure = hi
        while hi-lo > tolerance:
            mid = (lo+hi)/2
            if self.trial(distance, side*mid)['success']:
                lo = mid
            else:
                hi = mid
        # Sparse audit, not an exhaustive grid. More distant successes indicate
        # a disconnected feasible set; inner failures invalidate the envelope.
        for value in (.25*lo, .5*lo, .75*lo, .9*lo, hi+.02, initial_failure*1.25):
            self.trial(distance, side*value)
        samples = sorted((abs(r['lateral_m']), r['success']) for r in self.rows
                         if r['distance_m']==distance and r['facing']
                         and r['lateral_m']*side >= 0)
        failures = [x for x, ok in samples if not ok]
        first_failure = min(failures)
        inversions = [(x, ok) for x, ok in samples if ok and x > first_failure]
        guarded_lo, guarded_hi = lo, hi
        if inversions:
            # Refine the first observed failure, rather than selecting a later
            # successful island. Sparse probes cannot prove global continuity.
            for _ in range(5):
                guarded_hi = min(x for x, ok in samples if not ok)
                guarded_lo = max(x for x, ok in samples if ok and x < guarded_hi)
                while guarded_hi-guarded_lo > tolerance:
                    mid = (guarded_hi+guarded_lo)/2
                    if self.trial(distance, side*mid)['success']:
                        guarded_lo=mid
                    else:
                        guarded_hi=mid
                for value in (.25*guarded_lo, .5*guarded_lo, .75*guarded_lo, .9*guarded_lo):
                    self.trial(distance, side*value)
                samples = sorted((abs(r['lateral_m']), r['success']) for r in self.rows
                                 if r['distance_m']==distance and r['facing']
                                 and r['lateral_m']*side >= 0)
                first_failure=min(x for x, ok in samples if not ok)
                if first_failure > guarded_lo:
                    break
            else:
                raise RuntimeError('Repeated interior failures: no stable sampled envelope')
        return dict(distance_m=distance, side=side,
                    status='nonmonotonic' if inversions else 'sampled_monotonic',
                    success_m=lo, failure_m=hi, resolution_m=hi-lo,
                    conservative_success_m=guarded_lo, conservative_failure_m=guarded_hi,
                    first_sampled_failure_m=first_failure, samples=samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='verification/lateral_zero_error_20260909')
    parser.add_argument('--distances', type=float, nargs='+', default=[3.,3.5,4.,4.5,5.])
    parser.add_argument('--disable-predictive', action='store_true',
                        help='simulation-only: no predicted extra forward travel at STOP')
    args = parser.parse_args()
    experiment = Experiment(args.output, args.disable_predictive)
    results = []
    for distance in args.distances:
        for side in (1, -1):
            result = experiment.boundary(distance, side)
            results.append(result)
            print('BOUNDARY '+json.dumps(result), flush=True)
            (experiment.output/'boundaries.json').write_text(
                json.dumps(results, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
