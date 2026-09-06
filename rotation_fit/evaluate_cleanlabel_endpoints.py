"""Reinfer command clips using the pinned cleanlabel contract and fit two offline models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from rotation_fit.batch_infer_rotation_logs import Detection, YoloPosePredictor, run_batch_inference
from rotation_fit.extract_settled_rotation_pairs import extract
from rotation_fit.fit_delayed_linear_endpoint_model import main as fit_linear
from rotation_fit.fit_settled_endpoint_model import main as fit_selected

REVISION = 'c40e610f46331fc385a7ec6b0ff41876142c075c'
WEIGHT_SHA256 = '4ee578e02810caae56b786c23a084121f4dfb33941e11d4d3fb216b9bfbbe60e'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_json(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def save_csv(path, rows):
    if rows:
        with path.open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class CleanlabelPredictor(YoloPosePredictor):
    """Preserve original camera coordinates after mandatory reflect padding."""
    def predict(self, image):
        if image.shape != (480, 640, 3):
            raise ValueError(f'Expected model contract 640x480 BGR, got {image.shape}')
        padded = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
        result = self.model.predict(source=padded, imgsz=640, conf=.4,
                                    device=self.device, verbose=False,
                                    quantize=16 if self.use_half else None,
                                    augment=False)[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            return Detection(None, None, None)
        confidence = result.boxes.conf.detach().cpu().numpy()
        index = int(np.argmax(confidence))
        points = result.keypoints.data[index].detach().cpu().numpy().copy()
        if points.shape != (9, 3):
            raise ValueError(f'Expected nine (x,y,confidence) keypoints, got {points.shape}')
        points[:, :2] -= 100
        box = result.boxes.xyxy[index].detach().cpu().numpy().copy() - 100
        return Detection(points, float(confidence[index]), box)


def screen_logs(recordings, run):
    """Same >5deg adjacent-step screening as C4 audit, without phase-fit dependency."""
    from matplotlib import pyplot as plt
    batch = read_json(run / 'inference/batch_inference_manifest.json')
    clips = read_json(recordings / 'command_clips_manifest.json')
    selection = {r['prefix']: r for r in clips['recordings']}
    output = run / 'audit'
    output.mkdir(exist_ok=False)
    ranking, jumps = [], []
    for record in batch['recordings']:
        prefix = record['prefix']
        rows = read_csv(run / 'inference' / record['result_file'])
        timing = {int(r['frame_i']): r for r in read_csv(recordings / f'{prefix}_inference_timing.csv')}
        t = np.array([float(timing[int(r['frame_i'])]['camera_input_host_mono_ms']) / 1000 for r in rows])
        yaw = np.array([float(r['yaw_deg']) if r['yaw_deg'] else np.nan for r in rows])
        valid = np.array([r['pose_ok'] == '1' and float(r['confidence'] or 0) >= .4 for r in rows]) & np.isfinite(yaw)
        yaw[~valid] = np.nan
        dt = np.diff(t)
        adjacent = valid[1:] & valid[:-1] & (dt > 0) & (dt <= .35)
        delta = (np.diff(yaw) + 180) % 360 - 180
        indices = np.flatnonzero(adjacent & (abs(delta) > 5))
        for i in indices:
            jumps.append(dict(recording=prefix, frame_before=rows[i]['frame_i'],
                              frame_after=rows[i+1]['frame_i'], dt_s=float(dt[i]), delta_deg=float(delta[i])))
        ranking.append(dict(recording=prefix.removeprefix('forklift_v4_recording_'),
                            frames=len(rows), valid_poses=int(valid.sum()), jumps_over_5deg=len(indices),
                            jumps_over_30deg=int(np.sum(adjacent & (abs(delta) > 30)))))
        origin = t[0]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.scatter(t - origin, yaw, s=6)
        for window in selection[prefix]['command_windows']:
            ax.axvspan(window['command_host_s']-origin, window['stop_host_s']-origin, color='green', alpha=.13)
            ax.axvline(window['stop_host_s']+2-origin, color='orange', lw=.7, alpha=.4)
        ax.set(xlabel='Original host time since first retained frame (s)', ylabel='Raw PnP yaw (deg)',
               title=f"Cleanlabel: {ranking[-1]['recording']} | green=CAN, orange=STOP+2s")
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / f'{prefix}_yaw.png', dpi=120)
        plt.close(fig)
    report = dict(model_hash=batch['model_hash'], frames=batch['total_frames'],
                  valid_poses=batch['total_valid_poses'], ranking=ranking,
                  jump_event_count=len(jumps), method='Wrapped yaw step >5deg; adjacent valid rows with 0<dt<=.35s; do not bridge gaps',
                  deployment_performed=False)
    save_json(output / 'summary.json', report)
    save_csv(output / 'recording_quality.csv', ranking)
    save_csv(output / 'jump_events.csv', jumps)


def run(model, recordings, output, selection_report, analysis_only=False):
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != WEIGHT_SHA256:
        raise ValueError('Downloaded weights do not match pinned repository SHA256SUMS')
    for meta in recordings.glob('*_meta.json'):
        geometry = read_json(meta)['pallet_size_m']
        if not all(np.isclose(geometry[key], expected) for key, expected in
                   [('width', 1.1), ('length', 1.1), ('height', .15)]):
            raise ValueError(f'Geometry incompatible with this model: {meta}')
    if not analysis_only:
        output.mkdir(parents=True, exist_ok=False)
        predictor = CleanlabelPredictor(model, device='cuda:0', confidence_threshold=.4,
                                        front_class_name='item', use_half=True)
        if len(predictor.model.names) != 1:
            raise ValueError('Cleanlabel contract requires a single class')
        save_json(output / 'provenance.json', dict(
            repository='https://huggingface.co/CanelE452/pallet-pose-yolo26n-cleanlabel',
            revision=REVISION, model_sha256=digest, padding_px=100, padding_mode='BORDER_REFLECT_101',
            subtract_padding_before_pnp=True, imgsz=640, confidence=.4, instance_selection='max_box_conf',
            pnp='Existing live multi-face PnP solver retained; measured intrinsics and dimensions from each recording',
            angle_column='yaw_deg', angle_domain='heading',
            selection_source=str(selection_report.resolve()),
            linear_excludes_duration_at_or_above_s=5,
            selected_method='Previous 15-video selection, endpoint checks, recording-balanced isotonic interpolation; retains durations >=5s as in original selected fit',
            deployment_performed=False))
        run_batch_inference(model_path=model, recordings_dir=recordings, results_dir=output / 'inference',
                            device='cuda:0', confidence_threshold=.4, use_half=True,
                            predictor=predictor, overwrite=False, allow_low_pose=True)
    batch = read_json(output / 'inference/batch_inference_manifest.json')
    if batch['model_hash'] != 'sha256:' + digest:
        raise ValueError('Inference model identity mismatch')
    if not (output / 'audit').exists():
        screen_logs(recordings, output)
    if not (output / 'settled_endpoints').exists():
        extract(recordings, output, output / 'settled_endpoints')
    if not (output / 'linear_under5s').exists():
        fit_linear(output / 'settled_endpoints/accepted_command_pairs.csv', output / 'linear_under5s', 5.)
    selected_names = read_json(selection_report)['selected_recordings']
    if len(selected_names) != 15:
        raise ValueError('Expected previous 15-video selection')
    if not (output / 'selected_endpoints').exists():
        extract(recordings, output, output / 'selected_endpoints', selected_names)
    if not (output / 'selected_fit').exists():
        fit_selected(output / 'selected_endpoints', output / 'selected_fit')
    print('COMPLETE: two offline fit artifacts; no deployment', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--recordings', type=Path, default=Path('rotation_fit/out/command_clips'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-report', type=Path, default=Path('rotation_fit/out/c4_command_clips_20260906/user_selected_refit/endpoints/endpoint_extraction_report.json'))
    parser.add_argument('--analysis-only', action='store_true')
    args = parser.parse_args()
    run(args.model, args.recordings, args.output, args.selection_report, args.analysis_only)
