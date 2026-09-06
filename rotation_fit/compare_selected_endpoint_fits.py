"""Compare both methods on exactly the same held-out endpoint observations."""
import csv
import json
from pathlib import Path
import numpy as np
from rotation_fit.rotation_log_fit import FitSettings, discover_and_analyse, _fit_once
from rotation_fit.fit_settled_endpoint_model import metrics


def main():
    run=Path(__file__).resolve().parent/'out/c4_command_clips_20260906'
    output=run/'user_selected_refit'
    direct=json.loads((output/'direct_fit/endpoint_model.json').read_text(encoding='utf-8'))
    with (output/'direct_fit/predictions.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    settings=FitSettings()
    segments,_=discover_and_analyse(run.parent/'command_clips',results_dir=run/'inference',
                                   results_suffix='_new_pose.csv',recording_names=direct['selected_recordings'],
                                   valid_columns=['pose_ok'],confidence_column='confidence',min_confidence=.3,
                                   model_id_column='model_hash',settings=settings)
    predictions=[]
    for name in direct['usable_recordings']:
        # All data belonging to the held-out recording is excluded from legacy fitting too.
        _,_,_,model=_fit_once([s for s in segments if s.recording!=name],settings)
        if model is None:
            raise RuntimeError(f'Legacy model not identifiable with {name} held out')
        for row in rows:
            if row['recording']==name:
                y=float(row['rotation_magnitude_deg'])
                legacy=model.total_angle(float(row['command_duration_s']))
                predictions.append(dict(recording=name,step=row['step'],hold_sec=float(row['command_duration_s']),
                                        observed_deg=y,legacy_loo_prediction_deg=legacy,
                                        direct_loo_prediction_deg=float(row['loo_predicted_deg'])))
    results={}
    y=np.array([r['observed_deg'] for r in predictions])
    for method in ('legacy','direct'):
        p=np.array([r[f'{method}_loo_prediction_deg'] for r in predictions])
        fold_rmse={}
        for name in direct['usable_recordings']:
            mask=np.array([r['recording']==name for r in predictions])
            fold_rmse[name]=float(np.sqrt(np.mean((p[mask]-y[mask])**2)))
        results[method]=dict(**metrics(y,p),equal_recording_weight_rmse_deg=float(np.sqrt(np.mean(np.square(list(fold_rmse.values()))))),
                             rmse_by_recording=fold_rmse)
    result=dict(test_observations='Identical 32 stable-endpoint measurements; same 9 held-out recordings; neither method trains on the held-out recording',
                training='Both use the exact user-selected 15 recordings; legacy learns usable motion traces, direct learns stable endpoint pairs',
                results=results,predictions=predictions)
    (output/'same_endpoint_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':main()
