"""Monotone endpoint fit with recording-grouped validation; no runtime deployment."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_knots(times, angles, groups):
    times, angles, groups = np.asarray(times,float), np.asarray(angles,float), np.asarray(groups)
    if not len(times) or len(times)!=len(angles) or len(times)!=len(groups):
        raise ValueError("Expected matching, nonempty input arrays")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(angles)) or np.any(times<0) or np.any(angles<0):
        raise ValueError("Times and angle magnitudes must be finite and nonnegative")
    counts = Counter(groups)
    weights = [1 / counts[g] for g in groups]
    # Exact zero-command/zero-motion anchor. Every recording has total weight 1.
    fitted = IsotonicRegression(increasing=True, y_min=0, out_of_bounds="clip").fit(
        np.r_[0.,times], np.r_[0.,angles], sample_weight=np.r_[1.,weights])
    return fitted.X_thresholds_.tolist(), fitted.y_thresholds_.tolist()


def predict(knots, time_s):
    x,y = knots
    if not np.isfinite(time_s) or not x[0]<=time_s<=x[-1]:
        raise ValueError("Command time outside fitted support")
    return float(np.interp(time_s,x,y))


def earliest_duration(knots, angle_deg):
    x,y=map(np.asarray,knots)
    if not np.isfinite(angle_deg) or not y[0]<=angle_deg<=y[-1]:
        raise ValueError("Target angle outside fitted range")
    right=int(np.searchsorted(y,angle_deg,side="left"))
    if right==0 or y[right]==angle_deg:
        return float(x[right])
    left=right-1
    return float(x[left]+(angle_deg-y[left])/(y[right]-y[left])*(x[right]-x[left]))


def metrics(observed,predicted):
    observed,predicted=np.asarray(observed),np.asarray(predicted)
    error=predicted-observed
    ss=float(np.sum((observed-observed.mean())**2))
    return dict(n=len(error),rmse_deg=float(np.sqrt(np.mean(error**2))),
                mae_deg=float(np.mean(np.abs(error))),bias_deg=float(error.mean()),
                r2=float(1-np.sum(error**2)/ss) if ss>0 else None,
                p90_absolute_error_deg=float(np.percentile(np.abs(error),90)))


def write_csv(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(endpoint_dir,output,bootstrap_runs=200):
    with (endpoint_dir/'accepted_command_pairs.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    source=json.loads((endpoint_dir/'endpoint_extraction_report.json').read_text(encoding='utf-8'))
    if not rows or any(r['accepted']!='True' for r in rows):
        raise ValueError('Expected accepted endpoint observations')
    if {r['model_hash'] for r in rows}!={source['model_hash']}:
        raise ValueError('Mixed or mismatched model hashes')
    output.mkdir(parents=True,exist_ok=False)
    t=np.array([float(r['command_duration_s']) for r in rows])
    y=np.array([float(r['rotation_magnitude_deg']) for r in rows])
    g=np.array([r['recording'] for r in rows])
    names=sorted(set(g))
    if len(names)<4:
        raise ValueError('Need at least four independent recording groups')
    knots=fit_knots(t,y,g)
    fitted=np.interp(t,*knots)
    heldout=np.full(len(t),np.nan)
    out_of_support=np.zeros(len(t),dtype=bool)
    folds=[]
    for name in names:
        train,test=g!=name,g==name
        # Refitting includes no outcomes from the held-out recording.
        local=fit_knots(t[train],y[train],g[train])
        heldout[test]=np.interp(t[test],*local)
        unsupported=(t[test]<t[train].min()) | (t[test]>t[train].max())
        out_of_support[test]=unsupported
        folds.append(dict(recording=name,test_count=int(test.sum()),
                          training_min_hold_sec=float(t[train].min()),training_max_hold_sec=float(t[train].max()),
                          outside_training_hold_range=int(unsupported.sum()),
                          **metrics(y[test],heldout[test])))
    equal_rmse=float(np.sqrt(np.mean([f['rmse_deg']**2 for f in folds])))
    for row,fit_value,held_value,outside in zip(rows,fitted,heldout,out_of_support):
        row.update(fitted_deg=float(fit_value),loo_predicted_deg=float(held_value),
                   loo_error_deg=float(held_value)-float(row['rotation_magnitude_deg']),
                   loo_outside_measured_training_hold_range=bool(outside))
    write_csv(output/'predictions.csv',rows)
    write_csv(output/'recording_cross_validation.csv',folds)
    write_csv(output/'function_knots.csv',[dict(command_duration_s=x,predicted_total_rotation_deg=a) for x,a in zip(*knots)])
    targets=[a for a in (1,2,3,5,8,10,12,15) if knots[1][0]<=a<=knots[1][-1]]
    inverse=[dict(target_deg=a,diagnostic_earliest_duration_s=earliest_duration(knots,a)) for a in targets]
    write_csv(output/'diagnostic_inverse_table.csv',inverse)
    grid=np.linspace(0,t.max(),300)
    rng=np.random.default_rng(20260906)
    boot=[]
    for _ in range(bootstrap_runs):
        sampled=rng.choice(names,size=len(names),replace=True)
        bt,by,bg=[],[],[]
        for draw,name in enumerate(sampled):
            mask=g==name
            bt.extend(t[mask]); by.extend(y[mask]); bg.extend([str(draw)]*int(mask.sum()))
        local=fit_knots(bt,by,bg)
        curve=np.interp(grid,*local)
        # Bootstrap intervals do not claim support where a resample has no data.
        curve[(grid<min(bt)) | (grid>max(bt))]=np.nan
        boot.append(curve)
    band_low=np.full(len(grid),np.nan); band_high=band_low.copy(); band_count=np.zeros(len(grid),dtype=int)
    if boot:
        boot=np.asarray(boot)
        for i in range(len(grid)):
            values=boot[:,i][np.isfinite(boot[:,i])]
            band_count[i]=len(values)
            if len(values)>=max(20,int(.75*bootstrap_runs)):
                band_low[i],band_high[i]=np.percentile(values,[2.5,97.5])
    report=dict(model_type='monotone_piecewise_linear_hold_to_total_rotation',model_hash=source['model_hash'],
                selected_recordings=source['selected_recordings'],usable_recordings=names,
                command_count=len(rows),selected_command_count=source['total_commands'],
                method='Isotonic least squares; each recording total weight 1; zero anchor; linear interpolation between knots; fixed method before evaluation',
                in_sample=metrics(y,fitted),leave_one_recording_out=metrics(y,heldout),
                equal_recording_weight_loo_rmse_deg=equal_rmse,rmse_gate_deg=3.,rmse_gate_passed=equal_rmse<=3.,
                folds=folds,loo_outside_training_hold_range_count=int(out_of_support.sum()),
                loo_outside_range_policy='Boundary-clamped predictions are included in headline error and explicitly flagged; not evidence of valid extrapolation',
                measured_hold_range_sec=[float(t.min()),float(t.max())],
                measured_angle_range_deg=[float(y.min()),float(y.max())],
                knots=[dict(hold_sec=x,angle_deg=a) for x,a in zip(*knots)],
                function='theta(T)=linear interpolation of listed monotone knots; inverse returns earliest T attaining target; out-of-range requests rejected',
                deployment_performed=False,safe_for_control=False,
                limitations=['Endpoint angle is C4/PnP observation, not physical ground truth.',
                             'Physical face consistency at endpoints is not independently verified.',
                             'User selected videos after inspecting logs; validation is conditional on this selection.',
                             'Response-model leave-one-recording-out does not retrain the C4 perception model.',
                             'Long unsampled intervals between knots are not validated merely by interpolation.',
                             'This artifact is not compatible with the existing physical phase-model runtime loader.'],
                bootstrap_runs=bootstrap_runs)
    (output/'endpoint_model.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+ '\n',encoding='utf-8')
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font)); plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(13,5.7))
    for direction,color,marker,label in (('LEFT','#2378b5','o','좌회전'),('RIGHT','#db7732','^','우회전')):
        mask=np.array([r['direction']==direction for r in rows])
        axes[0].scatter(t[mask],y[mask],color=color,marker=marker,s=40,alpha=.8,label=label)
    axes[0].plot(grid,np.interp(grid,*knots),color='#202020',lw=1.8,label='최종 회전량 직접 적합')
    if boot is not None and bootstrap_runs:
        axes[0].fill_between(grid,band_low,band_high,color='#777777',alpha=.16,label='녹화 bootstrap 95% 계수 변동')
    axes[0].set(xlabel='실제 CAN 명령 유지 시간 (초)',ylabel='최종 회전량 (도, C4 추정)',title=f'선택 영상 직접 적합: {len(rows)}개 명령 / {len(names)}개 녹화')
    axes[0].set_xlim(left=0); axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=9)
    axes[1].scatter(y,heldout,s=42,alpha=.8,label='녹화 전체를 제외하고 예측')
    if out_of_support.any():
        axes[1].scatter(y[out_of_support],heldout[out_of_support],s=110,facecolors='none',edgecolors='red',label='학습 명령 시간 범위 밖')
    limit=max(float(y.max()),float(heldout.max()))*1.05
    axes[1].plot([0,limit],[0,limit],'k--',lw=1)
    axes[1].set(xlabel='관측 최종 회전량 (도)',ylabel='교차검증 예측 회전량 (도)',title=f'녹화 동일 가중 RMSE = {equal_rmse:.2f}° (기준 3°)')
    axes[1].legend(fontsize=9)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'fit_and_validation.png',dpi=155)
    plt.close(fig)
    print(json.dumps({k:report[k] for k in ('command_count','in_sample','leave_one_recording_out','equal_recording_weight_loo_rmse_deg','rmse_gate_passed','loo_outside_training_hold_range_count','measured_hold_range_sec')},ensure_ascii=False))
    print('KNOTS',report['knots'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--bootstrap-runs',type=int,default=200)
    args=parser.parse_args()
    main(args.endpoint_dir,args.output_dir,args.bootstrap_runs)
