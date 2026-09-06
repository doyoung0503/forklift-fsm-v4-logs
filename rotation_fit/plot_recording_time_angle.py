"""Plot every raw C4 yaw log as a point on the rotation fitter's time axis."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, transforms
import numpy as np

from rotation_fit.rotation_log_fit import load_frame_series


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_recording(record, recordings, inference, selections):
    prefix = record["prefix"]
    timing_path = recordings / f"{prefix}_inference_timing.csv"
    timing = read_csv(timing_path)
    by_id = {int(row["frame_i"]): row for row in timing}
    origin = min(float(row["camera_input_host_mono_ms"]) / 1000 for row in timing
                 if row["camera_input_host_mono_ms"].strip())
    result_path = inference / record["result_file"]
    rows = read_csv(result_path)
    valid = np.array([r["pose_ok"] == "1" and r["yaw_deg"] != "" and np.isfinite(float(r["yaw_deg"])) for r in rows])
    aligned = {}
    time_source = "camera_input_host_mono_ms"
    if valid.sum() >= 3:
        series = load_frame_series(timing_path, result_path, angle_column="yaw_deg", valid_columns=["pose_ok"],
                                   confidence_column="confidence", min_confidence=.3, model_id_column="model_hash")
        aligned = dict(zip(series.frame_i, series.analysis_time_s))
        time_source = series.frame_time_source
    times = np.array([aligned.get(int(r["frame_i"]), float(by_id[int(r["frame_i"])]["camera_input_host_mono_ms"])/1000) - origin for r in rows])
    # Preserve the yaw values from CSV exactly: no smoothing, unwrapping or rejection.
    angles = np.array([float(r["yaw_deg"]) if good else np.nan for r, good in zip(rows, valid)])
    assert len(rows) == record["frame_count"]
    assert int(valid.sum()) == record["valid_poses"]
    return dict(prefix=prefix, times=times, angles=angles, valid=valid, origin=origin,
                windows=selections[prefix]["command_windows"], time_source=time_source,
                count=len(rows), valid_count=int(valid.sum()))


def draw(ax, data, number, *, compact=False):
    t, yaw, valid = data["times"], data["angles"], data["valid"]
    for window in data["windows"]:
        lo, hi = window["command_host_s"]-data["origin"], window["stop_host_s"]-data["origin"]
        ax.axvspan(lo, hi, color="#6cc78b", alpha=.22, zorder=0)
        ax.axvline(hi+2, color="#cf8a29", lw=.65, alpha=.7, linestyle="--", zorder=1)
    ax.scatter(t[valid], yaw[valid], s=8 if compact else 15, c="#1e69b5", alpha=.85, linewidths=0, zorder=3)
    if (~valid).any():
        # Invalid frames have NO angle. Put ticks along the bottom border, not at yaw=0.
        ax.scatter(t[~valid], np.full(int((~valid).sum()), .018), transform=transforms.blended_transform_factory(ax.transData, ax.transAxes),
                   marker="|", s=24, c="#cb3434", linewidths=1, zorder=4)
    if not valid.any():
        ax.text(.5, .53, "유효한 각도 추론 없음", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="#b02727")
    label = data["prefix"].removeprefix("forklift_v4_recording_")
    ax.set_title(f'{number:02d}  {label}  |  유효 {data["valid_count"]}/{data["count"]}', fontsize=10 if compact else 13, loc="left")
    ax.set_xlabel("녹화 시작 기준 시간 (초)", fontsize=9 if compact else 11)
    ax.set_ylabel("원본 yaw (도)", fontsize=9 if compact else 11)
    # Retain real gaps between extracted command windows; do not concatenate their time axes.
    lo, hi = float(np.min(t)), float(np.max(t))
    margin = max(.15, (hi-lo)*.025)
    ax.set_xlim(lo-margin, hi+margin)
    if valid.any():
        bottom, top = float(np.nanmin(yaw)), float(np.nanmax(yaw))
        pad = max(.5, .08*(top-bottom))
        ax.set_ylim(bottom-pad, top+pad)
    ax.tick_params(labelsize=8 if compact else 10)
    ax.grid(alpha=.22)
    ax.set_axisbelow(True)


def plot(recordings, run, output):
    batch = json.loads((run/"inference/batch_inference_manifest.json").read_text(encoding="utf-8"))
    prepared = json.loads((recordings/"command_clips_manifest.json").read_text(encoding="utf-8"))
    selection = {r["prefix"]:r for r in prepared["recordings"]}
    output.mkdir(parents=True, exist_ok=False)
    font=Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    data=[load_recording(r,recordings,run/"inference",selection) for r in batch["recordings"]]
    index=["# 영상별 시간–각도 원본 점 그래프", "",
           "23개 영상 전체를 포함합니다. 원본 영상 전체를 새로 추론한 것이 아니라, CAN 구간 중심으로 분리한 4,163프레임의 C4 추론 로그입니다. 그래프 안에서 이상 영상·이상 각도를 제외하거나 평활화하지 않았습니다.", "",
           "X축은 원본 녹화 첫 프레임의 호스트 시각을 기준으로 한 경과 시간입니다. 유효 각도의 시간은 기존 적합기와 같은 센서→호스트 정렬 시각을 사용합니다. 잘라낸 구간의 시간 간격은 유지됩니다. Y축은 CSV의 yaw_deg 원래 값입니다(절댓값·누적 회전량·평활값 아님).",
           "", "파란 점: 유효한 각도 추론 / 녹색 배경: CAN 회전 명령 유지 / 주황 점선: STOP+2초 / 아래 빨간 눈금: 자세 추정 실패(각도값 없음). 영상마다 Y축 범위가 다릅니다.", ""]
    exports=[]
    for number,item in enumerate(data,1):
        name=item['prefix'].removeprefix('forklift_v4_recording_')
        fig,ax=plt.subplots(figsize=(12,5))
        draw(ax,item,number)
        fig.text(.5,.014,"파란 점: 원본 추론 | 녹색: CAN 회전 명령 | 주황 점선: STOP+2초 | 아래 빨간 눈금: 추론 실패",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.045,1,1))
        fig.savefig(output/f"{name}_scatter.png",dpi=145)
        fig.savefig(output/f"{name}_scatter.svg")
        plt.close(fig)
        index.extend([f"## {number:02d}. {name}", "", f"![{name}]({name}_scatter.png)", ""])
        exports.append(dict(recording=item['prefix'],count=item['count'],valid_count=item['valid_count'],
                            time_source=item['time_source'],origin_host_s=item['origin'],image=f"{name}_scatter.png"))
    overviews=[]
    for page,start in enumerate(range(0,len(data),6),1):
        group=data[start:start+6]
        fig,axes=plt.subplots(3,2,figsize=(13,10))
        for slot,(ax,item) in enumerate(zip(axes.flat,group)):
            draw(ax,item,start+slot+1,compact=True)
        for ax in list(axes.flat)[len(group):]:
            ax.axis('off')
        fig.suptitle(f"C4 실제 로그: 영상별 시간–각도 점 그래프 ({start+1}–{start+len(group)}/23)",fontsize=15)
        fig.text(.5,.018,"파란 점: 원본 yaw | 녹색: 회전 명령 | 주황: STOP+2초 | 아래 빨간 눈금: 추론 실패 | 영상별 Y축 범위 다름",ha="center",fontsize=10)
        fig.tight_layout(rect=(0,.05,1,.95),h_pad=2)
        name=f"overview_{page:02d}.png"
        fig.savefig(output/name,dpi=130)
        plt.close(fig)
        overviews.append(name)
    (output/"INDEX.md").write_text("\n".join(index),encoding="utf-8")
    (output/"plot_manifest.json").write_text(json.dumps(dict(model_hash=batch['model_hash'],
        total_rows=sum(d['count'] for d in data), valid_points=sum(d['valid_count'] for d in data),
        invalid_frames=sum(d['count']-d['valid_count'] for d in data),
        recordings=exports,overviews=overviews,angle_transform="none",smoothing=False,
        outliers_removed=False,source_scope="prepared command clips, not full original videos"),
        ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(recordings=len(data),valid_points=sum(d['valid_count'] for d in data),overviews=overviews,output=str(output.resolve()))))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings-dir',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    plot(args.recordings_dir,args.run_dir,args.output_dir)
