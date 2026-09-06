"""Extract CAN hold-time vs stable-endpoint yaw pairs without fitting motion traces."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from rotation_fit.rotation_log_fit import FitSettings, load_command_windows, load_frame_series, robust_sigma


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_stats(frames, results, mask):
    indices = np.flatnonzero(mask)
    if len(indices) < 5:
        return None, ["fewer_than_5_valid_frames"]
    t, angle = frames.analysis_time_s[indices], frames.angle_deg[indices]
    span = float(t[-1] - t[0])
    if span <= 0:
        return None, ["nonpositive_time_span"]
    centered = t - t.mean()
    slope = float(np.polyfit(centered, angle, 1)[0])
    sigma = float(robust_sigma(angle))
    width = float(np.percentile(angle, 90) - np.percentile(angle, 10))
    max_step = float(np.max(np.abs(np.diff(angle))))
    xy = np.array([[float(results[int(frames.frame_i[i])][k]) for k in ("pos_x_m", "pos_z_m")] for i in indices])
    if not np.all(np.isfinite(xy)):
        return None, ["nonfinite_position"]
    position_rate = float(np.linalg.norm(np.polyfit(centered, xy, 1)[0]))
    stats = dict(first_frame_i=int(frames.frame_i[indices[0]]), last_frame_i=int(frames.frame_i[indices[-1]]),
                 first_time_s=float(t[0]), last_time_s=float(t[-1]), frame_count=len(indices),
                 span_s=span, median_yaw_deg=float(np.median(angle)),
                 mad_sigma_deg=sigma, p90_p10_deg=width, slope_deg_s=slope,
                 max_step_deg=max_step, position_rate_m_s=position_rate,
                 max_gap_s=float(np.max(np.diff(t))))
    reasons = []
    if span < .60:
        reasons.append("stable_window_shorter_than_0.60s")
    if stats["max_gap_s"] > .35:
        reasons.append("frame_gap_above_0.35s")
    if abs(slope) > 1.0:
        reasons.append("yaw_rate_above_1deg_s")
    if sigma > .5:
        reasons.append("yaw_noise_above_0.5deg")
    if width > 1.5:
        reasons.append("yaw_p90_p10_above_1.5deg")
    if max_step > 2:
        reasons.append("yaw_step_above_2deg")
    if position_rate > .05:
        reasons.append("position_rate_above_0.05m_s")
    return stats, reasons


def extract(recordings, run, output, recording_names=None):
    quality = json.loads((run / "audit/summary.json").read_text(encoding="utf-8"))
    batch = json.loads((run / "inference/batch_inference_manifest.json").read_text(encoding="utf-8"))
    if quality["model_hash"] != batch["model_hash"]:
        raise ValueError("Quality screening model differs from inference model")
    selected = set(recording_names) if recording_names is not None else None
    if selected is not None and (not selected or selected - set(batch["successful_recording_prefixes"])):
        raise ValueError("Selected recordings must be present in the inference manifest")
    output.mkdir(parents=True, exist_ok=False)
    excluded_recordings = {"forklift_v4_recording_" + r["recording"] for r in quality["ranking"]
                           if r["jumps_over_5deg"] >= 2}
    if selected is not None:
        # An explicit human-reviewed list overrides prior whole-video screening.
        # Endpoint quality checks still apply independently to each command.
        excluded_recordings = set()
    rows, window_audit, failure_counts = [], [], Counter()
    frame_errors = {}
    for prefix in batch["successful_recording_prefixes"]:
        if selected is not None and prefix not in selected:
            continue
        windows = load_command_windows(recordings / f"{prefix}_control_seq.jsonl", FitSettings())
        result_path = run / "inference" / f"{prefix}_new_pose.csv"
        results = {int(r["frame_i"]): r for r in read_rows(result_path)}
        try:
            frames = load_frame_series(recordings / f"{prefix}_inference_timing.csv", result_path,
                                       angle_column="yaw_deg", valid_columns=["pose_ok"],
                                       confidence_column="confidence", min_confidence=.3,
                                       model_id_column="model_hash")
        except ValueError as exc:
            frames = None
            frame_errors[prefix] = str(exc)
        for window in windows:
            if window.strength != 30 or window.start.source != "can_tx":
                continue
            start, stop = window.start.write_return_s, window.stop.write_return_s
            record = dict(recording=prefix, step=window.step, direction=window.direction,
                          model_hash=batch["model_hash"], command_start_host_s=start,
                          command_stop_host_s=stop, command_start_iso=window.start.t_iso,
                          command_stop_iso=window.stop.t_iso, command_duration_s=stop-start,
                          start_angle_deg=None, end_angle_deg=None, signed_rotation_deg=None,
                          rotation_magnitude_deg=None, endpoint_noise_scale_deg=None,
                          start_first_frame_i=None, start_last_frame_i=None,
                          end_first_frame_i=None, end_last_frame_i=None,
                          end_observation_after_stop_s=None, accepted=False, exclusion_reasons="")
            reasons, before, after = [], None, None
            if prefix in excluded_recordings:
                reasons.append("recording_has_repeated_inference_anomalies")
            if window.stop.movement != "stop":
                reasons.append("rotation_not_terminated_by_STOP")
            if frames is None:
                reasons.append("insufficient_valid_pose_log")
            else:
                times = frames.analysis_time_s
                before, pre_reasons = stable_stats(frames, results, (times >= start-1.0) & (times < start))
                reasons += ["pre_" + reason for reason in pre_reasons]
                limit = stop + 2.7
                if window.next_active_command_s is not None:
                    limit = min(limit, window.next_active_command_s - .1)
                post_candidates = np.flatnonzero((times > stop) & (times <= limit))
                if len(post_candidates):
                    last = times[post_candidates[-1]]
                    if last-stop < 1.5:
                        reasons.append("post_observation_ends_before_STOP_plus_1.5s")
                    after, post_reasons = stable_stats(frames, results, (times >= max(stop, last-1.0)) & (times <= last))
                    reasons += ["post_" + reason for reason in post_reasons]
                else:
                    reasons.append("no_post_STOP_frames")
            if before is not None and after is not None:
                signed = (after["median_yaw_deg"]-before["median_yaw_deg"]+180)%360-180
                record.update(start_angle_deg=before["median_yaw_deg"], end_angle_deg=after["median_yaw_deg"],
                              signed_rotation_deg=signed, rotation_magnitude_deg=abs(signed),
                              endpoint_noise_scale_deg=float(np.hypot(before["mad_sigma_deg"], after["mad_sigma_deg"])),
                              start_first_frame_i=before["first_frame_i"], start_last_frame_i=before["last_frame_i"],
                              end_first_frame_i=after["first_frame_i"], end_last_frame_i=after["last_frame_i"],
                              end_observation_after_stop_s=after["last_time_s"]-stop)
            record["accepted"] = not reasons
            record["exclusion_reasons"] = "; ".join(reasons)
            failure_counts.update(reasons)
            rows.append(record)
            window_audit.append(dict(recording=prefix, step=window.step, before=before, after=after))
    accepted = [r for r in rows if r["accepted"]]
    for name, values in (("all_command_pairs.csv", rows), ("accepted_command_pairs.csv", accepted)):
        with (output / name).open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(values)
    thresholds = dict(pre_window_s=1, post_tail_window_s=1, post_max_horizon_s=2.7,
                      post_last_observation_min_s=1.5, min_frames=5, min_span_s=.60,
                      max_gap_s=.35, max_abs_yaw_rate_deg_s=1, max_yaw_mad_sigma_deg=.5,
                      max_yaw_p90_p10_deg=1.5, max_yaw_step_deg=2, max_position_rate_m_s=.05,
                      repeated_anomaly_policy=("Explicit user selection; no automatic whole-video anomaly exclusions" if selected is not None else
                      "Whole recording excluded if >=2 adjacent >5deg changes in previous audit (a spike and return count separately)"))
    report = dict(model_hash=batch["model_hash"], selection_thresholds=thresholds,
                  selected_recordings=sorted(selected) if selected is not None else batch["successful_recording_prefixes"],
                  total_commands=len(rows), accepted_commands=len(accepted),
                  accepted_recordings=len({r["recording"] for r in accepted}),
                  excluded_recordings=sorted(excluded_recordings),
                  exclusion_reason_counts=dict(failure_counts), frame_errors=frame_errors,
                  windows=window_audit,
                  limitations=["Rotation is a C4/PnP visual estimate, not external angle ground truth.",
                               "Stationary windows are visual candidates; stability does not prove zero motion or a correct physical face.",
                               "Existing logs do not contain selected-face IDs; equal physical face at endpoints is not independently verified.",
                               "Observations only extend to STOP+2.7s; cannot prove rest after that time.",
                               "Whole-video rejection uses prior inspection, so this is descriptive data, not unbiased model validation."])
    (output / "endpoint_extraction_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    for direction, color, marker, label in (("LEFT", "#2378b5", "o", "좌회전"), ("RIGHT", "#db7732", "^", "우회전")):
        group = [r for r in accepted if r["direction"] == direction]
        ax.scatter([r["command_duration_s"] for r in group], [r["rotation_magnitude_deg"] for r in group],
                   s=65, alpha=.8, color=color, marker=marker, edgecolors="white", linewidths=.6,
                   label=f"{label} ({len(group)}개)")
    ax.set(xlabel="실제 CAN 명령 유지 시간 (초)", ylabel="정지 후보 구간 간 회전량 |Δyaw| (도)",
           title=f"C4 로그: 명령 시간과 최종 회전량\n안정 구간 조건 통과 {len(accepted)} / {len(rows)}개 명령")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.legend()
    selection_label = "사용자 선택 영상" if selected is not None else "반복 이상 영상 제외"
    fig.text(.5,.025,f"명령 전·종료 후 안정 구간의 중앙값 차이 · {selection_label} · 회전량은 영상 추정치",ha="center",fontsize=10,color="#555555")
    fig.tight_layout(rect=(0,.055,1,1))
    fig.savefig(output / "command_time_vs_rotation.png",dpi=170)
    fig.savefig(output / "command_time_vs_rotation.svg")
    plt.close(fig)
    print(json.dumps({k:report[k] for k in ("total_commands","accepted_commands","accepted_recordings","exclusion_reason_counts")},ensure_ascii=False),flush=True)
    print('ACCEPTED', json.dumps([dict(recording=r['recording'],step=r['step'],seconds=r['command_duration_s'],angle=r['rotation_magnitude_deg']) for r in accepted],ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-dir",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--recording-name",action="append",default=None)
    args = parser.parse_args()
    extract(args.recordings_dir,args.run_dir,args.output_dir,args.recording_name)
