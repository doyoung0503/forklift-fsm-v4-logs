"""Read-only log screening plus reproducible charts for a rotation fit run."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


def quantile(values, q):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if len(values) else None


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(recordings, run):
    output = run / "audit"
    output.mkdir(exist_ok=False)
    batch = json.loads((run / "inference/batch_inference_manifest.json").read_text(encoding="utf-8"))
    fit = json.loads((run / "fit/fit_report.json").read_text(encoding="utf-8"))
    segments = read_csv(run / "fit/rotation_segments.csv")
    prepared = json.loads((recordings / "command_clips_manifest.json").read_text(encoding="utf-8"))
    selection = {item["prefix"]: item for item in prepared["recordings"]}
    ranking, jumps, plots = [], [], []
    for report in batch["recordings"]:
        prefix = report["prefix"]
        suffix = prefix.removeprefix("forklift_v4_recording_")
        rows = read_csv(run / "inference" / report["result_file"])
        timing = {int(row["frame_i"]): row for row in read_csv(recordings / f"{prefix}_inference_timing.csv")}
        t = np.asarray([number(timing[int(r["frame_i"])]["camera_input_host_mono_ms"]) / 1000 for r in rows])
        valid = np.asarray([r["pose_ok"] == "1" and number(r["confidence"]) >= .3 for r in rows])
        yaw = np.asarray([number(r["yaw_deg"]) for r in rows])
        yaw[~valid] = np.nan
        dt = np.diff(t)
        # Adjacent OUTPUT rows only: never bridge a failed inference or a removed interval.
        adjacent = valid[1:] & valid[:-1] & (dt > 0) & (dt <= .35)
        delta = (np.diff(yaw) + 180) % 360 - 180
        steps = np.abs(delta[adjacent])
        rates = steps / dt[adjacent]
        positions = np.where(adjacent & (np.abs(delta) > 5))[0]
        for i in positions:
            jumps.append(dict(recording=suffix, frame_before=int(rows[i]["frame_i"]),
                              frame_after=int(rows[i+1]["frame_i"]),
                              clip_before=int(rows[i]["video_frame_i"]),
                              clip_after=int(rows[i+1]["video_frame_i"]),
                              dt_sec=float(dt[i]), signed_jump_deg=float(delta[i]),
                              abs_jump_deg=float(abs(delta[i])),
                              apparent_rate_deg_s=float(abs(delta[i]) / dt[i]),
                              yaw_before_deg=float(yaw[i]), yaw_after_deg=float(yaw[i+1]),
                              rms_before_px=number(rows[i]["pose_rms_px"]),
                              rms_after_px=number(rows[i+1]["pose_rms_px"])))
        local = [s for s in segments if s["recording"] == prefix]
        noise = [number(s["baseline_noise_deg"]) for s in local]
        reasons = Counter(reason for s in local for reason in s["exclude_reasons"].split("; ") if reason)
        ranking.append(dict(recording=suffix, frames=len(rows), valid_poses=int(valid.sum()),
                            valid_percent=round(100 * valid.mean(), 2),
                            adjacent_pairs=int(adjacent.sum()), jumps_over_5deg=int((steps > 5).sum()),
                            jumps_over_30deg=int((steps > 30).sum()),
                            jumps_near_90deg=int(((steps >= 75) & (steps <= 105)).sum()),
                            step_p95_deg=quantile(steps, 95), max_step_deg=quantile(steps, 100),
                            rate_p95_deg_s=quantile(rates, 95), baseline_noise_p95_deg=quantile(noise, 95),
                            pose_rms_p95_px=quantile([number(r["pose_rms_px"]) for r in rows if r["pose_ok"] == "1"], 95),
                            segments=len(local), drive_usable=sum(s["drive_usable"] == "True" for s in local),
                            inertia_usable=sum(s["inertia_usable"] == "True" for s in local),
                            drive_exclusions=json.dumps(dict(reasons), ensure_ascii=False)))
        origin = float(t[0])
        fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        # Break plotting lines at each removed interval without discarding endpoint samples.
        breaks = np.r_[0, np.where(dt > .35)[0] + 1, len(rows)]
        for start, end in zip(breaks[:-1], breaks[1:]):
            axes[0].plot(t[start:end] - origin, yaw[start:end], ".-", ms=2.5, lw=.7, color="#2674a3")
        for window in selection[prefix]["command_windows"]:
            axes[0].axvspan(window["command_host_s"] - origin, window["stop_host_s"] - origin,
                           alpha=.14, color="#32a852")
            axes[0].axvline(window["stop_host_s"] + 2 - origin, color="#e89431", alpha=.4, lw=.8)
        axes[0].scatter(t[~valid]-origin, np.full((~valid).sum(), -85), color="red", marker="x", s=12, label="invalid pose")
        axes[0].set_ylabel("yaw / heading (deg)")
        axes[0].set_title(f"{suffix}: C4 raw PnP yaw | green=CAN rotation, orange=STOP+2s")
        axes[0].grid(alpha=.2)
        axes[1].scatter(t[1:][adjacent]-origin, np.abs(delta[adjacent]), s=8)
        axes[1].axhline(5, color="red", lw=.8, linestyle="--")
        axes[1].set_ylabel("abs step (deg)")
        axes[1].set_xlabel("Original host-monotonic seconds since first retained frame (gaps preserved)")
        axes[1].grid(alpha=.2)
        fig.tight_layout()
        name = f"{suffix}_yaw.png"
        fig.savefig(output / name, dpi=130)
        plt.close(fig)
        plots.append(name)
    ranking.sort(key=lambda r: (-r["jumps_over_30deg"], -(r["step_p95_deg"] or 0)))
    jumps.sort(key=lambda r: -r["abs_jump_deg"])
    write_csv(output / "recording_quality.csv", ranking)
    write_csv(output / "jump_events.csv", jumps)
    val = fit["validation"]
    write_csv(output / "in_sample_predictions.csv", val["rows"])
    write_csv(output / "leave_one_recording_out.csv", val["loo_rows"])
    validation_rows = val["rows"]
    r2 = None
    if validation_rows:
        observed = np.asarray([r["observed_deg"] for r in validation_rows])
        predicted = np.asarray([r["predicted_deg"] for r in validation_rows])
        ss = float(np.sum((observed - observed.mean())**2))
        r2 = float(1 - np.sum((predicted-observed)**2)/ss) if ss else None
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(observed, predicted)
        limits = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
        axes[0].plot(limits, limits, "k--")
        axes[0].set(xlabel="Observed rotation (PnP deg)", ylabel="Fitted rotation (deg)",
                    title=f"In-sample prediction: R2={r2:.3f}" if r2 is not None else "In-sample prediction")
        loo = val["leave_one_recording_out_folds"]["rmse_deg_by_recording"]
        names = list(loo)
        axes[1].barh([n.removeprefix("forklift_v4_recording_") for n in names], list(loo.values()))
        axes[1].axvline(3, color="red", linestyle="--")
        axes[1].set(xlabel="Held-out recording RMSE (deg)", title="Cross-validation: 3 deg quality gate")
        fig.tight_layout()
        fig.savefig(output / "fit_validation.png", dpi=140)
        plt.close(fig)
    summary = dict(model_hash=batch["model_hash"], recording_count=len(ranking),
                   frames=sum(r["frames"] for r in ranking), valid_poses=sum(r["valid_poses"] for r in ranking),
                   fit_status=fit["status"], safe_for_control=fit["safe_for_control"],
                   blockers=fit["deployment_blockers"], counts=fit["counts"],
                   in_sample_r2=r2, validation=val, identifiability=fit["identifiability"],
                   ranking=ranking, jump_event_count=len(jumps),
                   method="Wrapped 360-degree yaw step >5deg between adjacent valid output rows with 0<host dt<=0.35s; screening, not ground truth.",
                   limitations=["Removed intervals and invalid poses are never bridged for jump statistics.",
                                "Screening thresholds are diagnostic, not tuned to improve fit.",
                                "Fit error is against the same model's PnP observations, not independent physical yaw ground truth.",
                                "Existing training-data overlap can limit claims about unseen scenes."])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    text = ["# C4 회전 로그 및 적합 검증", "", f"모델 SHA256: `{batch['model_hash']}`", "",
            f"23개 분리 영상을 직접 추론했습니다. 유효 자세: {summary['valid_poses']}/{summary['frames']}.",
            f"적합 상태: **{fit['status']}**. 제어값 배포는 수행하지 않았습니다.", "",
            "튐 기준은 유효한 연속 출력 프레임, 실제 시간 간격 0.35초 이하에서 360도 주기 보정 후 5도 초과 변화입니다. 잘라낸 구간이나 추론 실패를 가로질러 비교하지 않습니다.",
            "각도 정답이 없으므로 튐은 이상 후보이고, 적합 오차도 C4 추론 각도에 대한 일치도입니다.", "",
            "## 적합 결과", "", f"- 사용 가능한 구간: drive={fit['counts']['segments_drive_usable']}, inertia={fit['counts']['segments_inertia_usable']}",
            f"- 훈련 데이터 내 오차: `{json.dumps(val['in_sample'])}`",
            f"- 녹화 단위 교차검증: `{json.dumps(val['leave_one_recording_out_folds'])}`",
            f"- 배포 차단 사유: `{json.dumps(fit['deployment_blockers'],ensure_ascii=False)}`", "",
            "[전체 적합 보고서](../fit/fit_report.json) · [구간별 제외 사유](../fit/rotation_segments.csv)", "",
            "## 영상별 진단", "", "| 녹화 | 유효/전체 | >5도 | >30도 | 최대 변화(도) | 로그 그래프 |",
            "|---|---:|---:|---:|---:|---|"]
    for row in ranking:
        maximum = row["max_step_deg"]
        text.append(f"| {row['recording']} | {row['valid_poses']}/{row['frames']} | {row['jumps_over_5deg']} | {row['jumps_over_30deg']} | {maximum:.2f} | [보기]({row['recording']}_yaw.png) |" if maximum is not None else
                    f"| {row['recording']} | {row['valid_poses']}/{row['frames']} | 0 | 0 | — | [보기]({row['recording']}_yaw.png) |")
    text += ["", "[프레임별 튐 목록](jump_events.csv) · [영상별 수치](recording_quality.csv)", ""]
    (output / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("frames", "valid_poses", "fit_status", "blockers", "in_sample_r2", "jump_event_count")},ensure_ascii=False))
    print("TOP", json.dumps(ranking[:6],ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.recordings_dir, args.run_dir)
