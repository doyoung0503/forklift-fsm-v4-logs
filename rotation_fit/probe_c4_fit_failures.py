"""Additional frame-level and fit-sensitivity diagnostics; never deploys a model."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rotation_fit.batch_infer_rotation_logs import YoloPosePredictor, load_camera_metadata, load_live_pose_solver
from rotation_fit.rotation_log_fit import FitSettings, discover_and_analyse, fit_rotation_response, _driven_angle


def main(model, recordings, run):
    audit = run / "audit"
    report = json.loads((run / "fit/fit_report.json").read_text(encoding="utf-8"))
    summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    settings = FitSettings()
    segments, source = discover_and_analyse(recordings.resolve(), results_dir=(run/"inference").resolve(),
                                            results_suffix="_new_pose.csv", angle_column="yaw_deg",
                                            valid_columns=["pose_ok"], confidence_column="confidence",
                                            min_confidence=.3, model_id_column="model_hash", settings=settings)
    d = report["dead_time"]["median"]
    phase, coast = report["phase_fit"], report["inertia_fit"]
    diagnostic = dict(status="DIAGNOSTIC_ONLY", safe_for_control=False,
                      purpose="Numerical candidate for review only; failed full-dataset validation",
                      model_hash=summary["model_hash"],
                      parameters=dict(dead_time_sec=d, accel_deg_s2=phase["accel_deg_s2"],
                                      accel_duration_sec=phase["accel_duration_sec"],
                                      max_rate_deg_s=phase["max_rate_deg_s"],
                                      inertia_slope_sec=coast["slope_sec"], inertia_intercept_deg=coast["intercept_deg"]),
                      formula=["q=max(0,T-dead_time_sec)", "r=min(q,accel_duration_sec)",
                               "drive_angle=0.5*accel_deg_s2*r*r+max_rate_deg_s*max(0,q-accel_duration_sec)",
                               "stop_rate=min(max_rate_deg_s,accel_deg_s2*q)",
                               "total_angle=drive_angle+inertia_slope_sec*stop_rate (zero intercept fit)"],
                      blockers=report["deployment_blockers"],
                      acceleration_lower_bound_warning="Fitted acceleration duration equals optimizer lower bound 0.15s; acceleration/cruise separation is not strongly supported even though the current identifiability flag is true.")
    (audit / "diagnostic_function.json").write_text(json.dumps(diagnostic,indent=2)+"\n",encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for s in segments:
        if s.drive_usable:
            axes[0].plot(s.drive_time_from_onset_s, s.drive_angle_from_onset_deg, ".-", lw=.6, ms=2, alpha=.45)
    q = np.linspace(0, max(max(s.drive_time_from_onset_s) for s in segments if s.drive_usable), 200)
    axes[0].plot(q, _driven_angle(q, phase["accel_deg_s2"], phase["accel_duration_sec"]), "k--", lw=2, label="pooled fit")
    axes[0].set(xlabel="Time after estimated onset (s)", ylabel="Rotation during command (deg)", title="Individual drive traces vs pooled fit")
    axes[0].legend()
    usable = [s for s in segments if s.inertia_usable]
    speeds = np.array([s.stop_rate_observed_deg_s for s in usable])
    inertia = np.array([s.inertia_rotation_deg for s in usable])
    axes[1].scatter(speeds, inertia)
    x = np.linspace(0, max(speeds)*1.1, 100)
    axes[1].plot(x, coast["intercept_deg"] + coast["slope_sec"]*x, "k--")
    axes[1].set(xlabel="Observed pre-STOP speed (deg/s)", ylabel="Observed coast at STOP+2s (deg)", title="Linear inertia fit")
    axes[2].scatter([s.command_duration_s for s in usable], speeds, label="observed pre-STOP speed")
    hold = np.linspace(0, max(s.command_duration_s for s in usable)*1.1,100)
    axes[2].plot(hold, np.minimum(phase["max_rate_deg_s"], phase["accel_deg_s2"]*np.maximum(0,hold-d)), "k--", label="model end speed")
    axes[2].set(xlabel="CAN command duration (s)", ylabel="Pre-STOP speed (deg/s)", title="Speed model consistency")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(audit/"phase_and_inertia.png",dpi=140)
    plt.close(fig)
    # Sensitivity only: disclose exactly which recordings were removed AFTER inspection.
    omitted = {"forklift_v4_recording_"+r["recording"] for r in summary["ranking"]
               if r["jumps_over_5deg"] > 0 or r["valid_poses"] < 3}
    retained = [s for s in segments if s.recording not in omitted]
    subset_source = dict(source)
    subset_source["recording_errors"] = {k:v for k,v in source["recording_errors"].items() if k not in omitted}
    subset_source["diagnostic_posthoc_omitted_recordings"] = sorted(omitted)
    subset_result = fit_rotation_response(retained, settings=settings, source_report=subset_source, bootstrap_runs=0)
    sensitivity = dict(purpose="Post-hoc sensitivity, NOT an unbiased held-out evaluation or deployment candidate",
                       omitted_recordings=sorted(omitted), result=subset_result.report)
    (audit/"sensitivity_without_jump_videos.json").write_text(json.dumps(sensitivity,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SENSITIVITY",json.dumps(dict(omitted=sorted(omitted),counts=subset_result.report["counts"],
                                       validation=subset_result.report["validation"]["leave_one_recording_out_folds"])),flush=True)

    torch.set_num_threads(4)
    predictor = YoloPosePredictor(model,device="cuda:0",confidence_threshold=.3,front_class_name="item",use_half=True)
    load_live_pose_solver()  # Establish the exact live geometry import path.
    from calib.geometry import pose_from_visible_kpts_pnp
    with (audit/"jump_events.csv").open(encoding="utf-8-sig",newline="") as handle:
        jumps=list(csv.DictReader(handle))
    chosen={}
    for row in jumps:
        chosen.setdefault(row["recording"],row)
        if len(chosen)>=4:
            break
    details=[]
    for name, event in chosen.items():
        prefix="forklift_v4_recording_"+name
        intrinsics, geometry=load_camera_metadata(recordings/f"{prefix}_meta.json")
        start=max(0,int(event["clip_before"])-1)
        stop=int(event["clip_after"])+1
        capture=cv2.VideoCapture(str(recordings/f"{prefix}_raw.avi"))
        records=[]
        images=[]
        try:
            for index in range(stop+1):
                ok, frame=capture.read()
                if not ok:
                    break
                if index<start:
                    continue
                detection=predictor.predict(frame)
                record=dict(recording=name,clip_index=index,detected=detection.keypoints is not None)
                canvas=frame.copy()
                if detection.keypoints is not None:
                    results=pose_from_visible_kpts_pnp(detection.keypoints,intrinsics,
                        face_w=geometry.width_m,face_h=geometry.height_m,
                        body_d=geometry.length_m,vis_thr=geometry.visibility_threshold)
                    good,yaw,_,_,_,_,_,info=results
                    record.update(pose_ok=bool(good),yaw_deg=float(yaw) if good else None,
                                  selected_face=info.get("selected_front_face"),n_used=info.get("n_used"),
                                  rms_px=info.get("rms"),keypoints=detection.keypoints.tolist())
                    for j,(px,py,vis) in enumerate(detection.keypoints):
                        cv2.circle(canvas,(round(float(px)),round(float(py))),3,(0,220,255),-1)
                        cv2.putText(canvas,str(j),(round(float(px))+4,round(float(py))-3),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,220,255),1)
                records.append(record)
                images.append(canvas)
        finally:
            capture.release()
        fig, axes=plt.subplots(1,len(images),figsize=(4*len(images),3.6),squeeze=False)
        for ax,img,record in zip(axes[0],images,records):
            ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            angle=record.get("yaw_deg")
            ax.set_title(f'clip {record["clip_index"]}: yaw={angle:.2f}\nface={record.get("selected_face")}, n={record.get("n_used")}' if angle is not None else f'clip {record["clip_index"]}: no pose',fontsize=10)
            ax.axis("off")
        fig.suptitle(f"C4 re-inference probe: {name} (PnP RANSAC may vary from full run)")
        fig.tight_layout()
        fig.savefig(audit/f"{name}_jump_probe.png",dpi=130)
        plt.close(fig)
        details.extend(records)
        print("PROBE",json.dumps([{k:v for k,v in r.items() if k!='keypoints'} for r in records]),flush=True)
    (audit/"jump_frame_probes.json").write_text(json.dumps(details,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--recordings-dir",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args()
    main(args.model,args.recordings_dir,args.run_dir)
