#!/usr/bin/env python3
"""CLI for the timestamp-aligned, linear-inertia rotation fit.

Example using the pose already present in each timing CSV::

    py -3 rotation_fit/fit_piecewise_rotation_model.py \
      --recordings-dir extracted/depth_cam/rec \
      --angle-column yaw_deg --angle-domain heading

Example after a new inference model writes one CSV per recording.  The model
must expose its orientation estimate under the canonical ``yaw_deg`` column::

    py -3 rotation_fit/fit_piecewise_rotation_model.py \
      --recordings-dir extracted/depth_cam/rec \
      --results-dir new_inference \
      --results-suffix _new_pose.csv \
      --angle-column yaw_deg --angle-domain heading \
      --valid-column pose_ok --model-id-column model_hash

The result CSV only needs ``frame_i``, the canonical ``yaw_deg`` column, and
any explicit validity/model-id columns.  Camera and command timing are joined
from the original recording by frame id.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

try:
    from .rotation_log_fit import (
        FitSettings,
        FIT_ANGLE_COLUMN,
        FIT_ANGLE_DOMAIN,
        discover_and_analyse,
        fit_rotation_response,
        validate_fit_angle_contract,
        write_segments_csv,
    )
except ImportError:
    from rotation_log_fit import (
        FitSettings,
        FIT_ANGLE_COLUMN,
        FIT_ANGLE_DOMAIN,
        discover_and_analyse,
        fit_rotation_response,
        validate_fit_angle_contract,
        write_segments_csv,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_RECORDINGS = HERE.parent / "extracted" / "depth_cam" / "rec"
DEFAULT_OUTPUT = HERE / "out" / "piecewise_linear_inertia"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit dead time -> constant acceleration -> cruise -> linear "
            "post-STOP inertia at fixed joystick deflection 30."
        )
    )
    parser.add_argument(
        "--recordings-dir", type=Path, default=DEFAULT_RECORDINGS,
        help="directory containing *_control_seq.jsonl and *_inference_timing.csv",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        help="directory containing new inference CSVs (default: recordings dir)",
    )
    parser.add_argument(
        "--results-suffix", default="_inference_timing.csv",
        help="suffix appended to each recording basename",
    )
    parser.add_argument(
        "--recording-name", action="append", default=None,
        help=(
            "explicit recording basename to include; may be repeated. "
            "The automatic pipeline supplies its validated recording manifest."
        ),
    )
    parser.add_argument(
        "--angle-column", default=FIT_ANGLE_COLUMN,
        help="fixed fit signal: yaw_deg (PnP orientation, not target bearing)",
    )
    parser.add_argument(
        "--angle-domain", default=FIT_ANGLE_DOMAIN,
        help="fixed fit domain: heading; other values are rejected after stale cleanup",
    )
    parser.add_argument("--angle-period", type=float, choices=(180.0, 360.0), default=360.0)
    parser.add_argument("--frame-id-column", default="frame_i")
    parser.add_argument(
        "--valid-column", action="append", default=[],
        help="boolean result column required to be true; may be repeated",
    )
    parser.add_argument("--confidence-column")
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument(
        "--model-id-column",
        help=(
            "required provenance column for external results; every valid row "
            "must have one id/hash and mixed ids are rejected"
        ),
    )
    parser.add_argument(
        "--allow-mixed-model-ids", action="store_true",
        help="diagnostic only: permit pooling different inference model versions",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-runs", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260904)
    parser.add_argument(
        "--min-loo-recordings", "--min-loo-predictions",
        dest="min_loo_predictions", type=int, default=4,
        help="minimum successful leave-one-recording-out folds for a runtime artifact",
    )
    parser.add_argument(
        "--max-loo-rmse-deg", type=float, default=3.0,
        help="maximum leave-one-recording-out RMSE allowed for a runtime artifact",
    )
    parser.add_argument(
        "--fit-inertia-intercept", action="store_true",
        help="fit non-negative b0+b1*w; default is physical through-origin b1*w",
    )
    parser.add_argument(
        "--allow-logical-command-time", action="store_true",
        help="permit old logs without actual can_tx write events",
    )
    parser.add_argument(
        "--can-time-point", choices=("start", "return", "midpoint"),
        default="return",
        help="nominal point inside each logged Kvaser write bracket",
    )
    return parser


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_command_table(path: Path, model) -> None:
    minimum = model.fitted_min_angle_deg
    maximum = model.fitted_max_angle_deg
    if maximum <= minimum:
        targets = [minimum]
    else:
        count = max(2, int((maximum - minimum) / 0.5) + 1)
        targets = [minimum + i * (maximum - minimum) / (count - 1) for i in range(count)]
    rows = []
    for target in targets:
        plan = model.command_duration(target)
        rows.append({
            "target_deg": target,
            "command_strength": plan.command_strength,
            "hold_sec": plan.hold_sec,
            "driven_sec": plan.driven_sec,
            "command_end_rate_deg_s": plan.command_end_rate_deg_s,
            "angle_before_stop_deg": plan.predicted_command_angle_deg,
            "inertia_angle_deg": plan.predicted_inertia_angle_deg,
            "predicted_total_deg": plan.predicted_total_angle_deg,
            "feasible": plan.feasible,
            "note": plan.note,
        })
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Runtime outputs describe only the current fit attempt.  Invalidate the
    # exact generated files up front, including when validation or input
    # discovery later fails, so an old calibration cannot look current.
    for stale_name in (
        "rotation_model.json", "command_table.csv",
        "rotation_segments.csv", "fit_report.json",
    ):
        (output / stale_name).unlink(missing_ok=True)

    if args.min_confidence is not None and not args.confidence_column:
        raise SystemExit("--min-confidence requires --confidence-column")
    if args.results_dir is not None and not args.model_id_column:
        raise SystemExit("--results-dir requires --model-id-column")
    if args.bootstrap_runs < 0:
        raise SystemExit("--bootstrap-runs must be non-negative")
    if args.min_loo_predictions < 1:
        raise SystemExit("--min-loo-recordings must be positive")
    if not math.isfinite(args.max_loo_rmse_deg) or args.max_loo_rmse_deg <= 0.0:
        raise SystemExit("--max-loo-rmse-deg must be positive")
    try:
        validate_fit_angle_contract(args.angle_column, args.angle_domain)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    settings = FitSettings(
        command_strength=30,
        angle_period_deg=args.angle_period,
        min_loo_predictions=args.min_loo_predictions,
        max_loo_rmse_deg=args.max_loo_rmse_deg,
        fit_inertia_intercept=args.fit_inertia_intercept,
        require_actual_can_timing=not args.allow_logical_command_time,
        can_time_point=args.can_time_point,
    )
    try:
        segments, source_report = discover_and_analyse(
            args.recordings_dir.resolve(),
            results_dir=(args.results_dir.resolve() if args.results_dir else None),
            results_suffix=args.results_suffix,
            recording_names=args.recording_name,
            angle_column=args.angle_column,
            angle_period_deg=args.angle_period,
            frame_id_column=args.frame_id_column,
            valid_columns=args.valid_column,
            confidence_column=args.confidence_column,
            min_confidence=args.min_confidence,
            model_id_column=args.model_id_column,
            allow_mixed_model_ids=args.allow_mixed_model_ids,
            settings=settings,
        )
    except (OSError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 1
    result = fit_rotation_response(
        segments,
        settings=settings,
        angle_column=args.angle_column,
        angle_domain=args.angle_domain,
        source_report=source_report,
        bootstrap_runs=args.bootstrap_runs,
        random_seed=args.random_seed,
    )

    write_segments_csv(output / "rotation_segments.csv", result.segments)
    _write_json(output / "fit_report.json", result.report)

    print(
        f"segments: total={len(result.segments)}, "
        f"drive={result.report['counts']['segments_drive_usable']}, "
        f"inertia={result.report['counts']['segments_inertia_usable']}"
    )
    print(f"fit status: {result.report['status']}")
    print(f"audit: {output / 'rotation_segments.csv'}")
    print(f"report: {output / 'fit_report.json'}")

    if result.model is None or not result.report["safe_for_control"]:
        notes = result.report["identifiability"]["notes"]
        for note in notes:
            print(f"not identifiable: {note}")
        for recording, error in (
            result.report.get("source", {}).get("recording_errors", {}).items()
        ):
            print(f"recording error: {recording}: {error}")
        for blocker in result.report["deployment_blockers"]:
            print(f"deployment blocked: {blocker}")
        print(
            "runtime model was not emitted; resolve the reported data/quality "
            "gates, then rerun"
        )
        return 2

    artifact = result.model.to_dict()
    artifact["fit_report"] = result.report
    _write_json(output / "rotation_model.json", artifact)
    _write_command_table(output / "command_table.csv", result.model)
    p = artifact["parameters"]
    print(
        "fitted: dead={dead_time_sec:.4f}s, accel={accel_deg_s2:.3f}deg/s^2, "
        "accel_time={accel_duration_sec:.4f}s, vmax={max_rate_deg_s:.3f}deg/s, "
        "coast=max(0,{inertia_intercept_deg:.3f}+{inertia_slope_sec:.4f}*w)".format(**p)
    )
    print(f"runtime artifact: {output / 'rotation_model.json'}")
    print(f"target table: {output / 'command_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
