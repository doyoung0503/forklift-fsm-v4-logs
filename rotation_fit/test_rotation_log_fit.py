"""End-to-end tests for the native-log rotation fitting pipeline."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

try:  # Supports namespace-package and direct unittest discovery styles.
    from .fit_piecewise_rotation_model import (
        build_parser as build_fit_parser,
        main as fit_cli_main,
    )
    from .rotation_log_fit import (
        FitSettings,
        FIT_ANGLE_COLUMN,
        FIT_ANGLE_DOMAIN,
        discover_and_analyse,
        fit_rotation_response,
        load_command_windows,
        load_frame_series,
        validate_fit_angle_contract,
    )
except ImportError:  # pragma: no cover - depends on discovery invocation
    from fit_piecewise_rotation_model import (
        build_parser as build_fit_parser,
        main as fit_cli_main,
    )
    from rotation_log_fit import (
        FitSettings,
        FIT_ANGLE_COLUMN,
        FIT_ANGLE_DOMAIN,
        discover_and_analyse,
        fit_rotation_response,
        load_command_windows,
        load_frame_series,
        validate_fit_angle_contract,
    )


TRUE_DEAD_SEC = 0.45
TRUE_ACCEL_DEG_S2 = 10.0
TRUE_ACCEL_SEC = 0.65
TRUE_MAX_RATE_DEG_S = TRUE_ACCEL_DEG_S2 * TRUE_ACCEL_SEC
TRUE_INERTIA_SLOPE_SEC = 0.25
FRAME_DT_SEC = 0.05
SENSOR_EPOCH_MS = 1_800_000_000_000.0


def _driven_state(elapsed_sec: float) -> tuple[float, float]:
    """Return ideal command-held angle and rate after visible onset."""
    q = max(0.0, float(elapsed_sec))
    ramp = min(q, TRUE_ACCEL_SEC)
    angle = 0.5 * TRUE_ACCEL_DEG_S2 * ramp * ramp
    if q > TRUE_ACCEL_SEC:
        angle += TRUE_MAX_RATE_DEG_S * (q - TRUE_ACCEL_SEC)
    return angle, min(TRUE_MAX_RATE_DEG_S, TRUE_ACCEL_DEG_S2 * q)


def _can_record(
    *,
    movement: str,
    step: int,
    write_start_s: float,
    write_return_s: float,
    payload: list[int],
    source: str = "command_burst",
    burst_index: int | None = 1,
) -> dict[str, object]:
    return {
        "phase": "can_tx",
        "event": "movement_write",
        "source": source,
        "movement": movement,
        "step": step,
        "burst_index": burst_index,
        "burst_count": 5 if movement.startswith("rotate_") else 1,
        "payload": payload,
        "write_start_host_mono_ms": 1000.0 * write_start_s,
        "write_return_host_mono_ms": 1000.0 * write_return_s,
        "t_mono_ms": 1000.0 * write_return_s,
    }


def _write_recording(
    directory: Path,
    name: str,
    *,
    command_return_s: float,
    hold_sec: float,
    direction: str,
    strength: int = 30,
    spike_after_command_s: float | None = None,
    post_record_sec: float = 2.25,
    next_active_after_stop_s: float | None = None,
    coast_duration_sec: float = 0.40,
    dead_sec: float = TRUE_DEAD_SEC,
    observed_sign: float | None = None,
) -> dict[str, float]:
    """Write one deterministic native control/timing pair."""
    if direction not in {"LEFT", "RIGHT"}:
        raise ValueError(direction)
    prefix = f"synthetic_{name}"
    control_path = directory / f"{prefix}_control_seq.jsonl"
    timing_path = directory / f"{prefix}_inference_timing.csv"
    meta_path = directory / f"{prefix}_meta.json"
    meta_path.write_text(
        json.dumps({
            "model": "synthetic_pose.pt",
            "model_sha256": "a" * 64,
        }),
        encoding="utf-8",
    )
    step = int(round(command_return_s))
    command_start_s = command_return_s - 0.013
    stop_return_s = command_return_s + hold_sec
    stop_start_s = stop_return_s - 0.009
    movement = (
        "rotate_left_slow" if direction == "LEFT" else "rotate_right_slow"
    )
    analog = 127 + strength if direction == "LEFT" else 127 - strength
    rotate_payload = [127, analog, 127, 127, 127, 127, 127, 127]
    stop_payload = [127] * 8

    first = _can_record(
        movement=movement,
        step=step,
        write_start_s=command_start_s,
        write_return_s=command_return_s,
        payload=rotate_payload,
    )
    # Deliberately misleading records prove that pairing is based on the first
    # command burst, not JSON order, periodic traffic, or later burst members.
    second_burst = _can_record(
        movement=movement,
        step=step,
        write_start_s=command_return_s + 0.015,
        write_return_s=command_return_s + 0.021,
        payload=rotate_payload,
        burst_index=2,
    )
    periodic = _can_record(
        movement=movement,
        step=step,
        write_start_s=command_return_s + 0.200,
        write_return_s=command_return_s + 0.205,
        payload=rotate_payload,
        source="periodic",
        burst_index=None,
    )
    stop = _can_record(
        movement="stop",
        step=step,
        write_start_s=stop_start_s,
        write_return_s=stop_return_s,
        payload=stop_payload,
    )
    records: list[dict[str, object]] = [
        {
            "phase": "cmd",
            "cmd": "ROT_LEFT" if direction == "LEFT" else "ROT_RIGHT",
            # Intentionally offset: the fitter must use actual CAN timing.
            "t_mono": command_return_s - 0.30,
            "step": step,
        },
        second_burst,
        stop,
        periodic,
        first,
        {
            "phase": "can_tx",
            "event": "command_sync_done",
            "source": "command_burst",
            "movement": movement,
            "step": step,
            "sync_done_host_mono_ms": 1000.0 * (command_return_s + 0.030),
        },
        {
            "phase": "can_tx",
            "event": "command_sync_done",
            "source": "command_burst",
            "movement": "stop",
            "step": step,
            "sync_done_host_mono_ms": 1000.0 * (stop_return_s + 0.012),
        },
    ]
    if next_active_after_stop_s is not None:
        active_return = stop_return_s + next_active_after_stop_s
        records.append(_can_record(
            movement="forward",
            step=step + 1,
            write_start_s=active_return - 0.004,
            write_return_s=active_return,
            payload=[127, 127, 67, 127, 127, 127, 127, 127],
        ))
    with control_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    true_drive_sec = max(0.0, hold_sec - dead_sec)
    stop_angle, stop_rate = _driven_state(true_drive_sec)
    coast_angle = TRUE_INERTIA_SLOPE_SEC * stop_rate
    sign = (
        float(observed_sign) if observed_sign is not None
        else -1.0 if direction == "LEFT" else 1.0
    )
    first_frame_s = command_return_s - 0.95
    last_frame_s = stop_return_s + post_record_sec
    count = int(math.floor((last_frame_s - first_frame_s) / FRAME_DT_SEC)) + 1
    frame_times = first_frame_s + FRAME_DT_SEC * np.arange(count)
    fields = [
        "frame_i",
        "camera_input_host_mono_ms",
        "camera_sensor_timestamp_ms",
        "camera_timestamp_domain",
        "inference_ran",
        "model_det_ok",
        "pnp_ok",
        "yaw_deg",
    ]
    with timing_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, timestamp_s in enumerate(frame_times, 1):
            if timestamp_s <= stop_return_s:
                angle, _rate = _driven_state(
                    timestamp_s - command_return_s - dead_sec
                )
            else:
                coast_fraction = min(
                    1.0,
                    (timestamp_s - stop_return_s) / max(coast_duration_sec, 1e-9),
                )
                angle = stop_angle + coast_angle * coast_fraction
            if (
                spike_after_command_s is not None
                and abs(
                    timestamp_s
                    - (command_return_s + spike_after_command_s)
                ) < 0.25 * FRAME_DT_SEC
            ):
                angle += 0.80
            writer.writerow({
                "frame_i": step * 1000 + index,
                "camera_input_host_mono_ms": f"{1000.0 * timestamp_s:.6f}",
                "camera_sensor_timestamp_ms": (
                    f"{SENSOR_EPOCH_MS + 1000.0 * timestamp_s:.6f}"
                ),
                "camera_timestamp_domain": "timestamp_domain.system_time",
                "inference_ran": 1,
                "model_det_ok": 1,
                "pnp_ok": 1,
                "yaw_deg": f"{sign * angle:.9f}",
            })
    return {
        "command_return_s": command_return_s,
        "command_start_s": command_start_s,
        "stop_return_s": stop_return_s,
        "stop_start_s": stop_start_s,
        "stop_rate": stop_rate,
        "coast_angle": coast_angle,
    }


def _fit_settings() -> FitSettings:
    return FitSettings(
        onset_sustain_sec=0.20,
        onset_confirm_frames=4,
        onset_min_delta_deg=0.005,
        onset_min_rate_deg_s=0.05,
        smoothing_window_sec=0.12,
        stop_rate_window_sec=0.55,
        inertia_window_sec=0.12,
        max_frame_gap_sec=0.11,
        min_cruise_observation_sec=0.20,
        min_cruise_segments=2,
        min_inertia_speed_span_deg_s=0.50,
        can_time_point="return",
    )


class NativeLogFitTests(unittest.TestCase):
    def test_yaw_heading_contract_is_strict_at_all_fit_entry_points(self):
        self.assertEqual(FIT_ANGLE_COLUMN, "yaw_deg")
        self.assertEqual(FIT_ANGLE_DOMAIN, "heading")
        parsed = build_fit_parser().parse_args([])
        self.assertEqual(parsed.angle_column, FIT_ANGLE_COLUMN)
        self.assertEqual(parsed.angle_domain, FIT_ANGLE_DOMAIN)
        validate_fit_angle_contract("yaw_deg", "heading")

        invalid_pairs = (
            ("center_bearing_deg", "bearing"),
            ("yaw_deg", "bearing"),
            ("yaw_deg", "bearng"),
            ("heading_deg", "heading"),
        )
        for angle_column, angle_domain in invalid_pairs:
            with self.subTest(
                angle_column=angle_column, angle_domain=angle_domain,
            ):
                with self.assertRaisesRegex(ValueError, "fixed to"):
                    validate_fit_angle_contract(angle_column, angle_domain)
                with self.assertRaisesRegex(ValueError, "fixed to"):
                    fit_rotation_response(
                        [], angle_column=angle_column,
                        angle_domain=angle_domain,
                    )

        missing = Path("unused-because-angle-contract-is-checked-first.csv")
        with self.assertRaisesRegex(ValueError, "fixed to"):
            load_frame_series(
                missing, missing, angle_column="center_bearing_deg",
            )
        with self.assertRaisesRegex(ValueError, "fixed to"):
            discover_and_analyse(
                missing, angle_column="heading_deg",
            )
        with self.assertRaisesRegex(ValueError, "source_report angle_column"):
            fit_rotation_response(
                [], source_report={"angle_column": "center_bearing_deg"},
            )
        with self.assertRaisesRegex(ValueError, "source_report angle_domain"):
            fit_rotation_response(
                [], source_report={
                    "angle_column": "yaw_deg",
                    "angle_domain": "bearing",
                },
            )
        with self.assertRaisesRegex(ValueError, "source_report angle_column"):
            fit_rotation_response([], source_report={})

    def test_cli_rejects_noncanonical_angle_after_stale_cleanup(self):
        stale_names = (
            "rotation_model.json",
            "command_table.csv",
            "rotation_segments.csv",
            "fit_report.json",
        )
        invalid_pairs = (
            ("center_bearing_deg", "bearing"),
            ("yaw_deg", "bearing"),
            ("yaw_deg", "bearng"),
            ("heading_deg", "heading"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "out"
            output.mkdir()
            for angle_column, angle_domain in invalid_pairs:
                for name in stale_names:
                    (output / name).write_text("stale", encoding="utf-8")
                with self.subTest(
                    angle_column=angle_column, angle_domain=angle_domain,
                ):
                    with self.assertRaisesRegex(SystemExit, "fixed to"):
                        fit_cli_main([
                            "--recordings-dir", str(root),
                            "--output-dir", str(output),
                            "--angle-column", angle_column,
                            "--angle-domain", angle_domain,
                        ])
                    self.assertTrue(all(
                        not (output / name).exists() for name in stale_names
                    ))

    def test_malformed_control_jsonl_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken_control_seq.jsonl"
            path.write_text("{}\n{not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_command_windows(path, FitSettings())

    def test_direction_sign_outlier_is_rejected_before_pooling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(3):
                _write_recording(
                    root,
                    f"left_sign_{index}",
                    command_return_s=10.0 + 10.0 * index,
                    hold_sec=1.70,
                    direction="LEFT",
                    observed_sign=1.0 if index == 2 else -1.0,
                )
                _write_recording(
                    root,
                    f"right_sign_{index}",
                    command_return_s=50.0 + 10.0 * index,
                    hold_sec=1.70,
                    direction="RIGHT",
                )
            segments, source = discover_and_analyse(
                root, settings=FitSettings()
            )
            outlier = next(
                segment for segment in segments
                if segment.recording == "synthetic_left_sign_2"
            )
            self.assertFalse(outlier.drive_usable)
            self.assertTrue(any(
                "dominant LEFT sign" in reason
                for reason in outlier.exclude_reasons
            ))
            self.assertTrue(
                source["direction_sign_audit"]
                ["left_right_modal_signs_are_opposite"]
            )

    def test_immediate_motion_dead_time_is_never_negative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_recording(
                root,
                "immediate",
                command_return_s=40.0,
                hold_sec=1.70,
                direction="RIGHT",
                dead_sec=0.0,
            )
            segments, _source = discover_and_analyse(
                root, settings=FitSettings()
            )
            self.assertEqual(len(segments), 1)
            self.assertTrue(segments[0].drive_usable)
            self.assertGreaterEqual(segments[0].dead_time_estimated_s, 0.0)

    def test_explicit_recording_selection_is_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, name in enumerate(("selected", "excluded")):
                _write_recording(
                    root,
                    name,
                    command_return_s=40.0 + 10.0 * index,
                    hold_sec=1.70,
                    direction="RIGHT",
                )

            selected_name = "synthetic_selected"
            segments, source = discover_and_analyse(
                root, recording_names=(selected_name,), settings=FitSettings(),
            )
            self.assertEqual(len(segments), 1)
            self.assertEqual(source["recordings_available"], 2)
            self.assertEqual(source["recordings_selected"], [selected_name])
            self.assertEqual(
                source["recordings_excluded_by_selection"],
                ["synthetic_excluded"],
            )
            self.assertEqual(source["recordings_seen"], 1)

            with self.assertRaisesRegex(ValueError, "not found"):
                discover_and_analyse(
                    root, recording_names=("missing_recording",),
                )

    def test_default_onset_backtracks_from_confirmation_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, hold in enumerate(
                (0.90, 1.05, 1.20, 1.45, 1.70, 1.95, 2.20, 2.45)
            ):
                _write_recording(
                    root,
                    f"default_{index}",
                    command_return_s=50.0 + 10.0 * index,
                    hold_sec=hold,
                    direction="LEFT" if index % 2 == 0 else "RIGHT",
                )
            settings = FitSettings()
            segments, source = discover_and_analyse(root, settings=settings)
            fitted = fit_rotation_response(
                segments,
                settings=settings,
                source_report=source,
                bootstrap_runs=0,
            )
            self.assertEqual(fitted.report["status"], "READY")
            self.assertIsNotNone(fitted.model)
            self.assertAlmostEqual(
                fitted.model.dead_time_sec, TRUE_DEAD_SEC, delta=0.08
            )

    def test_end_to_end_recovers_phases_and_pools_observed_signs(self):
        settings = _fit_settings()
        holds = (0.90, 1.05, 1.20, 1.45, 1.70, 1.95, 2.20, 2.45)
        truth: dict[str, dict[str, float]] = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, hold in enumerate(holds):
                name = f"fit_{index}"
                truth[f"synthetic_{name}"] = _write_recording(
                    root,
                    name,
                    command_return_s=100.0 + 10.0 * index,
                    hold_sec=hold,
                    direction="LEFT" if index % 2 == 0 else "RIGHT",
                    spike_after_command_s=0.15 if index == 6 else None,
                )

            segments, source = discover_and_analyse(root, settings=settings)
            self.assertEqual(len(segments), len(holds))
            self.assertEqual(source["recordings_loaded"], len(holds))
            self.assertTrue(all(segment.drive_usable for segment in segments))
            self.assertTrue(all(segment.inertia_usable for segment in segments))

            for segment in segments:
                expected = truth[segment.recording]
                self.assertEqual(segment.command_timestamp_source, "can_tx")
                self.assertAlmostEqual(
                    segment.command_time_s, expected["command_return_s"], places=9
                )
                self.assertAlmostEqual(
                    segment.stop_time_s, expected["stop_return_s"], places=9
                )
                self.assertAlmostEqual(
                    segment.cmd_write_start_s, expected["command_start_s"], places=9
                )
                self.assertAlmostEqual(
                    segment.stop_write_start_s, expected["stop_start_s"], places=9
                )
                self.assertEqual(segment.command_strength, 30)
                expected_sign = -1.0 if segment.command_direction == "LEFT" else 1.0
                self.assertEqual(segment.observed_angle_sign, expected_sign)
                self.assertGreaterEqual(
                    float(np.min(segment.drive_angle_from_onset_deg)), -1e-8
                )

                # Both frame identifiers and both clocks remain available for
                # auditing the interval-censored visible-motion boundary.
                self.assertIsNotNone(segment.last_still_frame_i)
                self.assertIsNotNone(segment.first_motion_frame_i)
                self.assertIsNotNone(segment.last_still_sensor_timestamp_ms)
                self.assertIsNotNone(segment.motion_sensor_timestamp_ms)
                self.assertEqual(
                    segment.frame_time_source,
                    "camera_sensor_timestamp_ms_affine_to_host_mono",
                )
                self.assertIsNotNone(segment.motion_analysis_time_s)
                self.assertLess(
                    segment.last_still_host_mono_s, segment.motion_host_mono_s
                )
                self.assertAlmostEqual(
                    segment.motion_sensor_timestamp_ms,
                    SENSOR_EPOCH_MS + 1000.0 * segment.motion_host_mono_s,
                    places=3,
                )

                # The STOP derivative must use command-held observations only;
                # the synthetic post-STOP slope is deliberately different.
                self.assertLessEqual(
                    segment.stop_observation_host_mono_s,
                    segment.stop_time_s + 1e-12,
                )
                self.assertAlmostEqual(
                    segment.stop_rate_observed_deg_s,
                    expected["stop_rate"],
                    # A quadratic one-sided boundary fit straddling the
                    # acceleration-to-plateau knee has a small overshoot.
                    delta=0.85,
                )
                self.assertAlmostEqual(
                    segment.inertia_rotation_deg,
                    expected["coast_angle"],
                    delta=0.12,
                )
                self.assertAlmostEqual(
                    segment.inertia_host_mono_s,
                    segment.stop_time_s + settings.inertia_horizon_sec,
                    delta=FRAME_DT_SEC,
                )

            spiked = next(
                segment for segment in segments
                if segment.recording == "synthetic_fit_6"
            )
            self.assertGreaterEqual(
                spiked.motion_host_mono_s - spiked.command_time_s,
                TRUE_DEAD_SEC - FRAME_DT_SEC,
                "a single early angle spike must not be accepted as onset",
            )

            fitted = fit_rotation_response(
                segments,
                settings=settings,
                source_report=source,
                bootstrap_runs=0,
            )
            self.assertEqual(fitted.report["status"], "READY")
            self.assertTrue(fitted.report["safe_for_control"])
            self.assertEqual(fitted.report["angle_column"], "yaw_deg")
            self.assertEqual(fitted.report["angle_domain"], "heading")
            self.assertEqual(fitted.report["command_strength"], 30)
            self.assertIsNotNone(fitted.model)
            model = fitted.model
            self.assertAlmostEqual(model.dead_time_sec, TRUE_DEAD_SEC, delta=0.12)
            self.assertAlmostEqual(
                model.accel_deg_s2, TRUE_ACCEL_DEG_S2, delta=2.0
            )
            self.assertAlmostEqual(
                model.accel_duration_sec, TRUE_ACCEL_SEC, delta=0.16
            )
            self.assertAlmostEqual(
                model.max_rate_deg_s, TRUE_MAX_RATE_DEG_S, delta=0.80
            )
            self.assertEqual(model.inertia_intercept_deg, 0.0)
            self.assertAlmostEqual(
                model.inertia_slope_sec, TRUE_INERTIA_SLOPE_SEC, delta=0.035
            )
            self.assertEqual(
                fitted.report["bootstrap_95pct_cluster_by_recording"]["dead_time_sec"]["n"],
                0,
            )

            diagnostic_source = dict(source)
            diagnostic_source["mixed_model_ids_detected"] = True
            diagnostic = fit_rotation_response(
                segments,
                settings=settings,
                source_report=diagnostic_source,
                bootstrap_runs=0,
            )
            self.assertIsNotNone(diagnostic.model)
            self.assertEqual(diagnostic.report["status"], "DIAGNOSTIC_ONLY")
            self.assertFalse(diagnostic.report["safe_for_control"])
            self.assertTrue(any(
                "different inference model ids" in reason
                for reason in diagnostic.report["deployment_blockers"]
            ))

            unproven_source = dict(source)
            unproven_source["unidentified_model_recordings"] = [
                "synthetic_unknown"
            ]
            unproven = fit_rotation_response(
                segments,
                settings=settings,
                source_report=unproven_source,
                bootstrap_runs=0,
            )
            self.assertEqual(unproven.report["status"], "DIAGNOSTIC_ONLY")
            self.assertTrue(any(
                "lack model provenance" in reason
                for reason in unproven.report["deployment_blockers"]
            ))

            same_recording = [
                replace(segment, recording="one_shared_recording")
                for segment in segments
            ]
            clustered_loo = fit_rotation_response(
                same_recording,
                settings=settings,
                source_report=source,
                bootstrap_runs=0,
            )
            folds = clustered_loo.report["validation"][
                "leave_one_recording_out_folds"
            ]
            self.assertEqual(folds["attempted"], 1)
            self.assertEqual(folds["successful"], 0)
            self.assertFalse(clustered_loo.report["safe_for_control"])

    def test_strength_horizon_and_contamination_rejections(self):
        settings = _fit_settings()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_recording(
                root,
                "wrong_strength",
                command_return_s=300.0,
                hold_sec=1.70,
                direction="LEFT",
                strength=29,
            )
            _write_recording(
                root,
                "missing_horizon",
                command_return_s=320.0,
                hold_sec=1.70,
                direction="RIGHT",
                post_record_sec=1.0,
            )
            _write_recording(
                root,
                "contaminated",
                command_return_s=340.0,
                hold_sec=1.70,
                direction="LEFT",
                next_active_after_stop_s=1.95,
            )

            segments, _source = discover_and_analyse(root, settings=settings)
            by_name = {segment.recording: segment for segment in segments}

            wrong = by_name["synthetic_wrong_strength"]
            self.assertEqual(wrong.command_strength, 29)
            self.assertFalse(wrong.drive_usable)
            self.assertTrue(any(
                "command strength is 29" in reason
                for reason in wrong.exclude_reasons
            ))

            missing = by_name["synthetic_missing_horizon"]
            self.assertTrue(missing.drive_usable)
            self.assertFalse(missing.inertia_usable)
            self.assertTrue(any(
                "inertia horizon" in reason
                for reason in missing.inertia_exclude_reasons
            ))

            contaminated = by_name["synthetic_contaminated"]
            self.assertTrue(contaminated.drive_usable)
            self.assertFalse(contaminated.inertia_usable)
            self.assertTrue(any(
                "another active command contaminates" in reason
                for reason in contaminated.inertia_exclude_reasons
            ))

    def test_external_results_join_uses_new_validity_and_model_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timing = root / "join_inference_timing.csv"
            results = root / "join_new_results.csv"
            with timing.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "frame_i",
                    "camera_input_host_mono_ms",
                    "camera_sensor_timestamp_ms",
                    "camera_timestamp_domain",
                    "pnp_ok",
                ])
                writer.writeheader()
                for frame_i in range(1, 6):
                    writer.writerow({
                        "frame_i": frame_i,
                        "camera_input_host_mono_ms": 1000 + 50 * frame_i,
                        "camera_sensor_timestamp_ms": SENSOR_EPOCH_MS + 50 * frame_i,
                        "camera_timestamp_domain": "timestamp_domain.system_time",
                        # Old-model failure must not reject an external result.
                        "pnp_ok": 0,
                    })
            with results.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "frame_i", "yaw_deg", "valid", "confidence", "model_id"
                ])
                writer.writeheader()
                for frame_i in range(1, 6):
                    writer.writerow({
                        "frame_i": frame_i,
                        "yaw_deg": 10.0 + frame_i,
                        "valid": 0 if frame_i == 4 else 1,
                        "confidence": 0.2 if frame_i == 5 else 0.95,
                        "model_id": "weights-sha256:abc",
                    })

            series = load_frame_series(
                timing,
                results,
                angle_column="yaw_deg",
                valid_columns=("valid",),
                confidence_column="confidence",
                min_confidence=0.8,
                model_id_column="model_id",
            )
            np.testing.assert_array_equal(series.frame_i, [1, 2, 3])
            np.testing.assert_allclose(series.angle_deg, [11.0, 12.0, 13.0])
            self.assertEqual(series.model_ids, ("weights-sha256:abc",))

            rows = []
            with results.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["model_id"] = ""
            with results.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "empty model id"):
                load_frame_series(
                    timing,
                    results,
                    angle_column="yaw_deg",
                    valid_columns=("valid",),
                    confidence_column="confidence",
                    min_confidence=0.8,
                    model_id_column="model_id",
                )

    def test_without_observed_plateau_is_not_identifiable(self):
        settings = _fit_settings()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, hold in enumerate((0.80, 0.85, 0.90, 0.95)):
                _write_recording(
                    root,
                    f"ramp_only_{index}",
                    command_return_s=400.0 + 10.0 * index,
                    hold_sec=hold,
                    direction="LEFT" if index % 2 == 0 else "RIGHT",
                )
            segments, source = discover_and_analyse(root, settings=settings)
            self.assertTrue(all(segment.drive_usable for segment in segments))
            result = fit_rotation_response(
                segments,
                settings=settings,
                source_report=source,
                bootstrap_runs=0,
            )
            self.assertEqual(result.report["status"], "NOT_IDENTIFIABLE")
            self.assertFalse(result.report["safe_for_control"])
            self.assertIsNone(result.model)
            self.assertFalse(
                result.report["identifiability"]["acceleration_and_cruise"]
            )
            self.assertTrue(any(
                "plateau" in note or "upper bound" in note
                for note in result.report["identifiability"]["notes"]
            ))

    def test_command_window_uses_first_burst_return_boundary(self):
        settings = _fit_settings()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = _write_recording(
                root,
                "boundary",
                command_return_s=500.0,
                hold_sec=1.75,
                direction="RIGHT",
            )
            path = root / "synthetic_boundary_control_seq.jsonl"
            windows = load_command_windows(path, settings)
            self.assertEqual(len(windows), 1)
            window = windows[0]
            self.assertEqual(window.strength, 30)
            self.assertEqual(window.direction, "RIGHT")
            self.assertAlmostEqual(
                window.start.nominal_time("return"),
                expected["command_return_s"],
                places=9,
            )
            self.assertAlmostEqual(
                window.stop.nominal_time("return"),
                expected["stop_return_s"],
                places=9,
            )
            self.assertAlmostEqual(
                window.start.sync_done_s,
                expected["command_return_s"] + 0.030,
                places=9,
            )


if __name__ == "__main__":
    unittest.main()
