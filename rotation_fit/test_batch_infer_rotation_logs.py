"""Tests for transactional recorded-video pose inference."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

try:
    from .batch_infer_rotation_logs import (
        BatchInferenceError,
        Detection,
        PoseEstimate,
        RecordingEligibilityError,
        RecordingInput,
        YoloPosePredictor,
        build_frame_mapping,
        discover_recordings_with_audit,
        run_batch_inference,
        select_detection_index,
    )
except ImportError:  # pragma: no cover - direct unittest discovery
    from batch_infer_rotation_logs import (
        BatchInferenceError,
        Detection,
        PoseEstimate,
        RecordingEligibilityError,
        RecordingInput,
        YoloPosePredictor,
        build_frame_mapping,
        discover_recordings_with_audit,
        run_batch_inference,
        select_detection_index,
    )


class FakeCapture:
    def __init__(self, frame_count: int, width: int = 2, height: int = 2):
        self.frames = [
            np.full((height, width, 3), index, dtype=np.uint8)
            for index in range(frame_count)
        ]
        self.width = width
        self.height = height
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        return 0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class FakePredictor:
    device = "cpu"

    def __init__(self, *, fail_after: int | None = None):
        self.calls = 0
        self.fail_after = fail_after

    def predict(self, image):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("synthetic model failure")
        keypoints = np.ones((9, 3), dtype=np.float32)
        keypoints[:, 0] = float(image[0, 0, 0])
        return Detection(keypoints, 0.91)


def fake_pose_solver(keypoints, _intrinsics, _geometry):
    value = float(keypoints[0, 0])
    return PoseEstimate(
        yaw_deg=value + 0.25,
        pos_x_m=0.1,
        pos_z_m=2.0,
        source="test_pnp",
        n_used=9,
        rms_px=1.2,
    )


def write_recording(
    root: Path,
    prefix: str,
    *,
    timing_last: int,
    explicit_indices: dict[int, int] | None = None,
) -> RecordingInput:
    video = root / f"{prefix}_raw.mp4"
    timing = root / f"{prefix}_inference_timing.csv"
    meta = root / f"{prefix}_meta.json"
    control = root / f"{prefix}_control_seq.jsonl"
    video.write_bytes(b"fake video placeholder")
    meta.write_text(
        json.dumps(
            {
                "intrinsics": {
                    "fx": 100.0,
                    "fy": 100.0,
                    "ppx": 1.0,
                    "ppy": 1.0,
                    "width": 2,
                    "height": 2,
                    "coeffs": [0, 0, 0, 0, 0],
                },
                "pallet_size_m": {
                    "width": 1.1,
                    "height": 0.15,
                    "length": 1.1,
                },
                "pose_kpt_visibility_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    control.write_text("{}\n", encoding="utf-8")
    fieldnames = ["frame_i"]
    if explicit_indices is not None:
        fieldnames.append("raw_video_frame_index")
    inverse = (
        {frame_i: raw_index for raw_index, frame_i in explicit_indices.items()}
        if explicit_indices is not None
        else {}
    )
    with timing.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_i in range(1, timing_last + 1):
            row = {"frame_i": frame_i}
            if explicit_indices is not None:
                row["raw_video_frame_index"] = inverse.get(frame_i, "")
            writer.writerow(row)
    return RecordingInput(prefix, video, timing, meta, control)


class FrameMappingTests(unittest.TestCase):
    def test_legacy_mapping_uses_source_code_frame_36_not_end_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rec = write_recording(root, "normal", timing_last=38)
            mapping = build_frame_mapping(rec.timing_path, rec.meta_path, 3)
            self.assertEqual(mapping.frame_ids, (36, 37, 38))
            self.assertEqual(mapping.method, "legacy_recorder_first_frame_i_36")
            self.assertEqual(mapping.joined_ratio, 1.0)
            self.assertEqual(mapping.raw_minus_timing_tail_frames, 0)

    def test_one_unflushed_final_timing_row_is_audited_but_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rec = write_recording(root, "one_extra_raw", timing_last=37)
            mapping = build_frame_mapping(rec.timing_path, rec.meta_path, 3)
            self.assertEqual(mapping.frame_ids, (36, 37, 38))
            self.assertEqual(mapping.joined_timing_frames, 2)
            self.assertAlmostEqual(mapping.joined_ratio, 2 / 3)
            self.assertEqual(mapping.raw_minus_timing_tail_frames, 1)

    def test_recovered_video_with_internal_loss_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rec = write_recording(root, "damaged", timing_last=56)
            with self.assertRaisesRegex(
                RecordingEligibilityError, "alignment is ambiguous"
            ) as caught:
                build_frame_mapping(rec.timing_path, rec.meta_path, 3)
            self.assertEqual(caught.exception.audit["raw_last_frame_i"], 38)
            self.assertEqual(caught.exception.audit["timing_last_frame_i"], 56)

    def test_explicit_per_frame_mapping_is_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rec = write_recording(
                root,
                "explicit",
                timing_last=80,
                explicit_indices={0: 50, 1: 53, 2: 80},
            )
            mapping = build_frame_mapping(rec.timing_path, rec.meta_path, 3)
            self.assertEqual(mapping.frame_ids, (50, 53, 80))
            self.assertEqual(mapping.method, "timing_raw_video_frame_index")


class DetectionSelectionTests(unittest.TestCase):
    def test_aspect_ratio_precedes_confidence_like_live_perception(self):
        boxes = np.asarray(
            [[0, 0, 80, 10], [0, 0, 20, 10], [0, 0, 73, 10]], dtype=float
        )
        confidences = np.asarray([0.6, 0.99, 0.7])
        classes = np.asarray([1, 1, 1])
        selected = select_detection_index(
            boxes,
            confidences,
            classes,
            target_class=1,
            confidence_threshold=0.3,
            target_aspect_ratio=7.33,
        )
        self.assertEqual(selected, 2)

    def test_batch_uses_the_same_quantize_precision_option_as_live_inference(self):
        calls = []

        class Model:
            def predict(self, **kwargs):
                calls.append(kwargs)
                return [SimpleNamespace(boxes=None, keypoints=None)]

        predictor = YoloPosePredictor.__new__(YoloPosePredictor)
        predictor.device = "cuda:0"
        predictor.confidence_threshold = 0.3
        predictor.use_half = True
        predictor.model = Model()

        result = predictor.predict(np.zeros((2, 2, 3), dtype=np.uint8))

        self.assertIsNone(result.keypoints)
        self.assertEqual(calls[0]["quantize"], 16)
        self.assertNotIn("half", calls[0])

    def test_outdated_ultralytics_is_rejected_before_model_loading(self):
        with mock.patch(
            "importlib.metadata.version", return_value="8.4.59"
        ):
            with self.assertRaisesRegex(BatchInferenceError, ">=8.4.60"):
                YoloPosePredictor(
                    Path("unused.pt"),
                    device="cpu",
                    confidence_threshold=0.3,
                    front_class_name="item",
                    use_half=False,
                )


class BatchTransactionTests(unittest.TestCase):
    def _run(self, root, recordings, predictor, captures):
        model = root / "new_model.pt"
        model.write_bytes(b"new test weights")
        results = root / "results"

        def capture_factory(path):
            return FakeCapture(captures[Path(path).name])

        manifest = run_batch_inference(
            model_path=model,
            recordings_dir=root,
            results_dir=results,
            recordings=recordings,
            predictor=predictor,
            pose_solver=fake_pose_solver,
            capture_factory=capture_factory,
            progress=lambda _message: None,
        )
        return model, results, manifest

    def test_fit_ready_csv_and_manifest_are_published_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rec = write_recording(root, "good", timing_last=38)
            model, results, manifest = self._run(
                root, [rec], FakePredictor(), {rec.video_path.name: 3}
            )
            model_id = "sha256:" + hashlib.sha256(model.read_bytes()).hexdigest()
            self.assertEqual(manifest["model_hash"], model_id)
            self.assertEqual(manifest["successful_recording_prefixes"], ["good"])
            self.assertEqual(manifest["recording_count"], 1)
            self.assertEqual(manifest["recording_audit"][0]["eligible"], True)
            self.assertTrue((results / "batch_inference_manifest.json").is_file())

            with (results / "good_new_pose.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["frame_i"]) for row in rows], [36, 37, 38])
            self.assertTrue(all(row["model_hash"] == model_id for row in rows))
            self.assertTrue(all(row["pose_ok"] == "1" for row in rows))
            self.assertTrue(all(row["angle_domain"] == "heading" for row in rows))

    def test_model_failure_leaves_no_partially_published_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_recording(root, "first", timing_last=38)
            second = write_recording(root, "second", timing_last=38)
            model = root / "new_model.pt"
            model.write_bytes(b"new test weights")
            results = root / "results"
            results.mkdir()
            old_target = results / "first_new_pose.csv"
            old_target.write_text("old generation\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
                run_batch_inference(
                    model_path=model,
                    recordings_dir=root,
                    results_dir=results,
                    recordings=[first, second],
                    predictor=FakePredictor(fail_after=3),
                    pose_solver=fake_pose_solver,
                    capture_factory=lambda _path: FakeCapture(3),
                    progress=lambda _message: None,
                )
            self.assertEqual(
                old_target.read_text(encoding="utf-8"), "old generation\n"
            )
            self.assertFalse((results / "second_new_pose.csv").exists())
            self.assertFalse((results / "batch_inference_manifest.json").exists())
            self.assertEqual(list(results.glob(".batch-infer-*")), [])

    def test_low_pose_coverage_is_not_silently_excluded_from_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rec = write_recording(root, "hard_recording", timing_last=38)
            model = root / "new_model.pt"
            model.write_bytes(b"new test weights")
            results = root / "results"

            with self.assertRaisesRegex(
                BatchInferenceError, "selection-bias calibration"
            ):
                run_batch_inference(
                    model_path=model,
                    recordings_dir=root,
                    results_dir=results,
                    recordings=[rec],
                    predictor=FakePredictor(),
                    pose_solver=lambda *_args: None,
                    capture_factory=lambda _path: FakeCapture(3),
                    progress=lambda _message: None,
                )

            self.assertFalse((results / "hard_recording_new_pose.csv").exists())
            self.assertFalse((results / "batch_inference_manifest.json").exists())

    def test_bad_alignment_is_explicitly_excluded_while_good_data_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = write_recording(root, "bad", timing_last=56)
            good = write_recording(root, "good", timing_last=38)
            _, results, manifest = self._run(
                root,
                [bad, good],
                FakePredictor(),
                {bad.video_path.name: 3, good.video_path.name: 3},
            )
            self.assertEqual(manifest["successful_recording_prefixes"], ["good"])
            self.assertEqual(manifest["excluded_recording_count"], 1)
            exclusion = manifest["excluded_recordings"][0]
            self.assertEqual(exclusion["prefix"], "bad")
            self.assertFalse(exclusion["eligible"])
            self.assertIn("alignment is ambiguous", exclusion["exclusion_reason"])
            self.assertEqual(exclusion["frame_mapping_method"],
                             "legacy_recorder_first_frame_i_36")
            self.assertFalse((results / "bad_new_pose.csv").exists())
            self.assertTrue((results / "good_new_pose.csv").exists())

    def test_unmapped_video_can_be_inferred_only_in_explicit_evaluation_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = write_recording(root, "damaged", timing_last=56)
            model = root / "new_model.pt"
            model.write_bytes(b"new test weights")
            results = root / "results"

            manifest = run_batch_inference(
                model_path=model,
                recordings_dir=root,
                results_dir=results,
                recordings=[recording],
                predictor=FakePredictor(),
                pose_solver=fake_pose_solver,
                capture_factory=lambda _path: FakeCapture(3),
                progress=lambda _message: None,
                allow_low_pose=True,
                allow_unmapped_video=True,
            )

            report = manifest["recordings"][0]
            self.assertEqual(report["frame_mapping_method"],
                             "evaluation_video_index_only")
            self.assertEqual(report["joined_timing_frames"], 0)
            self.assertTrue(manifest["allow_unmapped_video"])
            with (results / "damaged_new_pose.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["frame_i"]) for row in rows], [0, 1, 2])

    def test_discovery_reports_control_log_without_raw_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = write_recording(root, "complete", timing_last=38)
            (root / "missing_control_seq.jsonl").write_text("{}\n", encoding="utf-8")
            found, excluded = discover_recordings_with_audit(root)
            self.assertEqual(found, [complete])
            self.assertEqual([item.prefix for item in excluded], ["missing"])
            self.assertIn("missing_raw.mp4", excluded[0].exclusion_reason)


if __name__ == "__main__":
    unittest.main()
