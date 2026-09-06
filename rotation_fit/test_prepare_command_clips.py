"""Command-window coverage and lossless, timestamp-preserving export tests."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from rotation_fit.prepare_command_clips import select_frames, export_recording
from rotation_fit.rotation_log_fit import CanBoundary, CommandWindow
from rotation_fit.batch_infer_rotation_logs import (
    RecordingInput, build_frame_mapping, discover_recordings_with_audit,
    BatchInferenceError, infer_recording,
)
from rotation_fit.test_batch_infer_rotation_logs import (
    write_recording, FakePredictor, fake_pose_solver,
)


def window(start=10.0, stop=12.0, strength=30, source="can_tx"):
    return CommandWindow("sample", 1, "ROT_LEFT", "rotate_left_slow", strength,
                         CanBoundary("rotate_left_slow", 1, start, start, "", strength, source=source),
                         CanBoundary("stop", 1, stop, stop, "", 0, source=source), None)


def timing_rows():
    return [dict(frame_i=str(i), camera_input_host_mono_ms=str(i * 100),
                 camera_sensor_timestamp_ms=str(i * 100 + 100000),
                 camera_timestamp_domain="sensor", model_det_ok="0") for i in range(300)]


class ClipTests(unittest.TestCase):
    def test_baseline_entire_drive_and_coast_retained_without_model_flags(self):
        indices, _ = select_frames(range(300), timing_rows(), [window()])
        self.assertEqual(indices, list(range(85, 148)))

    def test_overlapping_windows_infer_each_frame_once(self):
        indices, _ = select_frames(range(300), timing_rows(), [window(), window(11, 13)])
        self.assertEqual(indices, list(range(85, 158)))

    def test_disjoint_windows_keep_original_gap_and_frame_ids(self):
        indices, _ = select_frames(range(300), timing_rows(), [window(), window(20, 22)])
        self.assertEqual(indices, list(range(85, 148)) + list(range(185, 248)))

    def test_logical_times_and_other_strengths_are_not_used(self):
        indices, _ = select_frames(range(300), timing_rows(),
                                   [window(strength=40), window(source="logical_cmd_fallback")])
        self.assertEqual(indices, [])

    def test_invalid_margin_or_clock_rejected(self):
        with self.assertRaises(ValueError):
            select_frames(range(300), timing_rows(), [window()], margin_sec=0)
        rows = timing_rows()
        rows[2]["camera_input_host_mono_ms"] = "nan"
        with self.assertRaises(ValueError):
            select_frames(range(300), rows, [window()])

    def test_lossless_export_and_real_batch_reader_preserve_frame_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.mkdir()
            target.mkdir()
            rec = write_recording(source, "sample", timing_last=45)
            video = source / "sample_input.avi"
            rng = np.random.default_rng(7)
            frames = rng.integers(0, 256, (10, 16, 16, 3), dtype=np.uint8)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10, (16, 16))
            self.assertTrue(writer.isOpened())
            for frame in frames:
                writer.write(frame)
            writer.release()
            meta = json.loads(rec.meta_path.read_text())
            meta["intrinsics"].update(width=16, height=16)
            rec.meta_path.write_text(json.dumps(meta))
            rec = RecordingInput(rec.prefix, video, rec.timing_path, rec.meta_path, rec.control_path)
            rows = [dict(frame_i=str(i), camera_input_host_mono_ms=str(i * 100),
                         camera_sensor_timestamp_ms=str(i * 100 + 100000)) for i in range(1, 46)]
            plan = dict(recording=rec, indices=[0, 1, 7, 9], frame_ids=tuple(range(36, 46)),
                        source_frames=10, width=16, height=16, fps=10, rows=rows,
                        fields=list(rows[0]), audit={})
            report = export_recording(plan, target)
            self.assertEqual(report["selected_frames"], 4)
            exported, exclusions = discover_recordings_with_audit(target)
            self.assertEqual(exclusions, [])
            self.assertEqual(len(exported), 1)
            mapping = build_frame_mapping(exported[0].timing_path, exported[0].meta_path, 4)
            self.assertEqual(mapping.frame_ids, (36, 37, 43, 45))
            with exported[0].timing_path.open(newline="") as handle:
                exported_rows = list(csv.DictReader(handle))
            self.assertEqual(len(exported_rows), len(rows))
            for before, after in zip(rows, exported_rows):
                for key, value in before.items():
                    self.assertEqual(value, after[key])
            predictor = FakePredictor()
            csv_path = root / "predictions.csv"
            infer_recording(exported[0], csv_path, predictor=predictor,
                            pose_solver=fake_pose_solver, model_id="test", progress=lambda _: None)
            self.assertEqual(predictor.calls, 4)
            with csv_path.open(newline="") as handle:
                result = list(csv.DictReader(handle))
            self.assertEqual([int(r["frame_i"]) for r in result], [36, 37, 43, 45])
            (target / "sample_raw.mp4").write_bytes(b"ambiguous source")
            with self.assertRaises(BatchInferenceError):
                discover_recordings_with_audit(target)


if __name__ == "__main__":
    unittest.main()
