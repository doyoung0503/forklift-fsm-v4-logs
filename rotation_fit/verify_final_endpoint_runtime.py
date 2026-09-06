"""Offline-only verification: runtime vs evaluation preprocessing on 23 saved frames."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'extracted/depth_cam'))
    from calib.perception import Perception
    from calib import config as pose_cfg
    from calib.fsm_v4 import config as cfg
    from rotation_fit.evaluate_cleanlabel_endpoints import CleanlabelPredictor
    cfg.validate()
    live = Perception()
    offline = CleanlabelPredictor(Path(pose_cfg.MODEL_PATH), device=live.device,
                                  confidence_threshold=.4, front_class_name='item', use_half=True)
    rows = []
    for video in sorted((root / 'rotation_fit/out/command_clips').glob('*_raw.avi')):
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f'Cannot read {video}')
        live_ok, _, _, live_points = live.infer_front(frame)
        reference = offline.predict(frame)
        if live_ok != (reference.keypoints is not None):
            raise AssertionError(f'Live/offline detection disagreement: {video.name}')
        delta = None
        if live_ok:
            delta = float(np.max(np.abs(live_points-reference.keypoints)))
            np.testing.assert_allclose(live_points, reference.keypoints, atol=1e-4, rtol=0)
        rows.append(dict(video=video.name, valid=live_ok, maximum_keypoint_difference=delta))
    output = root / 'rotation_fit/out/cleanlabel_command_clips_20260906/final_selected/runtime_verification.json'
    result = dict(checked_saved_frames=len(rows), live_offline_matched=True, checks=rows,
                  model=pose_cfg.MODEL_PATH, model_sha256=cfg.ROT_GENERATED_ARTIFACT_MODEL_SHA256,
                  response_kind=type(cfg.ROTATION_RESPONSE).__name__,
                  artifact_sha256=cfg.ROT_GENERATED_ARTIFACT_SHA256,
                  ten_degree_hold_s=cfg.ROTATION_RESPONSE.command_seconds(10),
                  camera_opened=False, can_opened=False, hardware_commands_sent=False)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
