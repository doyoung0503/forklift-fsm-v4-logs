"""Read back generated MP4s and save actual decoded frames for visual QA."""
import json
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent


def main():
    qa = BASE / "evaluation/video_qa"
    qa.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted((BASE / "evaluation").glob("**/*.mp4")):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot read generated video: {path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise RuntimeError(f"Empty video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        shape = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        indices = sorted({0, count//2, count-1, min(420,count-1)})
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Undecodable frame {index}: {path}")
            if ("night_full" in str(path) and index == 420) or ("190700" in path.name and index == count//2):
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    raise RuntimeError("Image encoding failed")
                (qa / f"{path.stem}_frame{index}.jpg").write_bytes(encoded.tobytes())
        cap.release()
        reports.append({"video":str(path),"frames":count,"fps":fps,"resolution":shape,"decoded_indices":indices})
    (qa/"verification.json").write_text(json.dumps(reports,indent=2),encoding="utf8")
    print(json.dumps(reports,indent=2))


if __name__ == "__main__":
    main()
