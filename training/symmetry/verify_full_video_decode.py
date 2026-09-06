"""Decode every input/output frame; capture native FFmpeg warnings per file."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

CHILD = """
import cv2, json, sys
cv2.setNumThreads(1)
cap = cv2.VideoCapture(sys.argv[1], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
assert cap.isOpened(), sys.argv[1]
expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    count += 1
cap.release()
print(json.dumps({'frames': count, 'expected': expected}))
assert count == expected, (count, expected)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "status.json").read_text(encoding="utf8"))
    jobs = []
    for item in manifest["completed"]:
        jobs += [("source", Path(item["source"])), ("comparison", args.directory / item["output"])]

    def check(job):
        kind, path = job
        result = subprocess.run([sys.executable, "-c", CHILD, str(path)], capture_output=True, text=True)
        return {"kind": kind, "path": str(path), "exit_code": result.returncode,
                "counts": json.loads(result.stdout) if result.returncode == 0 else None,
                "decoder_messages": result.stderr.strip()}

    report = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(check, job) for job in jobs]):
            item = future.result()
            report.append(item)
            print(f"CHECK {len(report)}/{len(jobs)} {item['kind']} {Path(item['path']).name} "
                  f"exit={item['exit_code']} warnings={bool(item['decoder_messages'])}", flush=True)
    report.sort(key=lambda item: (item["kind"], item["path"]))
    (args.directory / "full_decode_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    failures = [item for item in report if item["exit_code"]]
    warnings = [item for item in report if item["decoder_messages"]]
    print(json.dumps({"files": len(report), "failures": failures, "warnings": warnings},
                     indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
