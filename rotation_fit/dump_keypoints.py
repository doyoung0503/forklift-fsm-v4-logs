# -*- coding: utf-8 -*-
"""녹화된 raw 영상에 모델을 다시 돌려 전체 키포인트를 npz 로 덤프.

- raw.mp4 는 rec_fps 측정이 끝난 뒤부터 기록되므로 frame_i = raw_index + OFFSET.
  OFFSET 은 inference_timing.csv 행수 - 영상 프레임수 로 확인한다.
- 여기서는 pose 를 풀지 않고 키포인트(x, y, vis)와 bbox/conf 만 저장한다.
  객체 모델/ PnP 는 별도 단계에서 붙인다.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

REC = os.environ.get(
    "ROT_REC_DIR", r"c:\Users\dhshs\Downloads\로그수집\extracted\depth_cam\rec")
MODEL = os.environ.get(
    "ROT_MODEL", r"c:\Users\dhshs\Downloads\로그수집\extracted\pallet_yolo26n_pose_ft.pt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kpts")

CONF_THR = 0.30              # calib/config.py CONF_THR 와 동일
TARGET_AR = 1100.0 / 125.0   # perception.py 의 검출 선택 기준


def select_detection(boxes_xyxy, confs, classes, front_idx):
    idxs = [i for i, (c, p) in enumerate(zip(classes, confs))
            if (c == front_idx and p >= CONF_THR)]
    if not idxs:
        return None

    def key(i):
        x1, y1, x2, y2 = boxes_xyxy[i]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return (abs(w / h - TARGET_AR), -float(confs[i]))

    return min(idxs, key=key)


def run(video, model, offset):
    from ultralytics import YOLO
    net = YOLO(model)
    names = net.names
    front_idx = None
    for k, v in (names or {}).items():
        if v == "item":
            front_idx = int(k)
    if front_idx is None and names:
        front_idx = int(next(iter(names.keys())))

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    K = None
    kp, bb, cf, fi = [], [], [], []
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        res = net.predict(source=img, device="cpu", verbose=False,
                          conf=CONF_THR)[0]
        row_k = None
        row_b = np.full(4, np.nan)
        row_c = np.nan
        if res.boxes is not None and len(res.boxes) and res.keypoints is not None \
                and res.keypoints.data is not None and len(res.keypoints.data):
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy().astype(float)
            xyxy = res.boxes.xyxy.cpu().numpy().astype(float)
            kk = res.keypoints.data.cpu().numpy().astype(np.float32)
            b = select_detection(xyxy, conf, cls, front_idx)
            if b is not None:
                row_k = kk[b]
                row_b = xyxy[b]
                row_c = conf[b]
        if K is None and row_k is not None:
            K = row_k.shape[0]
        kp.append(row_k)
        bb.append(row_b)
        cf.append(row_c)
        fi.append(idx + offset)
        idx += 1
        if idx % 250 == 0:
            print("  %d/%d" % (idx, n), flush=True)
    cap.release()
    K = K or 9
    arr = np.full((len(kp), K, 3), np.nan, dtype=np.float32)
    for i, r in enumerate(kp):
        if r is not None:
            arr[i, :r.shape[0]] = r
    return arr, np.asarray(bb), np.asarray(cf), np.asarray(fi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for video in sorted(glob.glob(os.path.join(REC, "*_raw.mp4"))):
        base = video[: -len("_raw.mp4")]
        tag = os.path.basename(base)
        if args.only and args.only not in tag:
            continue
        timing = base + "_inference_timing.csv"
        if not os.path.exists(timing):
            print("skip (no timing):", tag)
            continue
        cap = cv2.VideoCapture(video)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        offset = len(pd.read_csv(timing)) - nframes + 1   # frame_i 는 1부터
        dst = os.path.join(OUT, tag + "_kpts.npz")
        if os.path.exists(dst):
            print("exists, skip:", os.path.basename(dst))
            continue
        print("[%s] frames=%d offset=%d" % (tag, nframes, offset), flush=True)
        arr, bb, cf, fi = run(video, MODEL, offset)
        np.savez_compressed(dst, kpts=arr, bbox=bb, conf=cf, frame_i=fi,
                            offset=offset)
        print("saved", dst, arr.shape, flush=True)


if __name__ == "__main__":
    main()
