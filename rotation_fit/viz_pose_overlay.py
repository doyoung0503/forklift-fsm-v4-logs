# -*- coding: utf-8 -*-
"""9점 객체모델 + PnP 결과 오버레이 검증.

초록 = 모델 키포인트 예측(0~8), 시안 = pnp9 큐보이드 재투영,
빨강 = pnp4(전면 4점 IPPE) 큐보이드 재투영, 축 = pnp9 자세(XYZ = RGB).
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_multi as pm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.environ.get("ROT_REC_DIR",
                     r"c:\Users\dhshs\Downloads\로그수집\extracted\depth_cam\rec")
OUT = os.path.join(HERE, "out", "overlay")

K = np.array([[605.906494140625, 0, 317.59619140625],
              [0, 605.9697875976562, 256.29229736328125], [0, 0, 1]])
D = np.zeros((5, 1))
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_box(img, obj, rvec, tvec, color, th=2):
    pr, _ = cv2.projectPoints(obj[:8], rvec, tvec, K, D)
    p = pr.reshape(-1, 2)
    for a, b in EDGES:
        cv2.line(img, tuple(np.round(p[a]).astype(int)),
                 tuple(np.round(p[b]).astype(int)), color, th, cv2.LINE_AA)
    return p


def main():
    obj = pm.pallet_object_points("box")
    os.makedirs(OUT, exist_ok=True)
    saved = []
    for f in sorted(glob.glob(os.path.join(HERE, "kpts", "*_kpts.npz"))):
        tag = os.path.basename(f)[: -len("_kpts.npz")]
        video = os.path.join(REC, tag + "_raw.mp4")
        if not os.path.exists(video):
            continue
        kp = np.load(f)["kpts"]
        cap = cv2.VideoCapture(video)
        fin = np.where(np.isfinite(kp[:, :, :2]).all(axis=(1, 2)))[0]
        if len(fin) == 0:
            continue
        picks = fin[np.linspace(0, len(fin) - 1, 3).astype(int)]
        for idx in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, img = cap.read()
            if not ok:
                continue
            k = kp[idx]
            o9 = pm.pose_from_kpts_pnp_multi(k, K, D, obj=obj, min_front=2)
            o4 = pm.pose_from_kpts_pnp4(k[:4, :2], K, D, face_w=1.100, face_h=0.150)
            if o4.get("ok"):
                draw_box(img, obj, o4["rvec"], o4["tvec"], (0, 0, 255), 1)
            if o9.get("ok"):
                draw_box(img, obj, o9["rvec"], o9["tvec"], (255, 255, 0), 2)
                org, _ = cv2.projectPoints(np.zeros((1, 3)), o9["rvec"],
                                           o9["tvec"], K, D)
                org = tuple(np.round(org.reshape(2)).astype(int))
                for j, col in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    e = np.zeros(3)
                    e[j] = 0.5
                    pe, _ = cv2.projectPoints(e.reshape(1, 3), o9["rvec"],
                                              o9["tvec"], K, D)
                    cv2.arrowedLine(img, org,
                                    tuple(np.round(pe.reshape(2)).astype(int)),
                                    col, 2, tipLength=0.15)
                txt = "pnp9 yaw %.1f z %.2f rms %.2f n%d" % (
                    o9["yaw"], o9["center"][2], o9["rms"], o9["n_used"])
            else:
                txt = "pnp9 FAIL"
            if o4.get("ok"):
                txt += " | pnp4 yaw %.1f z %.2f" % (o4["yaw"], o4["center"][2])
            for j in range(9):
                p = tuple(np.round(k[j, :2]).astype(int))
                cv2.circle(img, p, 3, (0, 255, 0), -1)
                cv2.putText(img, str(j), (p[0] + 4, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "cyan=pnp9  red=pnp4", (6, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            dst = os.path.join(OUT, "%s_f%05d.jpg" % (tag[-6:], int(idx)))
            # 경로에 한글이 있으면 cv2.imwrite 가 조용히 실패한다 -> imencode 사용
            ok_enc, buf = cv2.imencode(".jpg", img)
            if ok_enc:
                buf.tofile(dst)
            saved.append(dst)
        cap.release()
    print("saved %d images to %s" % (len(saved), OUT))
    for s in saved:
        print("  " + s)


if __name__ == "__main__":
    main()
