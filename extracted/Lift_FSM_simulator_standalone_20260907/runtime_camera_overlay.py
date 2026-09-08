"""Adapt synthetic pose packets to the live camera drawing function (no inference)."""
from pathlib import Path
import math
import sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'depth_cam'))
from calib.camera_overlay import draw_camera_overlay
from calib.geometry import pallet_keypoints_3d


def packet_overlay_args(packet, options):
    step = packet['step']
    meta = step['vision_meta']
    intrin = SimpleNamespace(fx=meta['fx'], fy=meta['fy'],
                            ppx=meta['ppx'], ppy=meta['ppy'], coeffs=[0.] * 5)
    args = dict(color_intrin=intrin)
    if not step['det_ok']:
        return args
    centre = np.asarray(step['offset_smooth'], dtype=float)
    yaw = step['yaw_smooth']
    name = meta['selected_front_face']
    width, depth = 1.1, options['pallet_depth']
    # Recover the original labelled object frame from the canonical selected
    # face pose. This keeps 0..7 identities stable when the selected face changes.
    turn, local_centre = {
        'z_min': (0, [0, 0, 0]), 'z_max': (180, [0, 0, depth]),
        'x_min': (90, [-width/2, 0, depth/2]),
        'x_max': (-90, [width/2, 0, depth/2]),
    }[name]
    rotation, _ = cv2.Rodrigues(np.array([0., math.radians(yaw-turn), 0.]))
    origin = centre - rotation @ np.asarray(local_centre)
    points = pallet_keypoints_3d(width, .15, depth) @ rotation.T + origin
    # Never project points behind the optical near plane into plausible labels.
    if not np.isfinite(points).all() or np.any(points[:, 2] <= .001):
        return args
    pixels = points[:, :2] / points[:, 2:3]
    pixels = pixels * [intrin.fx, intrin.fy] + [intrin.ppx, intrin.ppy]
    args.update(kpts_all=np.column_stack((pixels, np.ones(len(pixels)))),
                selected_face_corners_px=meta['face_corners_px'],
                selected_front_face=name, rvec=np.array([0., math.radians(yaw), 0.]),
                tvec=centre, yaw_deg=yaw, pitch_deg=0., roll_deg=0.)
    return args


def draw_packet_overlay(image, packet, options, *, vision_independent=False,
                        skip_detection=False):
    """Match runtime suppression: blind = raw RGB; skipped/missed = centre only."""
    args = packet_overlay_args(packet, options)
    if skip_detection:
        args = dict(color_intrin=args['color_intrin'])
    return draw_camera_overlay(image, **args, vision_independent=vision_independent)
