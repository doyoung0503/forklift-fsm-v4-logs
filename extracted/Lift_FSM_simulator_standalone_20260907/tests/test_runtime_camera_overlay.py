import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol_world import observation, scenario_options
from runtime_camera_overlay import draw_packet_overlay, packet_overlay_args


class RuntimeOverlayTests(unittest.TestCase):
    def packet(self, yaw=0, missed=False):
        options = scenario_options(dict(vertical_offset=.175))
        sample = dict(error=dict(x=.01, y=-.01, z=.02, yaw=1), interval=.1, missed=missed)
        packet, _ = observation(dict(x=0, z=2, yaw=yaw), 0, .12, 1, options, sample)
        return packet, options

    def test_selected_face_matches_reconstructed_numbered_corners(self):
        indices = dict(z_min=[0,1,2,3], z_max=[4,5,6,7],
                       x_min=[0,3,7,4], x_max=[1,5,6,2])
        faces = set()
        for yaw in (0, 70, -70, 180):
            packet, options = self.packet(yaw)
            args = packet_overlay_args(packet, options)
            name = args['selected_front_face']; faces.add(name)
            reconstructed = args['kpts_all'][indices[name], :2]
            selected = np.asarray(args['selected_face_corners_px'])
            # Canonical selected-face frame can reorder corners, not move them.
            distances = np.linalg.norm(reconstructed[:, None] - selected[None, :], axis=2)
            self.assertLess(distances.min(axis=1).max(), 1e-8)
        self.assertEqual(faces, set(indices))

    def test_miss_and_skip_keep_only_screen_cross_blind_is_raw(self):
        packet, options = self.packet()
        missed, _ = self.packet(missed=True)
        raw = np.zeros((480,640,3), np.uint8)
        skip = draw_packet_overlay(raw.copy(), packet, options, skip_detection=True)
        miss = draw_packet_overlay(raw.copy(), missed, options)
        self.assertTrue(np.array_equal(skip, miss))
        self.assertGreater(np.count_nonzero(miss[230:251,310:331]), 0)
        miss[228:253,308:333] = 0
        self.assertEqual(np.count_nonzero(miss), 0)
        blind = draw_packet_overlay(raw.copy(), packet, options, vision_independent=True)
        self.assertTrue(np.array_equal(raw, blind))

    def test_overlay_does_not_modify_fsm_packet(self):
        packet, options = self.packet(70)
        original = copy.deepcopy(packet)
        draw_packet_overlay(np.zeros((480,640,3),np.uint8), packet, options)
        self.assertEqual(packet, original)


if __name__ == '__main__':
    unittest.main()
