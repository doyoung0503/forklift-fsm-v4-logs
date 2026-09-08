"""Request/result causality, captured GT, and the real controller boundary."""
import unittest

from protocol_world import RequestedWorld
from process_runtime import Session, replay_trace


class RequestedInferenceTests(unittest.TestCase):
    def test_no_results_before_request_or_before_delay(self):
        world = RequestedWorld(dict(latency=.2, latency_sd=0, perception='oracle',
                                    x_bias=.1, z_bias=.2))
        world.advance(.5, [])
        self.assertEqual(world.model_count, 0)
        self.assertTrue(world.observe()['pending'])
        self.assertTrue(world.request_model())
        self.assertFalse(world.request_model())
        captured_z = world.plant.z
        # Movement after capture must not leak into the already pending result.
        world.plant.z += .4
        world.advance(.19, [])
        self.assertEqual(world.model_count, 0)
        world.advance(.01, [])
        packet = world.observe()
        self.assertAlmostEqual(packet['request_time_s'], .5)
        self.assertAlmostEqual(packet['sim_time_s'], .7)
        self.assertAlmostEqual(packet['step']['dist_z'], captured_z+.2)
        self.assertAlmostEqual(packet['step']['offset_smooth'][0], .1)
        world.advance(.5, [])
        self.assertEqual(world.model_count, 1)

    def test_zero_delay_and_seeded_gaussian_samples(self):
        options = dict(latency=0, latency_sd=.05, x_noise=.02, seed=15)
        worlds = [RequestedWorld(options), RequestedWorld(options)]
        for _ in range(12):
            for world in worlds:
                world.request_model()
                world.advance(.5, [])
            self.assertEqual(worlds[0].observe(), worlds[1].observe())
            self.assertGreaterEqual(worlds[0].last_diagnostic['latency'], 0)
        immediate = RequestedWorld(dict(latency=0, latency_sd=0))
        immediate.request_model()
        self.assertEqual(immediate.model_count, 1)
        self.assertEqual(immediate.observe()['sim_time_s'], 0)

    def test_actual_fsm_requests_stop_in_blind_states_and_replay(self):
        session = Session(dict(perception='oracle', latency=.12, model_hz=10), log_root=None)
        report = session.run()
        self.assertIsNone(report['controller_error'])
        self.assertEqual(report['options']['inference_mode'], 'requested')
        self.assertEqual(report['provenance']['loaded_camera_inference_modules'], [])
        self.assertTrue(report['success'], report['reasons'])
        self.assertTrue(any(f.get('vision_independent') for f in report['frames']))
        by_time = {round(f['t'], 8): f for f in report['frames']}
        packets = {r['model_packet']['sequence']: r['model_packet'] for r in report['trace']
                   if r['model_updated']}
        self.assertGreater(len(packets), 1)
        for packet in packets.values():
            requested = packet['request_time_s']
            self.assertGreaterEqual(packet['sim_time_s'], requested+.12-1e-9)
            # State at request is the previous tick's output, before this tick steps.
            previous = [f for t, f in by_time.items() if t < requested-1e-8]
            if previous:
                self.assertTrue(previous[-1]['model_requested'])
        self.assertTrue(replay_trace(report['trace'], source_id=report['source_id'])['passed'])


if __name__ == '__main__':
    unittest.main()
