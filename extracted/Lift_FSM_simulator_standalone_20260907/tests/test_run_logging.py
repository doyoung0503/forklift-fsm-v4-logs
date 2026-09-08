import csv
import json
import tempfile
import unittest
from pathlib import Path

from process_runtime import Session


def records(directory,name):
    return [json.loads(line) for line in (directory/name).read_text(encoding='utf-8').splitlines()]


class RunLoggingTests(unittest.TestCase):
    def test_complete_run_streams_all_can_models_and_steps_without_capture(self):
        with tempfile.TemporaryDirectory() as root:
            s=Session(dict(perception='oracle',model_hz=5),capture=False,log_root=root)
            report=s.run()
            self.assertIsNone(report['logging']['error'])
            self.assertEqual(report['logging']['status'],'closed')
            directory=Path(report['logging']['directory'])
            can=records(directory,'run_can_tx.jsonl')
            self.assertEqual([{k:r[k] for k in frame} for r,frame in zip(can,s.world.can_log)],s.world.can_log)
            self.assertEqual(len(can),len(s.world.can_log))
            models=records(directory,'run_model_results.jsonl')
            self.assertEqual([r['frame_i'] for r in models],list(range(s.world.model_count)))
            self.assertTrue(all(r['inference_ran'] is False for r in models))
            self.assertTrue(all(r['packet']['clock']['now']==r['t_mono'] for r in models))
            steps=records(directory,'run_fsm_steps.jsonl')
            self.assertGreater(len(steps),len(models))
            self.assertEqual(steps[-1]['state'],report['state'])
            events=records(directory,'run_control_seq.jsonl')
            self.assertEqual(events[0]['state'],'PRECHECK')
            self.assertEqual(events[-1]['lifecycle'],report['lifecycle'])
            with (directory/'run_pallet_state.csv').open(encoding='utf-8-sig',newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),len(models))
            summary=json.loads((directory/'run_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['log_counts']['can_frames'],len(can))
            self.assertEqual(summary['logging']['status'],'closed')
            with (directory/'run_relative_pose.csv').open(encoding='utf-8-sig',newline='') as handle:
                poses=list(csv.DictReader(handle))
            self.assertEqual(len(poses),len(steps)+2)
            self.assertEqual(poses[0]['phase'],'initial')
            self.assertEqual(poses[-1]['phase'],'final')
            for pose in poses:
                self.assertAlmostEqual(float(pose['pallet_world_x_m']),0)
                # Relative default: pivot Z=0, camera Z=0.68, pallet 2m ahead.
                self.assertAlmostEqual(float(pose['pallet_world_z_m']),2.68)
            self.assertAlmostEqual(float(poses[-1]['pallet_camera_z_m']),report['final']['z'])
            self.assertEqual(report['frames'],[])

    def test_truth_pose_coordinates_use_pivot_and_ignore_model_error(self):
        with tempfile.TemporaryDirectory() as root:
            s=Session(dict(placement_mode='world',forklift_x=1,forklift_z=2,forklift_heading=90,
                pallet_x=3.68,pallet_z=2,pallet_heading=-90,x_bias=.2,z_bias=.1),log_root=root)
            try:
                s.step()
                directory=s.logger.directory
                with (directory/'run_relative_pose.csv').open(encoding='utf-8-sig',newline='') as handle:
                    pose=next(csv.DictReader(handle))
                expected=dict(camera_world_x_m=1.68,camera_world_z_m=2,pallet_world_x_m=3.68,pallet_world_z_m=2,
                    pallet_heading_deg=-90,pallet_camera_x_m=0,pallet_camera_z_m=2,pallet_pivot_z_m=2.68,
                    pivot_to_pallet_distance_m=2.68,fork_tip_world_x_m=2.86,fork_tip_world_z_m=2,
                    fork_tip_pallet_x_m=0,fork_tip_pallet_z_m=-.82,left_outer_tip_pallet_x_m=-.3,
                    right_outer_tip_pallet_x_m=.3,tip_insertion_depth_m=0)
                for key,value in expected.items():
                    self.assertAlmostEqual(float(pose[key]),value,msg=key)
                self.assertEqual(records(directory,'run_model_results.jsonl'), [])
                while s.world.model_count == 0:
                    s.step()
                model=records(directory,'run_model_results.jsonl')[0]['packet']['step']
                self.assertAlmostEqual(model['dist_z'],2.1)
                self.assertAlmostEqual(model['offset_smooth'][0],.2)
            finally:
                s.close()

    def test_step_flush_and_cancel_stop_are_persisted_and_ids_unique(self):
        with tempfile.TemporaryDirectory() as root:
            s=Session(log_root=root)
            self.assertEqual(list(Path(root).iterdir()),[])  # Preview is not a run.
            s.step()
            directory=Path(s.logging_info()['directory'])
            self.assertEqual(len(records(directory,'run_fsm_steps.jsonl')),1)
            self.assertEqual(len(records(directory,'run_model_results.jsonl')),s.world.model_count)
            s.close()
            can=records(directory,'run_can_tx.jsonl')
            self.assertEqual(can[-1]['data'],[127]*8)
            self.assertEqual(records(directory,'run_control_seq.jsonl')[-1]['lifecycle']['outcome'],'cancelled')
            other=Session(log_root=root)
            try:
                other.step()
                self.assertNotEqual(other.logger.run_id,s.logger.run_id)
            finally:
                other.close()

    def test_worker_failure_finishes_partial_logs(self):
        with tempfile.TemporaryDirectory() as root:
            s=Session(log_root=root)
            s.step()
            s.controller.process.kill();s.controller.process.wait()
            result=s.run()
            self.assertEqual(result['lifecycle']['outcome'],'error')
            self.assertEqual(result['logging']['status'],'closed')
            directory=Path(result['logging']['directory'])
            summary=json.loads((directory/'run_summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['controller_error'])
            self.assertEqual(len(records(directory,'run_can_tx.jsonl')),len(s.world.can_log))
            self.assertEqual(len(records(directory,'run_model_results.jsonl')),s.world.model_count)

    def test_disk_error_is_visible_and_does_not_change_controller(self):
        with tempfile.TemporaryDirectory() as root:
            bad=Path(root)/'file';bad.write_text('not a directory',encoding='utf-8')
            s=Session(dict(max_seconds=1),log_root=bad)
            result=s.run()
            self.assertEqual(result['logging']['status'],'error')
            self.assertTrue(result['logging']['error'])
            self.assertEqual(result['lifecycle']['outcome'],'timeout')


if __name__=='__main__':
    unittest.main()
