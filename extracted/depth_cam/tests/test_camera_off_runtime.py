from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

DEPTH_CAM_DIR = Path(__file__).resolve().parents[1]
if str(DEPTH_CAM_DIR) not in sys.path:
    sys.path.insert(0, str(DEPTH_CAM_DIR))

import main_rec as runtime
from ui.diagram_v4 import draw_fsm_v4_diagram_panel


class CameraSwitchTests(unittest.TestCase):
    def test_camera_off_renders_without_touching_camera_or_actuators(self):
        fsm = SimpleNamespace(state="PRECHECK", cmd_status=SimpleNamespace(code="STOP"))
        factory = mock.Mock(return_value=fsm)
        with (
            mock.patch.object(runtime.runtime_config, "CAMERA_ENABLED", False),
            mock.patch.object(runtime, "rs", None),
            mock.patch.object(runtime, "realsense_check_or_exit") as check,
            mock.patch.object(runtime, "run_initialisation") as initialise,
            mock.patch.object(runtime, "Perception") as perception,
            mock.patch.object(runtime, "can_init") as can_init,
            mock.patch.object(runtime, "configure_can_enabled") as configure,
            mock.patch.object(runtime, "can_close") as close,
            mock.patch.object(runtime.cv2, "imshow") as show,
            mock.patch.object(runtime.cv2, "waitKey", return_value=27),
            mock.patch.object(runtime.cv2, "destroyAllWindows") as destroy,
        ):
            runtime.main(fsm_factory=factory, diagram_drawer=draw_fsm_v4_diagram_panel,
                         can_enabled=True)
        check.assert_not_called()
        initialise.assert_not_called()
        perception.assert_not_called()
        can_init.assert_not_called()
        configure.assert_called_once_with(False)
        factory.assert_called_once_with(tracer=None)
        self.assertEqual(show.call_args.args[1].ndim, 3)
        close.assert_called_once()
        destroy.assert_called_once()

    def test_camera_on_uses_real_device_check(self):
        with (
            mock.patch.object(runtime.runtime_config, "CAMERA_ENABLED", True),
            mock.patch.object(runtime, "realsense_check_or_exit", side_effect=SystemExit(1)) as check,
            mock.patch.object(runtime, "_run_camera_off_preview") as preview,
        ):
            with self.assertRaises(SystemExit):
                runtime.main()
        check.assert_called_once()
        preview.assert_not_called()

    def test_explicit_camera_argument_overrides_default(self):
        with (
            mock.patch.object(runtime.runtime_config, "CAMERA_ENABLED", True),
            mock.patch.object(runtime, "_run_camera_off_preview") as preview,
            mock.patch.object(runtime, "realsense_check_or_exit") as check,
        ):
            runtime.main(camera_enabled=False)
        preview.assert_called_once()
        check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
