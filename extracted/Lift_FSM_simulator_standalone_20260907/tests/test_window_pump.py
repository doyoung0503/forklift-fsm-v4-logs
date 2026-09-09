import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fsm_window import FSMWindow

class WindowPumpTests(unittest.TestCase):
    def window(self):
        w=FSMWindow.__new__(FSMWindow)
        w.closed=False;w.offscreen=False;w.next_event_pump=0.
        w.camera_future=None;w.title='test';w.close=Mock()
        w.cv2=SimpleNamespace(waitKey=Mock(return_value=-1),getWindowProperty=Mock(return_value=1),WND_PROP_VISIBLE=0,error=RuntimeError)
        return w

    def test_frequent_ticks_share_wall_time_limit_and_resume(self):
        w=self.window()
        with patch('fsm_window.time.perf_counter',return_value=10.) as clock:
            self.assertTrue(w.pump())
            for _ in range(100):self.assertFalse(w.pump())
            clock.return_value=10.032
            self.assertFalse(w.pump())
            clock.return_value=10.034
            self.assertTrue(w.pump())
            clock.return_value=50.
            self.assertTrue(w.pump())
        self.assertEqual(w.cv2.waitKey.call_count,3)

    def test_presentation_flush_and_close_key(self):
        w=self.window()
        with patch('fsm_window.time.perf_counter',return_value=10.):
            w.pump()
            self.assertTrue(w.pump(force=True))
            self.assertFalse(w.pump())
            w.cv2.waitKey.return_value=27
            w.pump(force=True)
        w.close.assert_called_once()

    def test_offscreen_never_pumps(self):
        w=self.window();w.offscreen=True
        self.assertFalse(w.pump(force=True))
        w.cv2.waitKey.assert_not_called()
