import unittest
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from rotation_fit.evaluate_cleanlabel_endpoints import CleanlabelPredictor


class CleanlabelPredictorTests(unittest.TestCase):
    def test_reflect_padding_max_confidence_and_coordinate_restore(self):
        predictor = CleanlabelPredictor.__new__(CleanlabelPredictor)
        predictor.device, predictor.use_half = 'cpu', False
        calls = []
        class Boxes:
            conf = torch.tensor([.5, .9])
            xyxy = torch.tensor([[120., 130., 200., 180.], [110., 120., 300., 200.]])
            def __len__(self): return 2
        points = torch.ones((2, 9, 3))
        points[1, :, 0], points[1, :, 1], points[1, :, 2] = 150, 160, .8
        def fake_predict(**kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=Boxes(), keypoints=SimpleNamespace(data=points))]
        predictor.model = SimpleNamespace(predict=fake_predict)
        image = np.arange(480*640*3, dtype=np.uint8).reshape(480, 640, 3)
        detection = predictor.predict(image)
        expected = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
        np.testing.assert_array_equal(calls[0]['source'], expected)
        np.testing.assert_allclose(detection.keypoints[:, :2], np.tile([50, 60], (9, 1)))
        np.testing.assert_allclose(detection.keypoints[:, 2], .8)
        self.assertAlmostEqual(detection.confidence, .9, places=6)
        self.assertEqual(calls[0]['imgsz'], 640)
        self.assertEqual(calls[0]['conf'], .4)
        self.assertFalse(calls[0]['augment'])

    def test_no_detection_and_wrong_shape(self):
        predictor = CleanlabelPredictor.__new__(CleanlabelPredictor)
        predictor.device, predictor.use_half = 'cpu', False
        predictor.model = SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(boxes=None)])
        self.assertIsNone(predictor.predict(np.zeros((480, 640, 3), np.uint8)).keypoints)
        with self.assertRaises(ValueError): predictor.predict(np.zeros((480, 800, 3), np.uint8))


if __name__ == '__main__':
    unittest.main()
