from __future__ import annotations

import unittest

import torch

from models.chimera import ChimeraODIS


class PredictSmokeTests(unittest.TestCase):
    """Robustness tests for `model.predict(...)` on synthetic inputs."""

    def test_predict_returns_list_of_dicts(self) -> None:
        model = ChimeraODIS(num_classes=1, proto_k=24)
        model.eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            predictions = model.predict(x, original_sizes=[(512, 512)])

        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        prediction = predictions[0]
        self.assertIn("boxes", prediction)
        self.assertIn("scores", prediction)
        self.assertIn("labels", prediction)
        self.assertIn("masks", prediction)

    def test_empty_prediction_case_does_not_crash(self) -> None:
        model = ChimeraODIS(num_classes=1, proto_k=24)
        model.eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            predictions = model.predict(x, original_sizes=[(512, 512)], conf_thresh=1.1)

        prediction = predictions[0]
        self.assertEqual(prediction["boxes"].shape[0], 0)
        self.assertEqual(prediction["scores"].shape[0], 0)
        self.assertEqual(prediction["labels"].shape[0], 0)
        self.assertEqual(prediction["masks"].shape[0], 0)

    def test_prediction_dimensions_are_valid(self) -> None:
        model = ChimeraODIS(num_classes=1, proto_k=24)
        model.eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            predictions = model.predict(x, original_sizes=[(512, 512)])

        prediction = predictions[0]
        self.assertEqual(prediction["boxes"].ndim, 2)
        self.assertEqual(prediction["scores"].ndim, 1)
        self.assertEqual(prediction["labels"].ndim, 1)
        self.assertEqual(prediction["masks"].ndim, 3)
        if prediction["boxes"].shape[0] > 0:
            self.assertEqual(prediction["boxes"].shape[-1], 4)
            self.assertEqual(prediction["scores"].shape[0], prediction["labels"].shape[0])
            self.assertEqual(prediction["masks"].shape[0], prediction["labels"].shape[0])

    def test_detect_task_still_returns_masks_key(self) -> None:
        model = ChimeraODIS(num_classes=1, proto_k=24)
        model.eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            predictions = model.predict(x, original_sizes=[(512, 512)], conf_thresh=0.0, task="detect")

        prediction = predictions[0]
        self.assertIn("masks", prediction)
        self.assertEqual(prediction["masks"].ndim, 3)
        self.assertEqual(prediction["masks"].shape[-2:], (512, 512))
        self.assertEqual(prediction["masks"].shape[0], prediction["boxes"].shape[0])


if __name__ == "__main__":
    unittest.main()


class TestMaskCroppingRegression(unittest.TestCase):
    """Regression: masks must be empty outside their box.

    Cropping *logits* to 0 and then applying sigmoid (0.5) with a ``>= 0.5`` threshold used to
    mark every pixel outside the box as foreground, so predicted masks covered ~the whole image.
    """

    def test_predicted_masks_stay_inside_their_boxes(self) -> None:
        import numpy as np

        from models.chimera import ChimeraODIS

        torch.manual_seed(0)
        model = ChimeraODIS(num_classes=2, proto_k=8).eval()
        size = 128
        x = torch.rand(1, 3, size, size)
        pred = model.predict(x, original_sizes=[(size, size)], conf_thresh=0.0, max_det=6, task="segment")[0]
        self.assertGreater(pred["boxes"].shape[0], 0)

        margin = 10  # prototype stride (4) x bilinear support, with slack
        for box, mask in zip(pred["boxes"].numpy(), pred["masks"].numpy()):
            outside = np.ones((size, size), dtype=bool)
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            outside[max(y1 - margin, 0): y2 + margin, max(x1 - margin, 0): x2 + margin] = False
            self.assertFalse(mask.astype(bool)[outside].any(), "mask leaks outside its bounding box")
