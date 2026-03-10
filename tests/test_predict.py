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


if __name__ == "__main__":
    unittest.main()
