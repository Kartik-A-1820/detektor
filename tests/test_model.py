from __future__ import annotations

import unittest

import torch

from models.chimera import ChimeraODIS


class ModelSmokeTests(unittest.TestCase):
    """Lightweight CPU-runnable model smoke tests."""

    def test_model_import_and_construction(self) -> None:
        model = ChimeraODIS(num_classes=1, proto_k=24)
        self.assertIsInstance(model, ChimeraODIS)

    def test_forward_output_structure(self) -> None:
        model = ChimeraODIS(num_classes=1, proto_k=24)
        model.eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            outputs = model(x)

        self.assertIn("features", outputs)
        self.assertIn("det", outputs)
        self.assertIn("proto", outputs)
        self.assertEqual(set(outputs["features"].keys()), {"p3", "p4", "p5", "n3", "n4", "n5"})
        self.assertEqual(set(outputs["det"].keys()), {"cls", "box", "obj", "mask_coeff"})
        self.assertEqual(outputs["proto"].ndim, 4)

    def test_forward_export_outputs(self) -> None:
        model = ChimeraODIS(num_classes=1, proto_k=24)
        model.eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            outputs = model.forward_export(x)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 5)
        cls_flat, box_flat, obj_flat, mask_coeff_flat, proto = outputs
        self.assertEqual(cls_flat.ndim, 3)
        self.assertEqual(box_flat.ndim, 3)
        self.assertEqual(obj_flat.ndim, 3)
        self.assertEqual(mask_coeff_flat.ndim, 3)
        self.assertEqual(proto.ndim, 4)
        self.assertEqual(cls_flat.shape[0], 1)
        self.assertEqual(box_flat.shape[-1], 4)
        self.assertEqual(obj_flat.shape[-1], 1)
        self.assertEqual(mask_coeff_flat.shape[-1], 24)
        self.assertEqual(proto.shape[1], 24)


if __name__ == "__main__":
    unittest.main()
