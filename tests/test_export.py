from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from export import ExportWrapper
from models.chimera import ChimeraODIS
from utils.export_utils import create_dummy_input, get_dynamic_axes, get_export_names
from utils.parity import compare_pytorch_onnx


class ExportSmokeTests(unittest.TestCase):
    """Lightweight export-related smoke tests with graceful optional skips."""

    def test_export_forward_available(self) -> None:
        model = ChimeraODIS(num_classes=1, proto_k=24)
        self.assertTrue(callable(model.forward_export))

    def test_export_utils_import_and_shapes(self) -> None:
        input_name, output_names = get_export_names()
        self.assertEqual(input_name, "images")
        self.assertEqual(len(output_names), 5)
        self.assertIsNone(get_dynamic_axes(dynamic_batch=False))
        self.assertIn("images", get_dynamic_axes(dynamic_batch=True))
        dummy = create_dummy_input(batch_size=1, image_size=512)
        self.assertEqual(tuple(dummy.shape), (1, 3, 512, 512))

    def test_parity_helper_import(self) -> None:
        self.assertTrue(callable(compare_pytorch_onnx))

    def test_optional_onnx_export_smoke(self) -> None:
        try:
            import onnx  # noqa: F401
        except Exception:
            self.skipTest("onnx is not installed; ONNX export smoke test skipped")

        model = ChimeraODIS(num_classes=1, proto_k=24).eval()
        wrapper = ExportWrapper(model)
        dummy = torch.randn(1, 3, 512, 512)
        input_name, output_names = get_export_names()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "detektor_test.onnx"
            torch.onnx.export(
                wrapper,
                dummy,
                str(output_path),
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=[input_name],
                output_names=output_names,
            )
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
