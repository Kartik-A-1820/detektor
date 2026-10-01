from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from export import ExportWrapper
from models.chimera import ChimeraODIS
from utils.export_utils import create_dummy_input, get_dynamic_axes, get_export_names, torch_onnx_export
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
            torch_onnx_export(
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


    def test_export_matches_pytorch_for_trained_style_weights(self) -> None:
        """Regression: exporting through a train-mode wrapper baked BatchNorm batch statistics into the graph.

        Randomly initialised BatchNorm layers (mean 0 / var 1) hide the problem, so give them
        non-trivial running statistics like a trained model has.
        """
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except Exception:
            self.skipTest("onnx/onnxruntime not installed")

        from export import export_onnx
        from models.factory import build_model_from_model_config, resolve_model_config

        torch.manual_seed(0)
        cfg = resolve_model_config({"profile": "firefly"}, num_classes=2)
        model = build_model_from_model_config(cfg, num_classes=2)
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.normal_(0.0, 0.5)
                module.running_var.uniform_(0.5, 2.0)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "m.pt"
            torch.save({"model_state": model.state_dict(), "model_config": cfg,
                        "config": {"train": {"img_size": 96}, "data": {"num_classes": 2}, "model": cfg}}, ckpt)
            result = export_onnx(None, str(ckpt), str(Path(tmp) / "m.onnx"), check_parity=True, opset=13)

        self.assertEqual(result["image_size"], 96)  # defaults to the checkpoint's training size
        self.assertTrue(result["parity_ok"], result["parity"])
        for item in result["parity"]["comparisons"]:
            self.assertLess(item["max_abs_diff"], 1e-3, item)


if __name__ == "__main__":
    unittest.main()
