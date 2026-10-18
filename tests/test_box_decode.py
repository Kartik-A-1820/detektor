"""Stride-unit box decoding (new default) and backward compatibility with pixel-decoding checkpoints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from models.chimera import ChimeraODIS
from models.factory import (
    build_model_from_checkpoint,
    build_model_from_model_config,
    load_model_weights,
    resolve_model_config,
)
from utils.box_ops import BOX_DECODE_PIXELS, BOX_DECODE_STRIDE, distances_to_boxes
from utils.checkpoints import save_checkpoint


class DecodeMathTests(unittest.TestCase):
    def test_pixel_mode_is_the_legacy_relu(self) -> None:
        points = torch.tensor([[100.0, 100.0]])
        raw = torch.tensor([[[10.0, -5.0, 20.0, 0.0]]])
        boxes = distances_to_boxes(points, raw)  # default mode must stay legacy
        self.assertEqual(boxes[0, 0].tolist(), [90.0, 100.0, 120.0, 100.0])

    def test_stride_mode_scales_with_stride_and_uses_softplus(self) -> None:
        points = torch.tensor([[64.0, 64.0], [64.0, 64.0]])
        strides = torch.tensor([8.0, 32.0])
        raw = torch.zeros(1, 2, 4)
        boxes = distances_to_boxes(points, raw, strides, BOX_DECODE_STRIDE)
        half = float(torch.nn.functional.softplus(torch.tensor(0.0)))  # ln 2
        self.assertAlmostEqual(float(boxes[0, 0, 2] - 64.0), half * 8, places=4)
        self.assertAlmostEqual(float(boxes[0, 1, 2] - 64.0), half * 32, places=4)

    def test_stride_mode_has_no_dead_zone(self) -> None:
        points = torch.tensor([[50.0, 50.0]])
        strides = torch.tensor([16.0])
        for mode, expect_grad in ((BOX_DECODE_STRIDE, True), (BOX_DECODE_PIXELS, False)):
            raw = torch.full((1, 1, 4), -3.0, requires_grad=True)  # strongly negative raw output
            distances_to_boxes(points, raw, strides, mode).sum().backward()
            self.assertEqual(bool(raw.grad.abs().sum() > 0), expect_grad, mode)

    def test_validation_errors(self) -> None:
        points, raw = torch.zeros(1, 2), torch.zeros(1, 1, 4)
        with self.assertRaises(AssertionError):
            distances_to_boxes(points, raw, None, BOX_DECODE_STRIDE)
        with self.assertRaises(ValueError):
            distances_to_boxes(points, raw, torch.ones(1), "bogus")


class ModelAndCheckpointTests(unittest.TestCase):
    def test_new_models_default_to_stride_decode_and_run(self) -> None:
        model = ChimeraODIS(num_classes=2, proto_k=8).eval()
        self.assertEqual(model.box_decode, BOX_DECODE_STRIDE)
        out = model.predict(torch.rand(1, 3, 64, 64), conf_thresh=0.0, task="detect")[0]
        self.assertTrue(torch.isfinite(out["boxes"]).all())
        model.train()
        targets = [{"boxes": torch.tensor([[0.2, 0.2, 0.7, 0.8]]), "labels": torch.tensor([1]), "masks": torch.ones(1, 64, 64)}]
        loss = model.compute_loss(torch.rand(1, 3, 64, 64), targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_initial_boxes_are_sensible_without_training(self) -> None:
        """With stride decoding an untrained head already proposes boxes of a plausible size at every level."""
        model = ChimeraODIS(num_classes=1, proto_k=8).eval()
        out = model.predict(torch.rand(1, 3, 128, 128), conf_thresh=0.0, topk_pre_nms=2000, max_det=1000, task="detect", iou_thresh=1.0)[0]
        widths = out["boxes"][:, 2] - out["boxes"][:, 0]
        self.assertGreater(float(widths.median()), 4.0)

    def test_checkpoints_without_the_option_rebuild_as_legacy_pixels(self) -> None:
        cfg = resolve_model_config({"profile": "firefly"}, num_classes=2)
        legacy_cfg = {k: v for k, v in cfg.items() if k != "box_decode"}
        model = build_model_from_model_config({**legacy_cfg, "box_decode": BOX_DECODE_PIXELS}, num_classes=2)
        for checkpoint in (
            {"model_state": model.state_dict(), "model_config": legacy_cfg},
            {"model_state": model.state_dict(), "config": {"model": legacy_cfg, "data": {"num_classes": 2}}},
        ):
            rebuilt = build_model_from_checkpoint(checkpoint, num_classes=2)
            self.assertEqual(rebuilt.box_decode, BOX_DECODE_PIXELS)
            load_model_weights(rebuilt, checkpoint)

        # A bare state dict carries no metadata: it is rebuilt as the default (quasar) architecture, legacy decode.
        bare = ChimeraODIS(num_classes=2, proto_k=24, box_decode=BOX_DECODE_PIXELS).state_dict()
        rebuilt = build_model_from_checkpoint(bare, num_classes=2)
        self.assertEqual(rebuilt.box_decode, BOX_DECODE_PIXELS)
        load_model_weights(rebuilt, bare)

    def test_new_checkpoints_round_trip_the_decode_mode(self) -> None:
        cfg = resolve_model_config({"profile": "firefly"}, num_classes=2)
        model = build_model_from_model_config(cfg, num_classes=2).eval()
        self.assertEqual(model.box_decode, BOX_DECODE_STRIDE)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            # training configs do not carry box_decode in their `model` block; the model attribute must be saved
            save_checkpoint(path, model, config={"model": {k: v for k, v in cfg.items() if k != "box_decode"},
                                                 "data": {"num_classes": 2}, "train": {"img_size": 64}})
            from utils.checkpoints import safe_torch_load

            checkpoint = safe_torch_load(path)
            self.assertEqual(checkpoint["model_config"]["box_decode"], BOX_DECODE_STRIDE)
            rebuilt = build_model_from_checkpoint(checkpoint)
            self.assertEqual(rebuilt.box_decode, BOX_DECODE_STRIDE)
            load_model_weights(rebuilt, checkpoint)
            rebuilt.eval()
            x = torch.rand(1, 3, 64, 64)
            torch.manual_seed(0)
            a = model.predict(x, conf_thresh=0.0, task="detect")[0]["boxes"]
            b = rebuilt.predict(x, conf_thresh=0.0, task="detect")[0]["boxes"]
            self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
