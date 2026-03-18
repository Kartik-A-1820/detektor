from __future__ import annotations

import unittest
from pathlib import Path

import torch

from models.factory import build_model_from_checkpoint, load_model_weights
from utils.checkpoints import build_checkpoint_payload, save_checkpoint


class CheckpointTests(unittest.TestCase):
    """Regression tests for self-contained checkpoint payloads."""

    def test_checkpoint_roundtrip_preserves_architecture_metadata(self) -> None:
        from models.chimera import ChimeraODIS

        model = ChimeraODIS(
            num_classes=3,
            proto_k=20,
            stem_channels=24,
            backbone_channels=(32, 64, 96, 128),
            backbone_depths=(1, 1, 2, 2),
            neck_channels=(64, 96, 128),
            head_feat_channels=64,
        )
        cfg = {
            "model": {
                "profile": "comet",
                "stem_channels": 24,
                "backbone_channels": (32, 64, 96, 128),
                "backbone_depths": (1, 1, 2, 2),
                "neck_channels": (64, 96, 128),
                "head_feat_channels": 64,
                "proto_k": 20,
            },
            "data": {"num_classes": 3},
        }

        payload = build_checkpoint_payload(model, config=cfg)
        rebuilt = build_model_from_checkpoint(payload)
        load_model_weights(rebuilt, payload)

        self.assertEqual(rebuilt.num_classes, 3)
        self.assertEqual(rebuilt.proto_k, 20)
        self.assertEqual(rebuilt.backbone.stem.conv.out_channels, 24)
        self.assertEqual(rebuilt.detect_head.num_classes, 3)
        self.assertEqual(rebuilt.detect_head.num_mask_coeffs, 20)

    def test_save_checkpoint_writes_model_config(self) -> None:
        from models.chimera import ChimeraODIS

        model = ChimeraODIS(num_classes=2, proto_k=16)
        cfg = {"model": {"profile": "firefly", "proto_k": 16}, "data": {"num_classes": 2}}

        tmp_dir = Path("reports/test_tmp/test_checkpoints")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = tmp_dir / "model.pt"
        save_checkpoint(checkpoint_path, model, config=cfg)
        payload = torch.load(checkpoint_path, map_location="cpu")

        self.assertIn("model_state", payload)
        self.assertIn("model_config", payload)
        self.assertEqual(payload["model_config"]["profile"], "firefly")
        self.assertEqual(payload["model_config"]["num_classes"], 2)


if __name__ == "__main__":
    unittest.main()
