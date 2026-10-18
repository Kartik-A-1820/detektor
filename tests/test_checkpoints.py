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


class TestCheckpointMetadataPrecedence(unittest.TestCase):
    """Regression: a default ``proto_k`` must not override the checkpoint's model config."""

    def test_checkpoint_proto_k_wins_over_default_argument(self) -> None:
        import torch

        from models.factory import (
            build_model_from_checkpoint,
            build_model_from_model_config,
            load_model_weights,
            resolve_model_config,
        )

        cfg = resolve_model_config({"profile": "firefly"}, num_classes=3)  # firefly uses proto_k=16
        source = build_model_from_model_config(cfg, num_classes=3)
        checkpoint = {"model_state": source.state_dict(), "model_config": cfg}

        rebuilt = build_model_from_checkpoint(checkpoint, num_classes=3, proto_k=24)
        self.assertEqual(rebuilt.proto_k, 16)
        load_model_weights(rebuilt, checkpoint, strict=True)
        self.assertTrue(torch.equal(rebuilt.state_dict()["proto_head.pred.weight"], source.state_dict()["proto_head.pred.weight"]))


class TestTrainImageSize(unittest.TestCase):
    """Serving defaults to the resolution a checkpoint was trained at."""

    def test_reads_size_from_embedded_config(self) -> None:
        from models.factory import checkpoint_train_img_size

        self.assertEqual(checkpoint_train_img_size({"config": {"train": {"img_size": 256}}}), 256)
        self.assertEqual(checkpoint_train_img_size({"config": {"train": {"img_size": "384"}}}), 384)

    def test_missing_or_invalid_metadata_yields_none(self) -> None:
        from models.factory import checkpoint_train_img_size

        for value in ({}, {"config": {}}, {"config": {"train": {}}}, {"config": {"train": {"img_size": 0}}},
                      {"config": {"train": {"img_size": "x"}}}, {"model_state": {}}, None, "oops"):
            self.assertIsNone(checkpoint_train_img_size(value), value)

    def test_load_model_records_train_size(self) -> None:
        import tempfile

        from api.utils import load_model
        from models.factory import build_model_from_model_config, resolve_model_config

        cfg = resolve_model_config({"profile": "firefly"}, num_classes=2)
        model = build_model_from_model_config(cfg, num_classes=2)
        with tempfile.TemporaryDirectory() as tmp:
            with_cfg = Path(tmp) / "with.pt"
            torch.save({"model_state": model.state_dict(), "model_config": cfg,
                        "config": {"train": {"img_size": 288}, "data": {"num_classes": 2}, "model": cfg}}, with_cfg)
            bare = Path(tmp) / "bare.pt"
            torch.save({"model_state": model.state_dict(), "model_config": cfg}, bare)
            self.assertEqual(load_model(str(with_cfg), device_name="cpu")[0].train_img_size, 288)
            self.assertIsNone(load_model(str(bare), device_name="cpu")[0].train_img_size)


class _Unsafe:
    """A non-tensor object: loading it requires full pickle support."""

    def __init__(self) -> None:
        self.payload = 1


class TestSafeCheckpointLoading(unittest.TestCase):
    def test_plain_checkpoint_loads_in_weights_only_mode(self) -> None:
        import tempfile

        from utils.checkpoints import safe_torch_load

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.pt"
            torch.save({"model_state": {"w": torch.ones(2)}, "epoch": 3, "config": {"train": {"img_size": 64}}}, path)
            loaded = safe_torch_load(path)
        self.assertEqual(loaded["epoch"], 3)
        self.assertTrue(torch.equal(loaded["model_state"]["w"], torch.ones(2)))

    def test_arbitrary_objects_are_refused_with_guidance(self) -> None:
        import tempfile

        from utils.checkpoints import UNSAFE_PICKLE_ENV, CheckpointLoadError, safe_torch_load

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evil.pt"
            torch.save({"obj": _Unsafe()}, path)
            with self.assertRaises(CheckpointLoadError) as ctx:
                safe_torch_load(path)
            self.assertIn(UNSAFE_PICKLE_ENV, str(ctx.exception))

    def test_explicit_opt_in_allows_trusted_legacy_files(self) -> None:
        import os
        import tempfile
        import warnings
        from unittest import mock

        from utils.checkpoints import UNSAFE_PICKLE_ENV, safe_torch_load

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save({"obj": _Unsafe()}, path)
            with mock.patch.dict(os.environ, {UNSAFE_PICKLE_ENV: "1"}), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = safe_torch_load(path)
        self.assertEqual(loaded["obj"].payload, 1)
        self.assertTrue(any(UNSAFE_PICKLE_ENV in str(w.message) for w in caught))

    def test_load_model_uses_safe_loading(self) -> None:
        import tempfile

        from api.utils import load_model
        from utils.checkpoints import CheckpointLoadError

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evil.pt"
            torch.save({"model_state": {}, "obj": _Unsafe()}, path)
            with self.assertRaises(CheckpointLoadError):
                load_model(str(path), device_name="cpu")
