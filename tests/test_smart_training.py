from __future__ import annotations

import copy
import unittest
from unittest import mock

import train
from utils.auto_train_config import DEFAULT_TRAINING_CONFIG


class TestSmartTrainingOrchestrator(unittest.TestCase):
    def test_train_retries_recoverable_amp_failure(self) -> None:
        cfg = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
        cfg["device"] = "cpu"
        cfg["train"]["amp"] = True
        cfg["logging"]["out_dir"] = "runs/smart_train_test"
        cfg["smart_training"]["base_out_dir"] = cfg["logging"]["out_dir"]
        resolution = {
            "device": "cpu",
            "gpu_name": "cpu",
            "total_vram_gb": 0.0,
            "resolved_train_root": "train",
            "resolved_val_root": "val",
            "img_size": cfg["train"]["img_size"],
            "batch_size": cfg["train"]["batch_size"],
            "grad_accum": cfg["train"]["grad_accum"],
            "effective_batch_size": cfg["train"]["batch_size"] * cfg["train"]["grad_accum"],
            "lr": cfg["train"]["lr"],
            "amp": cfg["train"]["amp"],
            "num_workers": cfg["train"]["num_workers"],
            "out_dir": cfg["logging"]["out_dir"],
            "model_profile": cfg["model"]["profile"],
            "model_display_name": cfg["model"]["display_name"],
            "model": copy.deepcopy(cfg["model"]),
            "augment": copy.deepcopy(cfg["augment"]),
            "smart_training": copy.deepcopy(cfg["smart_training"]),
        }
        attempt_cfgs: list[dict] = []

        def fake_train_once(**kwargs):
            attempt_cfgs.append(copy.deepcopy(kwargs["cfg"]))
            if len(attempt_cfgs) == 1:
                raise train.RecoverableTrainingError("amp_instability", "AMP became unstable")
            return {"final_weights": "runs/smart_train_test_retry02/chimera_final_weights.pt"}

        with (
            mock.patch("train.resolve_training_config", return_value=(cfg, resolution)),
            mock.patch("train._train_once", side_effect=fake_train_once),
        ):
            summary = train.train(config_path=None, data_yaml="F:/data/data.yaml")

        self.assertEqual(len(attempt_cfgs), 2)
        self.assertTrue(attempt_cfgs[0]["train"]["amp"])
        self.assertFalse(attempt_cfgs[1]["train"]["amp"])
        self.assertTrue(attempt_cfgs[1]["logging"]["out_dir"].endswith("retry02"))
        self.assertEqual(summary["smart_training"]["attempts_used"], 2)


if __name__ == "__main__":
    unittest.main()
