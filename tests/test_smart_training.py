from __future__ import annotations

import copy
import unittest
from contextlib import ExitStack
from unittest import mock

import torch

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

    def test_train_once_uses_validation_map_for_best_checkpoint(self) -> None:
        cfg = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
        cfg["device"] = "cpu"
        cfg["train"]["epochs"] = 2
        cfg["train"]["batch_size"] = 1
        cfg["train"]["grad_accum"] = 1
        cfg["train"]["num_workers"] = 0
        cfg["logging"]["out_dir"] = "runs/test_best_metric_selection"

        resolved_runtime = {
            "device": "cpu",
            "gpu_name": "cpu",
            "total_vram_gb": 0.0,
            "resolved_train_root": "train",
            "resolved_val_root": "val",
            "img_size": cfg["train"]["img_size"],
            "batch_size": cfg["train"]["batch_size"],
            "grad_accum": cfg["train"]["grad_accum"],
            "effective_batch_size": 1,
            "lr": cfg["train"]["lr"],
            "amp": False,
            "num_workers": 0,
            "out_dir": cfg["logging"]["out_dir"],
            "model_profile": cfg["model"]["profile"],
            "model_display_name": cfg["model"]["display_name"],
            "augment": copy.deepcopy(cfg["augment"]),
            "smart_training": copy.deepcopy(cfg["smart_training"]),
        }

        class DummyDataset:
            task_mode = "detect"

            def __len__(self):
                return 1

        class DummyModel:
            def __init__(self) -> None:
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.detection_loss = mock.Mock(current_epoch=0)
                self.proto_k = 24
                self.num_classes = 1
                self.backbone = mock.Mock()
                self.backbone.stem.conv.out_channels = 16
                self.backbone.stage_channels = (16, 32, 48, 64)
                self.backbone.stage_depths = (1, 1, 1, 1)
                self.neck = mock.Mock()
                self.neck.out_channels = (32, 48, 64)
                self.detect_head = mock.Mock()
                self.detect_head.feat_channels = 32

            def to(self, device):
                return self

            def parameters(self):
                return [self.weight]

            def train(self):
                return self

            def state_dict(self):
                return {"weight": self.weight.detach().clone()}

            def compute_loss(self, imgs, targets, return_dict=False, debug=False, task="detect"):
                loss = self.weight * 0 + torch.tensor(1.0, requires_grad=True)
                payload = {
                    "loss_total": loss,
                    "loss_cls": loss,
                    "loss_box": loss,
                    "loss_obj": loss,
                    "loss_mask": loss * 0,
                    "loss_mask_bce": loss * 0,
                    "loss_mask_dice": loss * 0,
                    "num_fg": loss * 0,
                    "num_mask_pos": loss * 0,
                }
                return payload if return_dict else payload["loss_total"]

        model = DummyModel()
        dummy_batch = [
            (
                torch.zeros((1, 3, 64, 64), dtype=torch.float32),
                [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long), "masks": torch.zeros((0, 64, 64))}],
            )
        ]
        checkpoint_calls = []

        def fake_save_checkpoint(**kwargs):
            checkpoint_calls.append(
                {
                    "best_metric": kwargs["best_metric"],
                    "is_best": kwargs["is_best"],
                    "checkpoint_path": kwargs["checkpoint_path"],
                }
            )

        with ExitStack() as stack:
            stack.enter_context(mock.patch("train.set_seed"))
            stack.enter_context(mock.patch("train.set_vram_cap"))
            stack.enter_context(mock.patch("train._resolve_device", return_value=torch.device("cpu")))
            stack.enter_context(mock.patch("train.build_model_from_config", return_value=model))
            stack.enter_context(mock.patch("train.get_model_info", return_value={"trainable_params": 1}))
            stack.enter_context(mock.patch("train.build_dataset", return_value=DummyDataset()))
            stack.enter_context(mock.patch("train._build_train_loader", return_value=dummy_batch))
            stack.enter_context(mock.patch("train._build_scheduler", return_value=None))
            stack.enter_context(mock.patch("train.write_resolved_config", return_value="runs/test_best_metric_selection/resolved.yaml"))
            stack.enter_context(mock.patch("train._log_resolved_runtime"))
            stack.enter_context(mock.patch("train.write_json"))
            stack.enter_context(mock.patch("train.append_metrics_row"))
            stack.enter_context(mock.patch("train.append_jsonl"))
            stack.enter_context(mock.patch("train.load_train_metrics", return_value=None))
            stack.enter_context(mock.patch("train.generate_metrics_summary"))
            stack.enter_context(mock.patch("train.plot_loss_curves"))
            stack.enter_context(mock.patch("train.plot_learning_rate"))
            stack.enter_context(mock.patch("train.plot_epoch_metrics"))
            stack.enter_context(mock.patch("train.vram_report", return_value=""))
            stack.enter_context(mock.patch("train.save_checkpoint", side_effect=fake_save_checkpoint))
            stack.enter_context(
                mock.patch(
                    "validate.validate",
                    side_effect=[
                        {"detection": {"precision": 0.1, "recall": 0.1, "ap50": 0.10, "mean_iou": 0.5}},
                        {"detection": {"precision": 0.2, "recall": 0.2, "ap50": 0.25, "mean_iou": 0.6}},
                    ],
                )
            )
            stack.enter_context(mock.patch("train.torch.save"))
            summary = train._train_once(
                cfg=cfg,
                resolved_runtime=resolved_runtime,
                config_path=None,
                data_yaml="F:/data/data.yaml",
                run_val=True,
                val_freq=1,
            )

        self.assertEqual(len(checkpoint_calls), 2)
        self.assertTrue(checkpoint_calls[0]["is_best"])
        self.assertTrue(checkpoint_calls[1]["is_best"])
        self.assertAlmostEqual(checkpoint_calls[0]["best_metric"], 0.10, places=6)
        self.assertAlmostEqual(checkpoint_calls[1]["best_metric"], 0.25, places=6)
        self.assertEqual(summary["best_metric"], 0.25)


if __name__ == "__main__":
    unittest.main()
