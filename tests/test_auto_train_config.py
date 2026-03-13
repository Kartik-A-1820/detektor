"""Tests for automatic training configuration and training augmentations."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import yaml

from datasets.yolo_seg import YOLOSegDataset
from utils.auto_train_config import resolve_training_config
from utils.data_config import load_dataset_yaml


class TestAutoTrainConfig(unittest.TestCase):
    """Unit tests for hardware-aware training config resolution."""

    def setUp(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / "reports" / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_path = temp_root / "auto_train_config"
        self.temp_path.mkdir(parents=True, exist_ok=True)

    def _create_dataset_yaml(self) -> Path:
        dataset_root = self.temp_path / "toy_dataset"
        for split in ("train", "val"):
            (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx in range(3):
            cv2.imwrite(str(dataset_root / "train" / "images" / f"img_{idx}.jpg"), image)
            (dataset_root / "train" / "labels" / f"img_{idx}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        cv2.imwrite(str(dataset_root / "val" / "images" / "img_0.jpg"), image)
        (dataset_root / "val" / "labels" / "img_0.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

        payload = {
            "train": str(dataset_root / "train" / "images"),
            "val": str(dataset_root / "val" / "images"),
            "nc": 1,
            "names": {0: "ball"},
        }
        yaml_path = self.temp_path / "data.yaml"
        yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return yaml_path

    def test_load_dataset_yaml_normalizes_dict_names(self) -> None:
        yaml_path = self._create_dataset_yaml()

        normalized = load_dataset_yaml(str(yaml_path))

        self.assertEqual(normalized["class_names"], ["ball"])
        self.assertTrue(normalized["train"].endswith("train"))
        self.assertTrue(normalized["val"].endswith("val"))

    def test_resolve_training_config_auto_tunes_from_small_gpu(self) -> None:
        yaml_path = self._create_dataset_yaml()
        fake_props = SimpleNamespace(name="Tiny GPU", total_memory=2 * 1024 ** 3, major=7, minor=5)

        with (
            mock.patch("utils.auto_train_config.torch.cuda.is_available", return_value=True),
            mock.patch("utils.auto_train_config.torch.cuda.get_device_properties", return_value=fake_props),
            mock.patch("utils.auto_train_config.torch.cuda.is_bf16_supported", return_value=False),
        ):
            cfg, summary = resolve_training_config(None, str(yaml_path))

        self.assertEqual(cfg["device"], "cuda")
        self.assertEqual(cfg["train"]["img_size"], 384)
        self.assertEqual(cfg["train"]["batch_size"], 2)
        self.assertEqual(cfg["train"]["grad_accum"], 4)
        self.assertFalse(cfg["train"]["amp"])
        self.assertEqual(cfg["data"]["num_classes"], 1)
        self.assertEqual(cfg["data"]["names"], ["ball"])
        self.assertEqual(cfg["model"]["profile"], "comet")
        self.assertEqual(summary["model_display_name"], "Comet")
        self.assertTrue(summary["out_dir"].startswith("runs"))
        self.assertGreater(summary["augment"]["scale"], 0.0)
        self.assertGreater(summary["augment"]["mosaic"], 0.0)

    def test_resolve_training_config_falls_back_to_cpu_below_half_gb(self) -> None:
        yaml_path = self._create_dataset_yaml()
        fake_props = SimpleNamespace(name="Tiny GPU", total_memory=400 * 1024 ** 2, major=7, minor=5)

        with (
            mock.patch("utils.auto_train_config.torch.cuda.is_available", return_value=True),
            mock.patch("utils.auto_train_config.torch.cuda.get_device_properties", return_value=fake_props),
            mock.patch("utils.auto_train_config.torch.cuda.is_bf16_supported", return_value=False),
        ):
            cfg, summary = resolve_training_config(None, str(yaml_path))

        self.assertEqual(cfg["device"], "cpu")
        self.assertEqual(cfg["model"]["profile"], "firefly")
        self.assertEqual(summary["model_display_name"], "Firefly")


class TestTrainingAugmentations(unittest.TestCase):
    """Unit tests for deterministic augmentation behavior."""

    def setUp(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / "reports" / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_path = temp_root / "training_augmentations"
        self.temp_path.mkdir(parents=True, exist_ok=True)

    def test_horizontal_flip_updates_boxes(self) -> None:
        dataset_root = self.temp_path / "flip_dataset"
        (dataset_root / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels").mkdir(parents=True, exist_ok=True)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :30] = 255
        cv2.imwrite(str(dataset_root / "images" / "sample.jpg"), image)
        (dataset_root / "labels" / "sample.txt").write_text("0 0.25 0.5 0.2 0.4\n", encoding="utf-8")

        dataset = YOLOSegDataset(
            str(dataset_root),
            img_size=100,
            task="detect",
            auto_detect_task=False,
            augment=True,
            augment_cfg={
                "enabled": True,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "fliplr": 1.0,
                "flipud": 0.0,
                "translate": 0.0,
                "scale": 0.0,
            },
        )

        _, target = dataset[0]
        box = target["boxes"][0].tolist()

        self.assertAlmostEqual(box[0], 0.65, places=2)
        self.assertAlmostEqual(box[2], 0.85, places=2)
        self.assertEqual(target["task_mode"], "detect")

    def test_mosaic_combines_multiple_sources(self) -> None:
        dataset_root = self.temp_path / "mosaic_dataset"
        (dataset_root / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels").mkdir(parents=True, exist_ok=True)

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        for idx in range(4):
            cv2.imwrite(str(dataset_root / "images" / f"sample_{idx}.jpg"), image)
            center_x = 0.2 + 0.2 * idx
            (dataset_root / "labels" / f"sample_{idx}.txt").write_text(
                f"0 {center_x} 0.5 0.15 0.2\n",
                encoding="utf-8",
            )

        dataset = YOLOSegDataset(
            str(dataset_root),
            img_size=64,
            task="detect",
            auto_detect_task=False,
            augment=True,
            augment_cfg={
                "enabled": True,
                "mosaic": 1.0,
                "cutmix": 0.0,
                "random_cut": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "fliplr": 0.0,
                "flipud": 0.0,
                "translate": 0.0,
                "scale": 0.0,
            },
        )

        np.random.seed(0)
        with mock.patch("numpy.random.randint", side_effect=[32, 32]):
            _, target = dataset[0]
        self.assertGreaterEqual(target["boxes"].shape[0], 2)

    def test_cutmix_can_replace_instance_set(self) -> None:
        dataset_root = self.temp_path / "cutmix_dataset"
        (dataset_root / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels").mkdir(parents=True, exist_ok=True)

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_root / "images" / "sample_0.jpg"), image)
        cv2.imwrite(str(dataset_root / "images" / "sample_1.jpg"), image)
        (dataset_root / "labels" / "sample_0.txt").write_text("0 0.20 0.20 0.10 0.10\n", encoding="utf-8")
        (dataset_root / "labels" / "sample_1.txt").write_text("1 0.80 0.80 0.10 0.10\n", encoding="utf-8")

        dataset = YOLOSegDataset(
            str(dataset_root),
            img_size=64,
            task="detect",
            auto_detect_task=False,
            augment=True,
            augment_cfg={
                "enabled": True,
                "mosaic": 0.0,
                "cutmix": 1.0,
                "random_cut": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "fliplr": 0.0,
                "flipud": 0.0,
                "translate": 0.0,
                "scale": 0.0,
            },
        )

        with mock.patch("numpy.random.randint", side_effect=[1, 45, 45]):
            with mock.patch("numpy.random.uniform", return_value=0.25):
                _, target = dataset[0]

        labels = sorted(target["labels"].tolist())
        self.assertIn(1, labels)


if __name__ == "__main__":
    unittest.main()
