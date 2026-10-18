"""Both common YOLO dataset layouts must work end to end (loader, auto-config, checker)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from check_dataset import main as check_main
from datasets.yolo_seg import YOLOSegDataset
from utils.auto_train_config import resolve_training_config
from utils.data_config import load_dataset_yaml, resolve_split_dirs


def _write_split(images_dir: Path, labels_dir: Path, count: int, prefix: str) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        cv2.imwrite(str(images_dir / f"{prefix}{i}.jpg"), np.full((64, 64, 3), 100 + i, dtype=np.uint8))
        (labels_dir / f"{prefix}{i}.txt").write_text("1 0.5 0.5 0.4 0.3\n0 0.2 0.3 0.1 0.2\n")


class LayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def roboflow(self) -> Path:
        ds = self.root / "rf"
        for split, n in (("train", 4), ("valid", 2)):
            _write_split(ds / split / "images", ds / split / "labels", n, split)
        yaml_path = ds / "data.yaml"
        yaml_path.write_text(yaml.safe_dump({"train": "train/images", "val": "valid/images", "nc": 2, "names": ["a", "b"]}))
        return yaml_path

    def ultralytics(self, with_nc: bool = False) -> Path:
        ds = self.root / "ul" / "mydata"  # the YAML sits *inside* the dataset folder it names
        for split, n in (("train", 4), ("val", 2)):
            _write_split(ds / "images" / split, ds / "labels" / split, n, split)
        payload = {"path": "mydata", "train": "images/train", "val": "images/val", "names": {0: "neg", 1: "pos"}}
        if with_nc:
            payload["nc"] = 2
        yaml_path = ds / "mydata.yaml"
        yaml_path.write_text(yaml.safe_dump(payload))
        return yaml_path

    # -- resolver -------------------------------------------------------------------------------
    def test_resolve_split_dirs_for_both_layouts(self) -> None:
        rf = self.root / "rf" / "train"
        (rf / "images").mkdir(parents=True)
        self.assertEqual(resolve_split_dirs(rf), (rf / "images", rf / "labels"))
        ul = self.root / "ul" / "images" / "train"
        self.assertEqual(resolve_split_dirs(ul), (ul, self.root / "ul" / "labels" / "train"))
        flat = self.root / "flat"
        self.assertEqual(resolve_split_dirs(flat), (flat, self.root / "labels"))

    # -- loader ---------------------------------------------------------------------------------
    def test_roboflow_layout_loads(self) -> None:
        info = load_dataset_yaml(str(self.roboflow()))
        ds = YOLOSegDataset(info["train"], img_size=64, task="detect")
        self.assertEqual(len(ds), 4)
        _, target = ds[0]
        self.assertEqual(int(target["boxes"].shape[0]), 2)
        self.assertEqual(info["num_classes"], 2)

    def test_ultralytics_layout_loads_with_path_key_and_names_only(self) -> None:
        info = load_dataset_yaml(str(self.ultralytics()))
        self.assertEqual(info["num_classes"], 2)  # inferred from `names`
        self.assertEqual(info["class_names"], ["neg", "pos"])
        self.assertTrue(info["train"].endswith("images/train") or info["train"].endswith("images\\train"))
        train = YOLOSegDataset(info["train"], img_size=64, task="detect")
        val = YOLOSegDataset(info["val"], img_size=64, task="detect")
        self.assertEqual((len(train), len(val)), (4, 2))
        _, target = train[0]
        self.assertEqual(int(target["boxes"].shape[0]), 2)  # labels were found under labels/train

    def test_absolute_path_key_and_explicit_nc(self) -> None:
        yaml_path = self.ultralytics(with_nc=True)
        payload = yaml.safe_load(yaml_path.read_text())
        payload["path"] = str(yaml_path.parent)  # absolute
        yaml_path.write_text(yaml.safe_dump(payload))
        info = load_dataset_yaml(str(yaml_path))
        self.assertEqual(len(YOLOSegDataset(info["val"], img_size=64, task="detect")), 2)

    # -- auto config ----------------------------------------------------------------------------
    def test_auto_config_counts_images_in_both_layouts(self) -> None:
        for yaml_path in (self.roboflow(), self.ultralytics()):
            cfg, summary = resolve_training_config(None, str(yaml_path))
            self.assertEqual(cfg["data"]["num_classes"], 2)
            self.assertGreater(summary.get("dataset_size", 0), 0, yaml_path)

    # -- checker --------------------------------------------------------------------------------
    def test_check_dataset_accepts_both_layouts_from_any_cwd(self) -> None:
        import os
        import sys
        from unittest import mock

        cwd = os.getcwd()
        os.chdir(self.root)  # relative split paths must resolve against the YAML, not the cwd
        try:
            for yaml_path in (self.roboflow(), self.ultralytics()):
                with mock.patch.object(sys, "argv", ["check_dataset.py", "--data-yaml", str(yaml_path), "--output-dir", str(self.root / "rep")]):
                    self.assertEqual(check_main(), 0, yaml_path)
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
