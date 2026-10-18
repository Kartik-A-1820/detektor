"""Dataset registry, safe extraction, checksum verification and subsetting (offline)."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

from benchmarks import datasets


def _make_dataset_zip(path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for split, n in (("train", 6), ("val", 3)):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for i in range(n):
                cv2.imwrite(str(root / "images" / split / f"{split}{i}.jpg"), np.full((32, 32, 3), 50 + i, np.uint8))
                (root / "labels" / split / f"{split}{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        (root / "toy.yaml").write_text(yaml.safe_dump({"path": "toy", "train": "images/train", "val": "images/val", "names": {0: "thing"}}))
        with zipfile.ZipFile(path, "w") as zf:
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(root).as_posix())


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch = mock.patch.dict(os.environ, {"DETEKTOR_DATA_DIR": str(self.root / "cache")})
        self.patch.start()
        self.zip = self.root / "toy.zip"
        _make_dataset_zip(self.zip)
        self.sha = hashlib.sha256(self.zip.read_bytes()).hexdigest()
        self.spec = datasets.DatasetSpec("toy", self.zip.as_uri(), self.sha, 0.0, "toy", "n/a", "toy.yaml")
        self.registry = mock.patch.dict(datasets.DATASETS, {"toy": self.spec})
        self.registry.start()

    def tearDown(self) -> None:
        self.registry.stop()
        self.patch.stop()
        self.tmp.cleanup()

    def test_registry_entries_are_complete_and_pinned(self) -> None:
        for name, spec in datasets.DATASETS.items():
            if name == "toy":
                continue
            self.assertEqual(len(spec.sha256), 64, name)
            self.assertTrue(spec.url.startswith("https://"), name)
            self.assertTrue(spec.license, name)

    def test_fetch_verifies_extracts_and_is_idempotent(self) -> None:
        yaml_path = datasets.fetch("toy")
        self.assertTrue(yaml_path.exists())
        self.assertEqual(yaml_path, datasets.fetch("toy"))  # cached, no re-download
        from utils.data_config import load_dataset_yaml

        info = load_dataset_yaml(str(yaml_path))
        self.assertEqual(info["num_classes"], 1)
        self.assertTrue(Path(info["train"]).exists())

    def test_checksum_mismatch_is_rejected_and_cleaned_up(self) -> None:
        with self.assertRaises(RuntimeError):
            datasets.fetch("toy", sha256="0" * 64)
        self.assertFalse((datasets.cache_dir() / "toy").exists())
        self.assertFalse((datasets.cache_dir() / "toy.zip.part").exists())

    def test_unknown_dataset(self) -> None:
        with self.assertRaises(KeyError):
            datasets.fetch("nope")

    def test_zip_slip_is_blocked(self) -> None:
        evil = self.root / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../escaped.txt", "x")
        with self.assertRaises(RuntimeError):
            datasets.safe_extract(evil, self.root / "dest")
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_make_subset_is_deterministic_and_keeps_labels(self) -> None:
        yaml_path = datasets.fetch("toy")
        a = datasets.make_subset(yaml_path, self.root / "a", max_train=3, max_val=2, seed=1)
        b = datasets.make_subset(yaml_path, self.root / "b", max_train=3, max_val=2, seed=1)
        names = lambda d: sorted(p.name for p in (d.parent / "train" / "images").iterdir())  # noqa: E731
        self.assertEqual(names(a), names(b))
        self.assertEqual(len(names(a)), 3)
        self.assertEqual(len(list((a.parent / "train" / "labels").iterdir())), 3)
        self.assertEqual(len(list((a.parent / "val" / "images").iterdir())), 2)
        sub = yaml.safe_load(a.read_text())
        self.assertEqual(sub["nc"], 1)
        # the subset must itself be loadable by the training stack
        from datasets.yolo_seg import YOLOSegDataset
        from utils.data_config import load_dataset_yaml

        info = load_dataset_yaml(str(a))
        self.assertEqual(len(YOLOSegDataset(info["train"], img_size=32, task="detect")), 3)


if __name__ == "__main__":
    unittest.main()
