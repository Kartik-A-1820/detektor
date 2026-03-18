from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.phase3_diagnostics import build_phase3_diagnostics
from tests import get_test_tmp_root


class TestPhase3Diagnostics(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = get_test_tmp_root()
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_path = temp_root / "phase3_diagnostics"
        self.temp_path.mkdir(parents=True, exist_ok=True)

    def test_reports_zero_assignment_for_sub_stride_ball_boxes(self) -> None:
        dataset_root = self.temp_path / "toy_dataset"
        for split in ("train", "val"):
            (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_root / "train" / "images" / "img_0.jpg"), image)
        cv2.imwrite(str(dataset_root / "val" / "images" / "img_0.jpg"), image)
        # 0.01 * 512 ~= 5.12 px wide/high, which is too small to reliably contain a stride-8 point center.
        (dataset_root / "train" / "labels" / "img_0.txt").write_text("0 0.5 0.5 0.01 0.01\n", encoding="utf-8")
        (dataset_root / "val" / "labels" / "img_0.txt").write_text("0 0.5 0.5 0.01 0.01\n", encoding="utf-8")

        data_yaml = self.temp_path / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "train": str(dataset_root / "train" / "images"),
                    "val": str(dataset_root / "val" / "images"),
                    "nc": 1,
                    "names": ["ball"],
                }
            ),
            encoding="utf-8",
        )

        diagnostics = build_phase3_diagnostics(data_yaml, img_size=512, center_radius=2.5)

        self.assertEqual(diagnostics["train"]["classes"]["ball"]["count"], 1)
        self.assertEqual(diagnostics["train"]["classes"]["ball"]["zero_assignment_gt"], 1)
        self.assertEqual(diagnostics["val"]["classes"]["ball"]["zero_assignment_gt"], 1)


if __name__ == "__main__":
    unittest.main()
