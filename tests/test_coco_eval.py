"""COCO-style evaluation: semantics pinned against hand-computed values and the legacy helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from metrics.coco_eval import coco_evaluate
from utils.metrics_helpers import compute_ap50_95, compute_per_class_ap


def _t(boxes, labels):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), "labels": torch.tensor(labels, dtype=torch.long)}


def _p(boxes, scores, labels):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.long)}


class CocoEvalTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self) -> None:
        targets = [_t([[10, 10, 50, 60]], [0]), _t([[5, 5, 100, 100], [0, 0, 20, 20]], [1, 0])]
        preds = [_p(t["boxes"].tolist(), [0.9] * len(t["labels"]), t["labels"].tolist()) for t in targets]
        r = coco_evaluate(preds, targets, num_classes=2)
        self.assertAlmostEqual(r["map50"], 1.0, places=6)
        self.assertAlmostEqual(r["map50_95"], 1.0, places=6)
        self.assertEqual(r["num_gt"], 3)

    def test_hand_computed_iou_thresholds(self) -> None:
        # One 100x100 GT, one detection shifted so IoU = 0.8 (a 20 px shift: 80*100 / 120*100 -> 0.667?) -> use width
        # box [0,0,80,100] vs GT [0,0,100,100]: IoU = 0.8 exactly.
        r = coco_evaluate([_p([[0, 0, 80, 100]], [0.9], [0])], [_t([[0, 0, 100, 100]], [0])], num_classes=1)
        # TP at IoU thresholds 0.50, 0.55, ..., 0.80 -> 7 of the 10 COCO thresholds
        self.assertAlmostEqual(r["map50_95"], 0.7, places=3)
        self.assertAlmostEqual(r["map50"], 1.0, places=3)
        self.assertAlmostEqual(r["map75"], 1.0, places=3)

    def test_false_positives_on_empty_images_are_penalised(self) -> None:
        """Pooled COCO AP counts a confident false positive on a no-GT image; per-image averaging skips it."""
        targets = [_t([[10, 10, 60, 60]], [0]), _t([], [])]
        preds = [_p([[10, 10, 60, 60]], [0.90], [0]), _p([[100, 100, 150, 150]], [0.95], [0])]
        coco = coco_evaluate(preds, targets, 1)
        self.assertAlmostEqual(coco["map50"], 0.5, places=2)  # precision 0.5 at full recall
        legacy = np.mean([compute_ap50_95(p["boxes"], p["scores"], p["labels"], t["boxes"], t["labels"])
                          for p, t in zip(preds, targets) if p["boxes"].numel() and t["boxes"].numel()])
        self.assertGreater(legacy, 0.95)  # the legacy per-image figure ignores the empty image entirely
        self.assertLess(coco["map50_95"], legacy)

    def test_size_buckets(self) -> None:
        targets = [_t([[0, 0, 20, 20]], [0]), _t([[0, 0, 200, 200]], [0])]
        preds = [_p(t["boxes"].tolist(), [0.9], [0]) for t in targets]
        r = coco_evaluate(preds, targets, 1)
        self.assertAlmostEqual(r["ap_small"], 1.0, places=3)
        self.assertAlmostEqual(r["ap_large"], 1.0, places=3)
        self.assertIsNone(r["ap_medium"])  # no medium objects -> "no data", not 0

    def test_per_class_curves_and_classes_without_gt(self) -> None:
        targets = [_t([[0, 0, 50, 50]], [0])]
        preds = [_p([[0, 0, 50, 50]], [0.9], [0])]
        r = coco_evaluate(preds, targets, num_classes=3)
        self.assertAlmostEqual(r["per_class_ap50"][0], 1.0, places=3)
        self.assertEqual(r["per_class_ap50"][1], 0.0)  # class with no GT reports 0, does not crash
        self.assertEqual(len(r["recall_thresholds"]), 101)
        self.assertEqual(len(r["precision_at_iou50"][0]), 101)

    def test_degenerate_inputs(self) -> None:
        empty = coco_evaluate([_p([], [], [])], [_t([], [])], 2)
        self.assertEqual((empty["map50"], empty["num_gt"]), (0.0, 0))
        no_preds = coco_evaluate([_p([], [], [])], [_t([[0, 0, 10, 10]], [0])], 1)
        self.assertEqual(no_preds["map50_95"], 0.0)
        nan_box = coco_evaluate([_p([[float("nan")] * 4, [0, 0, 10, 10]], [0.9, 0.8], [0, 0])], [_t([[0, 0, 10, 10]], [0])], 1)
        self.assertAlmostEqual(nan_box["map50"], 1.0, places=3)  # the NaN detection is ignored
        with self.assertRaises(ValueError):
            coco_evaluate([], [_t([], [])], 1)

    def test_legacy_ap50_stays_close_to_the_reference(self) -> None:
        """Guards the older per-class AP50 helper against drifting from pycocotools."""
        rng = np.random.default_rng(0)
        targets, preds = [], []
        for _ in range(40):
            n = int(rng.integers(1, 5))
            xy = rng.uniform(0, 150, (n, 2))
            wh = rng.uniform(30, 90, (n, 2))
            boxes = np.concatenate([xy, xy + wh], axis=1)
            labels = rng.integers(0, 3, n)
            targets.append(_t(boxes.tolist(), labels.tolist()))
            jitter = rng.normal(0, 4, boxes.shape)
            keep = rng.random(n) > 0.2
            fp = rng.uniform(0, 200, (2, 4))
            fp[:, 2:] = fp[:, :2] + 40
            pb = np.concatenate([(boxes + jitter)[keep], fp])
            ps = np.concatenate([rng.uniform(0.5, 1.0, keep.sum()), rng.uniform(0.05, 0.6, 2)])
            pl = np.concatenate([labels[keep], rng.integers(0, 3, 2)])
            preds.append(_p(pb.tolist(), ps.tolist(), pl.tolist()))
        coco = coco_evaluate(preds, targets, 3)["map50"]
        legacy = float(np.mean(list(compute_per_class_ap(preds, targets, 3).values())))
        self.assertLess(abs(coco - legacy), 0.05, (coco, legacy))


if __name__ == "__main__":
    unittest.main()
