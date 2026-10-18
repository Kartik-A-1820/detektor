"""Class-aware vs class-agnostic NMS."""

from __future__ import annotations

import unittest

import torch

from utils.postprocess import class_aware_nms


def _two_classes_on_one_object():
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [11.0, 11.0, 51.0, 51.0], [200.0, 200.0, 240.0, 240.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([0, 1, 0])  # first two overlap almost perfectly but have different classes
    return boxes, scores, labels


class NmsTests(unittest.TestCase):
    def test_default_keeps_overlapping_boxes_of_different_classes(self) -> None:
        keep = class_aware_nms(*_two_classes_on_one_object(), iou_thresh=0.5, max_det=10)
        self.assertEqual(sorted(keep.tolist()), [0, 1, 2])

    def test_agnostic_suppresses_across_classes_keeping_the_best(self) -> None:
        keep = class_aware_nms(*_two_classes_on_one_object(), iou_thresh=0.5, max_det=10, agnostic=True)
        self.assertEqual(keep.tolist(), [0, 2])  # the 0.8 duplicate of class 1 is removed; order by score

    def test_agnostic_respects_max_det_and_empty_input(self) -> None:
        boxes, scores, labels = _two_classes_on_one_object()
        self.assertEqual(class_aware_nms(boxes, scores, labels, 0.5, max_det=1, agnostic=True).tolist(), [0])
        empty = class_aware_nms(torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long), 0.5, 5, agnostic=True)
        self.assertEqual(empty.numel(), 0)

    def test_pure_torch_fallback_matches(self) -> None:
        from unittest import mock

        import utils.postprocess as pp

        boxes, scores, labels = _two_classes_on_one_object()
        with mock.patch.object(pp, "torchvision_nms", None):
            self.assertEqual(pp.class_aware_nms(boxes, scores, labels, 0.5, 10, agnostic=True).tolist(), [0, 2])
            self.assertEqual(sorted(pp.class_aware_nms(boxes, scores, labels, 0.5, 10).tolist()), [0, 1, 2])

    def test_model_predict_accepts_agnostic_flag(self) -> None:
        from models.chimera import ChimeraODIS

        torch.manual_seed(0)
        model = ChimeraODIS(num_classes=3, proto_k=8).eval()
        x = torch.rand(1, 3, 96, 96)
        per_class = model.predict(x, conf_thresh=0.0, iou_thresh=0.3, task="detect")[0]["boxes"].shape[0]
        agnostic = model.predict(x, conf_thresh=0.0, iou_thresh=0.3, task="detect", agnostic_nms=True)[0]["boxes"].shape[0]
        self.assertLessEqual(agnostic, per_class)


if __name__ == "__main__":
    unittest.main()
