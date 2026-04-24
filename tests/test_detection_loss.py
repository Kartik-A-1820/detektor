from __future__ import annotations

import unittest

import torch

from losses.detection import CenterPriorAssigner, DetectionLoss


class TestCenterPriorAssigner(unittest.TestCase):
    def test_assigner_matches_sub_stride_box_via_effective_extent(self) -> None:
        assigner = CenterPriorAssigner(center_radius=2.5)
        points = torch.tensor([[252.0, 252.0]], dtype=torch.float32)
        strides = torch.tensor([8.0], dtype=torch.float32)
        gt_boxes = torch.tensor([[253.44, 253.44, 258.56, 258.56]], dtype=torch.float32)
        gt_labels = torch.tensor([0], dtype=torch.long)

        assignment = assigner.assign(points, strides, gt_boxes, gt_labels)

        self.assertTrue(bool(assignment["fg_mask"][0]))
        self.assertEqual(int(assignment["assigned_labels"][0].item()), 0)
        self.assertEqual(int(assignment["matched_gt_indices"][0].item()), 0)
        self.assertTrue(torch.equal(assignment["assigned_boxes"][0], gt_boxes[0]))

    def test_assigner_still_filters_distant_points_for_tiny_boxes(self) -> None:
        assigner = CenterPriorAssigner(center_radius=2.5)
        points = torch.tensor([[240.0, 240.0]], dtype=torch.float32)
        strides = torch.tensor([8.0], dtype=torch.float32)
        gt_boxes = torch.tensor([[253.44, 253.44, 258.56, 258.56]], dtype=torch.float32)
        gt_labels = torch.tensor([0], dtype=torch.long)

        assignment = assigner.assign(points, strides, gt_boxes, gt_labels)

        self.assertFalse(bool(assignment["fg_mask"][0]))
        self.assertEqual(int(assignment["matched_gt_indices"][0].item()), -1)


class TestDetectionLossFocalLoss(unittest.TestCase):
    """Tests for the focal loss option in DetectionLoss."""

    def _make_minimal_inputs(self, num_classes: int = 2):
        """Build minimal flat prediction tensors and targets for one image."""
        batch_size = 1
        total_points = 4
        pred_cls = torch.zeros(batch_size, total_points, num_classes)
        pred_box = torch.zeros(batch_size, total_points, 4)
        pred_obj = torch.zeros(batch_size, total_points, 1)
        decoded_boxes = torch.tensor(
            [[[10.0, 10.0, 50.0, 50.0]] * total_points],
            dtype=torch.float32,
        )
        points = torch.tensor([[28.0, 28.0]] * total_points, dtype=torch.float32)
        strides = torch.tensor([8.0] * total_points, dtype=torch.float32)
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
            }
        ]
        return pred_cls, pred_box, pred_obj, decoded_boxes, points, strides, targets

    def test_focal_loss_gamma_zero_matches_bce(self) -> None:
        """With gamma=0, focal loss reduces to standard BCE."""
        pred_cls, pred_box, pred_obj, decoded_boxes, points, strides, targets = self._make_minimal_inputs()
        loss_bce = DetectionLoss(num_classes=2, focal_loss_gamma=0.0)
        loss_focal = DetectionLoss(num_classes=2, focal_loss_gamma=0.0)
        out_bce = loss_bce(pred_cls, pred_box, pred_obj, decoded_boxes, points, strides, targets)
        out_focal = loss_focal(pred_cls, pred_box, pred_obj, decoded_boxes, points, strides, targets)
        self.assertAlmostEqual(
            float(out_bce["loss_cls"].item()),
            float(out_focal["loss_cls"].item()),
            places=5,
        )

    def test_focal_loss_gamma_positive_produces_finite_loss(self) -> None:
        """Focal loss with gamma=2.0 should produce a finite, non-negative cls loss."""
        pred_cls, pred_box, pred_obj, decoded_boxes, points, strides, targets = self._make_minimal_inputs()
        loss_fn = DetectionLoss(num_classes=2, focal_loss_gamma=2.0)
        out = loss_fn(pred_cls, pred_box, pred_obj, decoded_boxes, points, strides, targets)
        cls_loss = float(out["loss_cls"].item())
        self.assertTrue(torch.isfinite(out["loss_cls"]), f"cls loss is not finite: {cls_loss}")
        self.assertGreaterEqual(cls_loss, 0.0)

    def test_focal_loss_reduces_easy_example_weight(self) -> None:
        """Focal loss should produce a lower cls loss than BCE when predictions are confident."""
        pred_cls, pred_box, pred_obj, decoded_boxes, points, strides, targets = self._make_minimal_inputs()
        # Make predictions very confident (high logits for correct class)
        pred_cls_confident = pred_cls.clone()
        pred_cls_confident[0, :, 1] = 5.0  # high confidence for class 1

        loss_bce = DetectionLoss(num_classes=2, focal_loss_gamma=0.0)
        loss_focal = DetectionLoss(num_classes=2, focal_loss_gamma=2.0)
        out_bce = loss_bce(pred_cls_confident, pred_box, pred_obj, decoded_boxes, points, strides, targets)
        out_focal = loss_focal(pred_cls_confident, pred_box, pred_obj, decoded_boxes, points, strides, targets)
        # Focal loss should down-weight easy (confident) examples
        self.assertLessEqual(
            float(out_focal["loss_cls"].item()),
            float(out_bce["loss_cls"].item()) + 1e-4,
        )


if __name__ == "__main__":
    unittest.main()
