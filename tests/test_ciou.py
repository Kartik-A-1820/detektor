"""Unit tests for CIoU loss computation."""

from __future__ import annotations

import unittest

import torch

from utils.box_ops import box_area, box_iou, ciou_loss


class TestCIoU(unittest.TestCase):
    """Unit tests for CIoU loss and IoU helpers."""

    def test_box_area_basic(self) -> None:
        """Test basic box area computation."""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
        areas = box_area(boxes)
        
        expected = torch.tensor([100.0, 100.0])
        torch.testing.assert_close(areas, expected)

    def test_box_area_invalid_boxes(self) -> None:
        """Test that invalid boxes (x2 < x1) are handled."""
        boxes = torch.tensor([[10.0, 10.0, 5.0, 5.0]])  # Invalid box
        areas = box_area(boxes)
        
        # Should clamp to 0
        self.assertEqual(areas.item(), 0.0)

    def test_box_iou_identical_boxes(self) -> None:
        """Test IoU of identical boxes."""
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        
        iou = box_iou(boxes1, boxes2)
        
        torch.testing.assert_close(iou, torch.tensor([[1.0]]))

    def test_box_iou_no_overlap(self) -> None:
        """Test IoU of non-overlapping boxes."""
        boxes1 = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
        boxes2 = torch.tensor([[10.0, 10.0, 15.0, 15.0]])
        
        iou = box_iou(boxes1, boxes2)
        
        torch.testing.assert_close(iou, torch.tensor([[0.0]]))

    def test_box_iou_partial_overlap(self) -> None:
        """Test IoU of partially overlapping boxes."""
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        
        iou = box_iou(boxes1, boxes2)
        
        # Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        expected_iou = 25.0 / 175.0
        torch.testing.assert_close(iou, torch.tensor([[expected_iou]]), atol=1e-5, rtol=1e-5)

    def test_ciou_loss_identical_boxes(self) -> None:
        """Test CIoU loss for identical boxes."""
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        target = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        
        loss = ciou_loss(pred, target)
        
        # Perfect match should give loss close to 0
        self.assertLess(loss.item(), 0.01)

    def test_ciou_loss_no_overlap(self) -> None:
        """Test CIoU loss for non-overlapping boxes."""
        pred = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
        target = torch.tensor([[20.0, 20.0, 25.0, 25.0]])
        
        loss = ciou_loss(pred, target)
        
        # No overlap should give high loss
        self.assertGreater(loss.item(), 1.0)

    def test_ciou_loss_empty_input(self) -> None:
        """Test CIoU loss with empty input."""
        pred = torch.empty((0, 4))
        target = torch.empty((0, 4))
        
        loss = ciou_loss(pred, target)
        
        self.assertEqual(loss.shape, (0,))

    def test_ciou_loss_invalid_boxes(self) -> None:
        """Test CIoU loss handles invalid box coordinates."""
        pred = torch.tensor([[10.0, 10.0, 5.0, 5.0]])  # x2 < x1
        target = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        
        loss = ciou_loss(pred, target)
        
        # Should not produce NaN
        self.assertFalse(torch.isnan(loss).any())
        self.assertTrue((loss >= 0.0).all())
        self.assertTrue((loss <= 2.0).all())

    def test_ciou_loss_batch(self) -> None:
        """Test CIoU loss with batch of boxes."""
        pred = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 15.0],
            [20.0, 20.0, 30.0, 30.0],
        ])
        target = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [6.0, 6.0, 16.0, 16.0],
            [25.0, 25.0, 35.0, 35.0],
        ])
        
        loss = ciou_loss(pred, target)
        
        self.assertEqual(loss.shape, (3,))
        self.assertFalse(torch.isnan(loss).any())
        self.assertTrue((loss >= 0.0).all())
        self.assertTrue((loss <= 2.0).all())

    def test_ciou_loss_aspect_ratio_penalty(self) -> None:
        """Test that CIoU penalizes aspect ratio differences."""
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])  # Square
        target = torch.tensor([[0.0, 0.0, 10.0, 5.0]])  # Rectangle
        
        loss = ciou_loss(pred, target)
        
        # Should have some loss due to aspect ratio difference
        self.assertGreater(loss.item(), 0.1)

    def test_ciou_loss_no_nan_on_extreme_values(self) -> None:
        """Test CIoU loss doesn't produce NaN with extreme values."""
        pred = torch.tensor([[0.0, 0.0, 1000.0, 1000.0]])
        target = torch.tensor([[500.0, 500.0, 1500.0, 1500.0]])
        
        loss = ciou_loss(pred, target)
        
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())


if __name__ == "__main__":
    unittest.main()
