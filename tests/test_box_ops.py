"""Unit tests for box operations helpers."""

from __future__ import annotations

import unittest

import torch

from utils.box_ops import (
    distances_to_boxes,
    flatten_prediction_levels,
    flatten_prediction_map,
)


class TestBoxOps(unittest.TestCase):
    """Unit tests for box decoding and flattening operations."""

    def test_distances_to_boxes_basic(self) -> None:
        """Test basic box decoding from distances."""
        points = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
        distances = torch.tensor([[[5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 3.0]]])
        
        boxes = distances_to_boxes(points, distances)
        
        self.assertEqual(boxes.shape, (1, 2, 4))
        expected = torch.tensor([[[5.0, 5.0, 15.0, 15.0], [17.0, 17.0, 23.0, 23.0]]])
        torch.testing.assert_close(boxes, expected)

    def test_distances_to_boxes_negative_distances(self) -> None:
        """Test that negative distances are handled via ReLU."""
        points = torch.tensor([[10.0, 10.0]])
        distances = torch.tensor([[[-5.0, -3.0, 2.0, 4.0]]])
        
        boxes = distances_to_boxes(points, distances)
        
        # Negative distances should be clamped to 0 by ReLU
        expected = torch.tensor([[[10.0, 10.0, 12.0, 14.0]]])
        torch.testing.assert_close(boxes, expected)

    def test_distances_to_boxes_nan_handling(self) -> None:
        """Test that NaN values are sanitized."""
        points = torch.tensor([[10.0, 10.0]])
        distances = torch.tensor([[[float('nan'), 5.0, 5.0, 5.0]]])
        
        boxes = distances_to_boxes(points, distances)
        
        # NaN should be replaced with 0.0
        self.assertFalse(torch.isnan(boxes).any())
        expected = torch.tensor([[[10.0, 5.0, 15.0, 15.0]]])
        torch.testing.assert_close(boxes, expected)

    def test_distances_to_boxes_large_values(self) -> None:
        """Test that very large distances are clamped."""
        points = torch.tensor([[10.0, 10.0]])
        distances = torch.tensor([[[5000.0, 5.0, 5.0, 5.0]]])
        
        boxes = distances_to_boxes(points, distances)
        
        # Large values should be clamped to 1000.0
        self.assertTrue((boxes.abs() <= 1000.0).all())

    def test_distances_to_boxes_shape_validation(self) -> None:
        """Test that invalid shapes raise errors."""
        points = torch.tensor([[10.0, 10.0]])
        
        # Wrong distances shape
        with self.assertRaises(AssertionError):
            distances_to_boxes(points, torch.tensor([[5.0, 5.0]]))
        
        # Wrong points shape
        with self.assertRaises(AssertionError):
            distances_to_boxes(torch.tensor([10.0, 10.0]), torch.tensor([[[5.0, 5.0, 5.0, 5.0]]]))

    def test_distances_to_boxes_batch(self) -> None:
        """Test batch processing."""
        points = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
        distances = torch.tensor([
            [[5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 3.0]],
            [[2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0]],
        ])
        
        boxes = distances_to_boxes(points, distances)
        
        self.assertEqual(boxes.shape, (2, 2, 4))
        self.assertFalse(torch.isnan(boxes).any())

    def test_flatten_prediction_map(self) -> None:
        """Test flattening of prediction maps."""
        pred = torch.randn(2, 4, 8, 8)  # [B, C, H, W]
        
        flattened = flatten_prediction_map(pred)
        
        self.assertEqual(flattened.shape, (2, 64, 4))  # [B, H*W, C]

    def test_flatten_prediction_map_invalid_shape(self) -> None:
        """Test that invalid shapes raise errors."""
        pred = torch.randn(2, 4, 8)  # Wrong ndim
        
        with self.assertRaises(AssertionError):
            flatten_prediction_map(pred)

    def test_flatten_prediction_levels(self) -> None:
        """Test flattening and concatenation of multi-level predictions."""
        level1 = torch.randn(2, 4, 8, 8)
        level2 = torch.randn(2, 4, 4, 4)
        level3 = torch.randn(2, 4, 2, 2)
        
        flattened = flatten_prediction_levels([level1, level2, level3])
        
        # Total points: 8*8 + 4*4 + 2*2 = 64 + 16 + 4 = 84
        self.assertEqual(flattened.shape, (2, 84, 4))


if __name__ == "__main__":
    unittest.main()
