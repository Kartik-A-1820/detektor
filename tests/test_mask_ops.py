"""Unit tests for mask operations helpers."""

from __future__ import annotations

import unittest

import torch

from utils.mask_ops import (
    boxes_to_proto_coordinates,
    compose_instance_masks,
    crop_mask_region,
    flatten_mask_coefficients,
    resize_instance_masks,
    threshold_masks,
    upsample_masks_to_image,
)


class TestMaskOps(unittest.TestCase):
    """Unit tests for mask operations."""

    def test_flatten_mask_coefficients(self) -> None:
        """Test flattening of mask coefficients."""
        level1 = torch.randn(2, 24, 8, 8)  # [B, K, H, W]
        level2 = torch.randn(2, 24, 4, 4)
        
        flattened = flatten_mask_coefficients([level1, level2])
        
        # Total points: 8*8 + 4*4 = 64 + 16 = 80
        self.assertEqual(flattened.shape, (2, 80, 24))

    def test_flatten_mask_coefficients_empty(self) -> None:
        """Test that empty input raises error."""
        with self.assertRaises(AssertionError):
            flatten_mask_coefficients([])

    def test_compose_instance_masks(self) -> None:
        """Test composing instance masks from prototypes and coefficients."""
        prototypes = torch.randn(24, 32, 32)  # [K, Hp, Wp]
        coefficients = torch.randn(5, 24)  # [N, K]
        
        masks = compose_instance_masks(prototypes, coefficients)
        
        self.assertEqual(masks.shape, (5, 32, 32))

    def test_compose_instance_masks_empty(self) -> None:
        """Test composing with empty coefficients."""
        prototypes = torch.randn(24, 32, 32)
        coefficients = torch.empty((0, 24))
        
        masks = compose_instance_masks(prototypes, coefficients)
        
        self.assertEqual(masks.shape, (0, 32, 32))

    def test_resize_instance_masks(self) -> None:
        """Test resizing instance masks."""
        masks = torch.rand(3, 128, 128)
        proto_size = (32, 32)
        
        resized = resize_instance_masks(masks, proto_size)
        
        self.assertEqual(resized.shape, (3, 32, 32))

    def test_resize_instance_masks_empty(self) -> None:
        """Test resizing empty masks."""
        masks = torch.empty((0, 128, 128))
        proto_size = (32, 32)
        
        resized = resize_instance_masks(masks, proto_size)
        
        self.assertEqual(resized.shape, (0, 32, 32))

    def test_boxes_to_proto_coordinates(self) -> None:
        """Test converting boxes to prototype coordinates."""
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [50.0, 50.0, 150.0, 150.0]])
        image_size = (200, 200)
        proto_size = (32, 32)
        
        proto_boxes = boxes_to_proto_coordinates(boxes, image_size, proto_size)
        
        self.assertEqual(proto_boxes.shape, (2, 4))
        # First box should map to [0, 0, 16, 16] (100/200 * 32 = 16)
        expected_first = torch.tensor([0.0, 0.0, 16.0, 16.0])
        torch.testing.assert_close(proto_boxes[0], expected_first)

    def test_boxes_to_proto_coordinates_clamping(self) -> None:
        """Test that boxes are clamped to proto bounds."""
        boxes = torch.tensor([[-10.0, -10.0, 300.0, 300.0]])
        image_size = (200, 200)
        proto_size = (32, 32)
        
        proto_boxes = boxes_to_proto_coordinates(boxes, image_size, proto_size)
        
        # Should be clamped to [0, 0, 32, 32]
        self.assertTrue((proto_boxes >= 0.0).all())
        self.assertTrue((proto_boxes[:, 0::2] <= 32.0).all())
        self.assertTrue((proto_boxes[:, 1::2] <= 32.0).all())

    def test_boxes_to_proto_coordinates_empty(self) -> None:
        """Test converting empty boxes."""
        boxes = torch.empty((0, 4))
        image_size = (200, 200)
        proto_size = (32, 32)
        
        proto_boxes = boxes_to_proto_coordinates(boxes, image_size, proto_size)
        
        self.assertEqual(proto_boxes.shape, (0, 4))

    def test_crop_mask_region(self) -> None:
        """Test cropping mask regions."""
        masks = torch.ones(2, 32, 32)
        boxes = torch.tensor([[0.0, 0.0, 16.0, 16.0], [16.0, 16.0, 32.0, 32.0]])
        
        cropped = crop_mask_region(masks, boxes)
        
        self.assertEqual(cropped.shape, (2, 32, 32))
        # First mask should be non-zero only in top-left quadrant
        self.assertGreater(cropped[0, :16, :16].sum(), 0)
        self.assertEqual(cropped[0, 16:, 16:].sum(), 0)

    def test_crop_mask_region_empty(self) -> None:
        """Test cropping empty masks."""
        masks = torch.empty((0, 32, 32))
        boxes = torch.empty((0, 4))
        
        cropped = crop_mask_region(masks, boxes)
        
        self.assertEqual(cropped.shape, (0, 32, 32))

    def test_crop_mask_region_invalid_boxes(self) -> None:
        """Test that invalid boxes are handled gracefully."""
        masks = torch.ones(1, 32, 32)
        boxes = torch.tensor([[20.0, 20.0, 10.0, 10.0]])  # x2 < x1
        
        cropped = crop_mask_region(masks, boxes)
        
        # Should not crash and should produce valid output
        self.assertEqual(cropped.shape, (1, 32, 32))
        self.assertFalse(torch.isnan(cropped).any())

    def test_upsample_masks_to_image(self) -> None:
        """Test upsampling masks to image resolution."""
        masks = torch.rand(3, 32, 32)
        image_size = (128, 128)
        
        upsampled = upsample_masks_to_image(masks, image_size)
        
        self.assertEqual(upsampled.shape, (3, 128, 128))

    def test_upsample_masks_to_image_empty(self) -> None:
        """Test upsampling empty masks."""
        masks = torch.empty((0, 32, 32))
        image_size = (128, 128)
        
        upsampled = upsample_masks_to_image(masks, image_size)
        
        self.assertEqual(upsampled.shape, (0, 128, 128))

    def test_threshold_masks(self) -> None:
        """Test thresholding probabilistic masks."""
        mask_probs = torch.tensor([[0.3, 0.6], [0.8, 0.2]])
        
        binary = threshold_masks(mask_probs, threshold=0.5)
        
        expected = torch.tensor([[False, True], [True, False]])
        torch.testing.assert_close(binary, expected)

    def test_threshold_masks_empty(self) -> None:
        """Test thresholding empty masks."""
        mask_probs = torch.empty((0, 32, 32))
        
        binary = threshold_masks(mask_probs)
        
        self.assertEqual(binary.shape, (0, 32, 32))


if __name__ == "__main__":
    unittest.main()
