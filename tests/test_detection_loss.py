from __future__ import annotations

import unittest

import torch

from losses.detection import CenterPriorAssigner


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


if __name__ == "__main__":
    unittest.main()
