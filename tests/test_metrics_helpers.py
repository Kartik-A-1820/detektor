"""Tests for production-grade metrics helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from utils.metrics_helpers import (
    compute_ap50_95,
    compute_ap_from_pr_curve,
    compute_confusion_matrix,
    compute_dice_score,
    compute_mask_iou,
    compute_per_class_ap,
    compute_precision_recall_f1,
    threshold_sweep,
)


class TestMetricsHelpers(unittest.TestCase):
    """Test metrics computation helpers."""
    
    def test_precision_recall_f1_perfect(self):
        """Test perfect predictions."""
        precision, recall, f1 = compute_precision_recall_f1(tp=10, fp=0, fn=0)
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)
    
    def test_precision_recall_f1_no_predictions(self):
        """Test with no predictions."""
        precision, recall, f1 = compute_precision_recall_f1(tp=0, fp=0, fn=10)
        self.assertAlmostEqual(precision, 0.0)
        self.assertAlmostEqual(recall, 0.0)
        self.assertAlmostEqual(f1, 0.0)
    
    def test_precision_recall_f1_mixed(self):
        """Test with mixed results."""
        precision, recall, f1 = compute_precision_recall_f1(tp=7, fp=3, fn=2)
        self.assertAlmostEqual(precision, 0.7)
        self.assertAlmostEqual(recall, 7/9)
        expected_f1 = 2 * (0.7 * (7/9)) / (0.7 + 7/9)
        self.assertAlmostEqual(f1, expected_f1, places=5)
    
    def test_ap_from_pr_curve_empty(self):
        """Test AP computation with empty curve."""
        ap = compute_ap_from_pr_curve([], [])
        self.assertEqual(ap, 0.0)
    
    def test_ap_from_pr_curve_perfect(self):
        """Test AP computation with perfect precision."""
        precisions = [1.0, 1.0, 1.0]
        recalls = [0.33, 0.67, 1.0]
        ap = compute_ap_from_pr_curve(precisions, recalls, method="continuous")
        self.assertAlmostEqual(ap, 1.0)
    
    def test_ap_from_pr_curve_interp(self):
        """Test AP computation with interpolation."""
        precisions = [1.0, 0.8, 0.6, 0.5]
        recalls = [0.2, 0.4, 0.6, 0.8]
        ap = compute_ap_from_pr_curve(precisions, recalls, method="interp")
        self.assertGreater(ap, 0.0)
        self.assertLessEqual(ap, 1.0)
    
    def test_mask_iou_perfect(self):
        """Test perfect mask IoU."""
        mask = torch.ones((10, 10), dtype=torch.bool)
        iou = compute_mask_iou(mask, mask)
        self.assertAlmostEqual(iou, 1.0)
    
    def test_mask_iou_no_overlap(self):
        """Test masks with no overlap."""
        mask1 = torch.zeros((10, 10), dtype=torch.bool)
        mask1[:5, :5] = True
        mask2 = torch.zeros((10, 10), dtype=torch.bool)
        mask2[5:, 5:] = True
        iou = compute_mask_iou(mask1, mask2)
        self.assertAlmostEqual(iou, 0.0)
    
    def test_mask_iou_partial_overlap(self):
        """Test masks with partial overlap."""
        mask1 = torch.zeros((10, 10), dtype=torch.bool)
        mask1[:6, :6] = True  # 36 pixels
        mask2 = torch.zeros((10, 10), dtype=torch.bool)
        mask2[4:, 4:] = True  # 36 pixels
        # Intersection: 2x2 = 4 pixels
        # Union: 36 + 36 - 4 = 68 pixels
        iou = compute_mask_iou(mask1, mask2)
        expected_iou = 4.0 / 68.0
        self.assertAlmostEqual(iou, expected_iou, places=5)
    
    def test_mask_iou_empty(self):
        """Test empty masks."""
        mask1 = torch.zeros((10, 10), dtype=torch.bool)
        mask2 = torch.zeros((10, 10), dtype=torch.bool)
        iou = compute_mask_iou(mask1, mask2)
        self.assertEqual(iou, 0.0)
    
    def test_dice_score_perfect(self):
        """Test perfect Dice score."""
        mask = torch.ones((10, 10), dtype=torch.bool)
        dice = compute_dice_score(mask, mask)
        self.assertAlmostEqual(dice, 1.0)
    
    def test_dice_score_no_overlap(self):
        """Test Dice with no overlap."""
        mask1 = torch.zeros((10, 10), dtype=torch.bool)
        mask1[:5, :5] = True
        mask2 = torch.zeros((10, 10), dtype=torch.bool)
        mask2[5:, 5:] = True
        dice = compute_dice_score(mask1, mask2)
        self.assertAlmostEqual(dice, 0.0)
    
    def test_dice_score_partial_overlap(self):
        """Test Dice with partial overlap."""
        mask1 = torch.zeros((10, 10), dtype=torch.bool)
        mask1[:6, :6] = True  # 36 pixels
        mask2 = torch.zeros((10, 10), dtype=torch.bool)
        mask2[4:, 4:] = True  # 36 pixels
        # Intersection: 2x2 = 4 pixels
        dice = compute_dice_score(mask1, mask2)
        expected_dice = (2 * 4) / (36 + 36)
        self.assertAlmostEqual(dice, expected_dice, places=5)
    
    def test_confusion_matrix_simple(self):
        """Test simple confusion matrix."""
        pred_labels = torch.tensor([0, 1, 0, 1])
        gt_labels = torch.tensor([0, 1, 1, 1])
        matched_indices = torch.tensor([0, 1, -1, 2])  # 3rd pred is FP
        
        cm = compute_confusion_matrix(pred_labels, gt_labels, num_classes=2, matched_indices=matched_indices)
        
        # Check shape (num_classes + 1 for background)
        self.assertEqual(cm.shape, (3, 3))
        
        # Check correct matches
        self.assertEqual(cm[0, 0], 1)  # class 0 -> class 0
        self.assertEqual(cm[1, 1], 2)  # class 1 -> class 1
    
    def test_confusion_matrix_no_matches(self):
        """Test confusion matrix with no matches."""
        pred_labels = torch.tensor([0, 1])
        gt_labels = torch.tensor([0, 1])
        matched_indices = torch.tensor([-1, -1])  # All FP
        
        cm = compute_confusion_matrix(pred_labels, gt_labels, num_classes=2, matched_indices=matched_indices)
        
        # All predictions should be in background row (FP)
        self.assertEqual(cm[2, 0], 1)
        self.assertEqual(cm[2, 1], 1)
    
    def test_threshold_sweep_basic(self):
        """Test threshold sweep."""
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        tp_flags = [1, 1, 0, 1, 0, 0, 1, 0]
        total_gt = 5
        
        result = threshold_sweep(scores, tp_flags, total_gt, thresholds=[0.5, 0.7])
        
        self.assertIn("sweep", result)
        self.assertIn("best_threshold", result)
        self.assertIn("best_f1", result)
        self.assertEqual(len(result["sweep"]), 2)
        
        # Check that F1 is computed
        for sweep_point in result["sweep"]:
            self.assertIn("f1", sweep_point)
            self.assertIn("precision", sweep_point)
            self.assertIn("recall", sweep_point)
    
    def test_threshold_sweep_empty(self):
        """Test threshold sweep with no predictions."""
        result = threshold_sweep([], [], total_gt=10)
        
        self.assertIn("sweep", result)
        self.assertEqual(result["best_f1"], 0.0)
    
    def test_ap50_95_empty_predictions(self):
        """Test AP50-95 with empty predictions."""
        pred_boxes = torch.zeros((0, 4))
        pred_scores = torch.zeros((0,))
        pred_labels = torch.zeros((0,), dtype=torch.long)
        gt_boxes = torch.tensor([[10, 10, 20, 20]])
        gt_labels = torch.tensor([0])
        
        ap = compute_ap50_95(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
        self.assertEqual(ap, 0.0)
    
    def test_ap50_95_empty_gt(self):
        """Test AP50-95 with empty ground truth."""
        pred_boxes = torch.tensor([[10, 10, 20, 20]])
        pred_scores = torch.tensor([0.9])
        pred_labels = torch.tensor([0])
        gt_boxes = torch.zeros((0, 4))
        gt_labels = torch.zeros((0,), dtype=torch.long)
        
        ap = compute_ap50_95(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
        self.assertEqual(ap, 0.0)
    
    def test_ap50_95_perfect_match(self):
        """Test AP50-95 with perfect match."""
        pred_boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
        pred_scores = torch.tensor([0.95])
        pred_labels = torch.tensor([0])
        gt_boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
        gt_labels = torch.tensor([0])
        
        ap = compute_ap50_95(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
        self.assertGreater(ap, 0.9)  # Should be very high for perfect match
    
    def test_per_class_ap_single_class(self):
        """Test per-class AP with single class."""
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
                "labels": torch.tensor([0]),
            }
        ]
        
        per_class_ap = compute_per_class_ap(predictions, targets, num_classes=1)
        
        self.assertIn(0, per_class_ap)
        self.assertGreater(per_class_ap[0], 0.0)
        self.assertLessEqual(per_class_ap[0], 1.0)
    
    def test_per_class_ap_multi_class(self):
        """Test per-class AP with multiple classes."""
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        
        per_class_ap = compute_per_class_ap(predictions, targets, num_classes=2)
        
        self.assertEqual(len(per_class_ap), 2)
        self.assertIn(0, per_class_ap)
        self.assertIn(1, per_class_ap)
    
    def test_per_class_ap_empty_class(self):
        """Test per-class AP with class that has no instances."""
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
                "labels": torch.tensor([0]),
            }
        ]
        
        per_class_ap = compute_per_class_ap(predictions, targets, num_classes=3)
        
        # Classes 1 and 2 should have AP of 0
        self.assertEqual(per_class_ap[1], 0.0)
        self.assertEqual(per_class_ap[2], 0.0)


if __name__ == "__main__":
    unittest.main()
