from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.robust_loss import sanitize_tensor, clamp_loss


EPS = 1e-6


class SegmentationLoss(nn.Module):
    """Prototype-mask segmentation loss with BCE and Dice terms on positives only."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred_masks: Tensor, target_masks: Tensor) -> Dict[str, Tensor]:
        """Compute BCE-with-logits and Dice losses for aligned mask tensors.

        Args:
            pred_masks: ``[N, Hp, Wp]`` raw logits.
            target_masks: ``[N, Hp, Wp]`` binary targets in ``{0, 1}``.
        """
        if pred_masks.shape != target_masks.shape:
            raise AssertionError(
                f"Predicted masks and target masks must share shape, got {tuple(pred_masks.shape)} vs {tuple(target_masks.shape)}"
            )

        if pred_masks.numel() == 0:
            zero = pred_masks.sum() * 0.0
            return {
                "loss_mask_bce": zero,
                "loss_mask_dice": zero,
                "loss_mask": zero,
            }

        target_masks = target_masks.to(dtype=pred_masks.dtype)
        
        # Clamp predictions to prevent extreme values
        pred_masks_clamped = pred_masks.clamp(min=-100, max=100)
        
        # Compute BCE loss with sanitization
        loss_bce_raw = F.binary_cross_entropy_with_logits(pred_masks_clamped, target_masks, reduction="mean")
        loss_bce = sanitize_tensor(loss_bce_raw, name="mask_bce")
        loss_bce = clamp_loss(loss_bce, max_value=10.0, name="mask_bce")

        # Compute Dice loss with sanitization
        pred_probs = pred_masks_clamped.sigmoid().reshape(pred_masks.shape[0], -1)
        target_flat = target_masks.reshape(target_masks.shape[0], -1)
        intersection = (pred_probs * target_flat).sum(dim=1)
        denom = pred_probs.sum(dim=1) + target_flat.sum(dim=1)
        
        # Robust Dice computation
        dice_scores = (2.0 * intersection + EPS) / (denom + EPS)
        dice_scores = dice_scores.clamp(min=0.0, max=1.0)
        loss_dice_raw = 1.0 - dice_scores
        loss_dice = sanitize_tensor(loss_dice_raw.mean(), name="mask_dice")
        loss_dice = clamp_loss(loss_dice, max_value=1.0, name="mask_dice")

        # Compute total mask loss with sanitization
        loss_mask = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        loss_mask = sanitize_tensor(loss_mask, name="mask_total")
        loss_mask = clamp_loss(loss_mask, max_value=20.0, name="mask_total")
        
        return {
            "loss_mask_bce": loss_bce,
            "loss_mask_dice": loss_dice,
            "loss_mask": loss_mask,
        }
