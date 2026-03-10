from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
        loss_bce = F.binary_cross_entropy_with_logits(pred_masks, target_masks, reduction="mean")

        pred_probs = pred_masks.sigmoid().reshape(pred_masks.shape[0], -1)
        target_flat = target_masks.reshape(target_masks.shape[0], -1)
        intersection = (pred_probs * target_flat).sum(dim=1)
        denom = pred_probs.sum(dim=1) + target_flat.sum(dim=1)
        loss_dice = 1.0 - ((2.0 * intersection + EPS) / (denom + EPS))
        loss_dice = loss_dice.mean()

        loss_mask = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        return {
            "loss_mask_bce": loss_bce,
            "loss_mask_dice": loss_dice,
            "loss_mask": loss_mask,
        }
