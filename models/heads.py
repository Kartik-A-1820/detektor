from __future__ import annotations

from typing import Dict, List, Sequence

from torch import Tensor, nn

from .blocks import ConvBNAct


class DetectionHead(nn.Module):
    """Anchor-free multi-level detection head with mask coefficient prediction."""

    def __init__(
        self,
        num_classes: int,
        in_channels: Sequence[int] = (128, 192, 256),
        feat_channels: int = 128,
        num_mask_coeffs: int = 24,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_mask_coeffs = num_mask_coeffs

        self.cls_towers = nn.ModuleList([self._make_tower(ch, feat_channels) for ch in in_channels])
        self.box_towers = nn.ModuleList([self._make_tower(ch, feat_channels) for ch in in_channels])
        self.obj_towers = nn.ModuleList([self._make_tower(ch, feat_channels) for ch in in_channels])
        self.mask_towers = nn.ModuleList([self._make_tower(ch, feat_channels) for ch in in_channels])

        self.cls_preds = nn.ModuleList([nn.Conv2d(feat_channels, num_classes, kernel_size=1) for _ in in_channels])
        self.box_preds = nn.ModuleList([nn.Conv2d(feat_channels, 4, kernel_size=1) for _ in in_channels])
        self.obj_preds = nn.ModuleList([nn.Conv2d(feat_channels, 1, kernel_size=1) for _ in in_channels])
        self.mask_preds = nn.ModuleList(
            [nn.Conv2d(feat_channels, num_mask_coeffs, kernel_size=1) for _ in in_channels]
        )
        
        # Initialize weights (Ultralytics-style)
        self._initialize_weights()

    def _make_tower(self, in_channels: int, feat_channels: int) -> nn.Sequential:
        return nn.Sequential(
            ConvBNAct(in_channels, feat_channels, kernel_size=3, stride=1),
            ConvBNAct(feat_channels, feat_channels, kernel_size=3, stride=1),
        )
    
    def _initialize_weights(self) -> None:
        """Initialize weights to prevent NaN in early training (Ultralytics-style)."""
        import math
        
        # Initialize prediction heads with small values
        for module_list in [self.cls_preds, self.box_preds, self.obj_preds, self.mask_preds]:
            for m in module_list:
                if isinstance(m, nn.Conv2d):
                    # Use normal initialization with small std
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        
        # Special initialization for classification head (reduce initial confidence)
        # This prevents extreme logits early in training
        for m in self.cls_preds:
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                # Initialize bias to log((1 - prior) / prior) where prior = 0.01
                # This gives initial probability of ~0.01 for each class
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(m.bias, bias_value)

    def forward(self, features: Sequence[Tensor]) -> Dict[str, List[Tensor]]:
        """Return raw per-level predictions for classification, box, objectness, and mask coefficients."""
        if len(features) != 3:
            raise AssertionError(f"Expected exactly 3 neck feature maps, got {len(features)}")

        cls_outputs: List[Tensor] = []
        box_outputs: List[Tensor] = []
        obj_outputs: List[Tensor] = []
        mask_outputs: List[Tensor] = []

        for index, feature in enumerate(features):
            cls_feature = self.cls_towers[index](feature)
            box_feature = self.box_towers[index](feature)
            obj_feature = self.obj_towers[index](feature)
            mask_feature = self.mask_towers[index](feature)

            cls_outputs.append(self.cls_preds[index](cls_feature))
            box_outputs.append(self.box_preds[index](box_feature))
            obj_outputs.append(self.obj_preds[index](obj_feature))
            mask_outputs.append(self.mask_preds[index](mask_feature))

        return {
            "cls": cls_outputs,
            "box": box_outputs,
            "obj": obj_outputs,
            "mask_coeff": mask_outputs,
        }


class PrototypeMaskHead(nn.Module):
    """Prototype mask generator from the highest-resolution neck feature map."""

    def __init__(self, in_channels: int = 128, hidden_channels: int = 96, proto_k: int = 24) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=3, stride=1),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=1),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=1),
        )
        self.pred = nn.Conv2d(hidden_channels, proto_k, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Return prototype masks in [B, K, H, W] format."""
        return self.pred(self.stem(x))
