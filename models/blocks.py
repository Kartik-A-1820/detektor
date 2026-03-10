from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn


class ConvBNAct(nn.Module):
    """Standard convolution block with BatchNorm and SiLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution, normalization, and activation."""
        return self.act(self.bn(self.conv(x)))


class DWConvBNAct(nn.Module):
    """Depthwise separable convolution block for lightweight feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply depthwise followed by pointwise convolution."""
        return self.pointwise(self.depthwise(x))


class RepBlock(nn.Module):
    """Lightweight residual block using standard and depthwise separable convolutions."""

    def __init__(self, channels: int, expansion: float = 1.0, use_depthwise: bool = False) -> None:
        super().__init__()
        hidden_channels = max(int(channels * expansion), channels)
        conv_cls = DWConvBNAct if use_depthwise else ConvBNAct
        self.conv1 = ConvBNAct(channels, hidden_channels, kernel_size=3, stride=1)
        self.conv2 = conv_cls(hidden_channels, channels, kernel_size=3, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a residual feature refinement block."""
        return x + self.conv2(self.conv1(x))


class RepCSPBlock(nn.Module):
    """CSP-style block with repeated residual refinement for efficient capacity."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        use_depthwise: bool = False,
    ) -> None:
        super().__init__()
        hidden_channels = out_channels // 2
        self.reduce_left = ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.reduce_right = ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.blocks = nn.Sequential(
            *[RepBlock(hidden_channels, use_depthwise=use_depthwise) for _ in range(num_blocks)]
        )
        self.fuse = ConvBNAct(hidden_channels * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        """Split, refine, concatenate, and fuse features."""
        left = self.blocks(self.reduce_left(x))
        right = self.reduce_right(x)
        return self.fuse(torch.cat((left, right), dim=1))


class SPPF_Lite(nn.Module):
    """Lightweight spatial pyramid pooling fast block using repeated max pooling."""

    def __init__(self, in_channels: int, out_channels: int, pool_kernel_size: int = 5) -> None:
        super().__init__()
        hidden_channels = max(in_channels // 2, 1)
        self.reduce = ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size // 2)
        self.fuse = ConvBNAct(hidden_channels * 4, out_channels, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        """Aggregate multi-scale pooled features with minimal overhead."""
        x = self.reduce(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.fuse(torch.cat((x, y1, y2, y3), dim=1))


class ChimeraBackbone(nn.Module):
    """Lightweight CNN backbone that returns P3, P4, and P5 feature maps."""

    def __init__(
        self,
        stem_channels: int = 32,
        stage_channels: Tuple[int, int, int, int] = (64, 128, 192, 256),
        stage_depths: Tuple[int, int, int, int] = (1, 2, 2, 2),
    ) -> None:
        super().__init__()
        c2, c3, c4, c5 = stage_channels
        d2, d3, d4, d5 = stage_depths

        self.stem = ConvBNAct(3, stem_channels, kernel_size=3, stride=2)

        self.stage2_down = ConvBNAct(stem_channels, c2, kernel_size=3, stride=2)
        self.stage2 = RepCSPBlock(c2, c2, num_blocks=d2, use_depthwise=False)

        self.stage3_down = ConvBNAct(c2, c3, kernel_size=3, stride=2)
        self.stage3 = RepCSPBlock(c3, c3, num_blocks=d3, use_depthwise=False)

        self.stage4_down = ConvBNAct(c3, c4, kernel_size=3, stride=2)
        self.stage4 = RepCSPBlock(c4, c4, num_blocks=d4, use_depthwise=True)

        self.stage5_down = ConvBNAct(c4, c5, kernel_size=3, stride=2)
        self.stage5 = RepCSPBlock(c5, c5, num_blocks=d5, use_depthwise=True)
        self.sppf = SPPF_Lite(c5, c5)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return P3, P4, and P5 features for downstream neck fusion."""
        if x.ndim != 4:
            raise AssertionError(f"Expected input with 4 dimensions [B, C, H, W], got shape {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise AssertionError(f"Expected RGB input with 3 channels, got {x.shape[1]}")

        x = self.stem(x)
        x = self.stage2(self.stage2_down(x))
        p3 = self.stage3(self.stage3_down(x))
        p4 = self.stage4(self.stage4_down(p3))
        p5 = self.sppf(self.stage5(self.stage5_down(p4)))
        return p3, p4, p5
