from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from .blocks import ConvBNAct, RepBlock


class ChimeraPANNeck(nn.Module):
    """Lightweight PAN-FPN neck that fuses P3, P4, and P5 into N3, N4, and N5."""

    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (128, 192, 256),
        out_channels: Tuple[int, int, int] = (128, 192, 256),
    ) -> None:
        super().__init__()
        p3_channels, p4_channels, p5_channels = in_channels
        n3_channels, n4_channels, n5_channels = out_channels

        self.p5_lateral = ConvBNAct(p5_channels, n4_channels, kernel_size=1, stride=1)
        self.p4_lateral = ConvBNAct(p4_channels, n4_channels, kernel_size=1, stride=1)
        self.topdown_p4 = nn.Sequential(
            ConvBNAct(n4_channels * 2, n4_channels, kernel_size=1, stride=1),
            RepBlock(n4_channels, use_depthwise=False),
        )

        self.p4_to_p3 = ConvBNAct(n4_channels, n3_channels, kernel_size=1, stride=1)
        self.p3_lateral = ConvBNAct(p3_channels, n3_channels, kernel_size=1, stride=1)
        self.topdown_p3 = nn.Sequential(
            ConvBNAct(n3_channels * 2, n3_channels, kernel_size=1, stride=1),
            RepBlock(n3_channels, use_depthwise=False),
        )

        self.down_n3 = ConvBNAct(n3_channels, n4_channels, kernel_size=3, stride=2)
        self.bottomup_p4 = nn.Sequential(
            ConvBNAct(n4_channels * 2, n4_channels, kernel_size=1, stride=1),
            RepBlock(n4_channels, use_depthwise=True),
        )

        self.down_n4 = ConvBNAct(n4_channels, n5_channels, kernel_size=3, stride=2)
        self.p5_out = ConvBNAct(p5_channels, n5_channels, kernel_size=1, stride=1)
        self.bottomup_p5 = nn.Sequential(
            ConvBNAct(n5_channels * 2, n5_channels, kernel_size=1, stride=1),
            RepBlock(n5_channels, use_depthwise=True),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, p3: Tensor, p4: Tensor, p5: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Fuse multi-scale backbone outputs into neck features."""
        if p3.ndim != 4 or p4.ndim != 4 or p5.ndim != 4:
            raise AssertionError("Expected P3, P4, and P5 to be 4D feature maps")

        p5_lat = self.p5_lateral(p5)
        p4_lat = self.p4_lateral(p4)
        f4 = self.topdown_p4(torch.cat((p4_lat, self.upsample(p5_lat)), dim=1))

        p3_lat = self.p3_lateral(p3)
        f3 = self.topdown_p3(torch.cat((p3_lat, self.upsample(self.p4_to_p3(f4))), dim=1))

        n4 = self.bottomup_p4(torch.cat((f4, self.down_n3(f3)), dim=1))
        n5 = self.bottomup_p5(torch.cat((self.p5_out(p5), self.down_n4(n4)), dim=1))
        return f3, n4, n5
