import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ChimeraODIS(nn.Module):
    def __init__(self, num_classes=1, proto_k=24):
        super().__init__()
        self.backbone = nn.Sequential(
            Conv(3, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 192, 3, 2),
            Conv(192, 256, 3, 2),
        )
        self.detect_head = nn.Conv2d(256, num_classes + 5, 1)
        self.proto_head = nn.Conv2d(128, proto_k, 1)

    def forward(self, x):
        feats = self.backbone(x)
        det = self.detect_head(feats)
        return det

    def compute_loss(self, imgs, targets):
        preds = self.forward(imgs)
        return preds.mean()  # placeholder loss
