"""Regression: box regression must be learnable for large objects (the dead-ReLU / pixel-regression collapse)."""

from __future__ import annotations

import unittest

import torch

from models.chimera import ChimeraODIS
from utils.box_ops import BOX_DECODE_PIXELS, BOX_DECODE_STRIDE


def _large_object_batch(n: int = 8, size: int = 128, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, size, size, generator=gen) * 0.2
    targets = []
    for i in range(n):
        x1, y1 = 0.10 + 0.1 * torch.rand(1, generator=gen).item(), 0.10 + 0.1 * torch.rand(1, generator=gen).item()
        x2, y2 = 0.85 + 0.1 * torch.rand(1, generator=gen).item(), 0.85 + 0.1 * torch.rand(1, generator=gen).item()
        images[i, :, int(y1 * size): int(y2 * size), int(x1 * size): int(x2 * size)] += 0.7  # a big bright object
        mask = torch.zeros(size, size)
        mask[int(y1 * size): int(y2 * size), int(x1 * size): int(x2 * size)] = 1.0
        targets.append({"boxes": torch.tensor([[x1, y1, x2, y2]]), "labels": torch.tensor([0]), "masks": mask.unsqueeze(0)})
    return images, targets


def _train_box_loss(box_decode: str, steps: int = 40) -> tuple[float, float]:
    torch.manual_seed(0)
    model = ChimeraODIS(
        num_classes=1, proto_k=8, stem_channels=16, backbone_channels=(24, 48, 72, 96), backbone_depths=(1, 1, 1, 1),
        neck_channels=(48, 72, 96), head_feat_channels=48, box_decode=box_decode,
    ).train()
    model.detection_loss.current_epoch = 5  # past the box-weight warm-up
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    images, targets = _large_object_batch()
    first = last = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        losses = model.compute_loss(images, targets, return_dict=True)
        losses["loss_total"].backward()
        optimizer.step()
        value = float(losses["loss_box"].detach())
        first = value if step == 0 else first
        last = value
    return first, last


class LargeObjectTrainingTests(unittest.TestCase):
    def test_box_loss_falls_with_stride_decoding(self) -> None:
        first, last = _train_box_loss(BOX_DECODE_STRIDE)
        self.assertLess(last, 0.8 * first, (first, last))
        self.assertLess(last, 0.9, (first, last))

    def test_stride_decoding_beats_legacy_pixel_decoding(self) -> None:
        _, stride_last = _train_box_loss(BOX_DECODE_STRIDE)
        _, pixel_last = _train_box_loss(BOX_DECODE_PIXELS)
        self.assertLess(stride_last, pixel_last, (stride_last, pixel_last))


if __name__ == "__main__":
    unittest.main()
