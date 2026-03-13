# Optimizer and Loss Baseline Documentation

## Overview
This document describes the production-ready optimizer and loss configuration baseline for the Detektor detection+segmentation model, optimized for numerical stability and GTX 1650 Ti 4GB hardware.

## Optimizer Configuration

### Default: AdamW
- **Type**: `adamw` (default)
- **Learning Rate**: `0.002` (configurable)
- **Weight Decay**: `0.05` (configurable)
- **Why AdamW**: Adaptive learning rates, better generalization, less sensitive to LR tuning than SGD

### Alternative: SGD with Nesterov Momentum
- **Type**: `sgd` (optional)
- **Momentum**: `0.937` (default, configurable)
- **Nesterov**: `true` (always enabled for SGD)
- **Use Case**: When you need more stable convergence or want YOLO-style training

### Configuration
```yaml
train:
  optimizer: "adamw"  # or "sgd"
  lr: 0.002
  weight_decay: 0.05
  momentum: 0.937  # only used for SGD
```

## Learning Rate Scheduler

### Default: Cosine Annealing with Warmup
- **Type**: `cosine` (default)
- **Warmup Epochs**: `3` (configurable)
- **Schedule**:
  - Epochs 0-2: Linear warmup from `0` to `lr`
  - Epochs 3+: Cosine decay from `lr` to `0.5 * lr`
- **Why Cosine**: Smooth decay, proven effective for detection tasks, prevents abrupt LR drops

### Alternative: No Scheduler
- **Type**: `none`
- **Use Case**: Constant LR for debugging or short runs

### Configuration
```yaml
train:
  scheduler: "cosine"  # or "none"
  warmup_epochs: 3
```

## Loss Configuration

### Detection Losses

#### Classification Loss
- **Type**: `bce` (Binary Cross-Entropy with Logits)
- **Implementation**: `F.binary_cross_entropy_with_logits`
- **Stability**: Numerically stable logit-space computation
- **Normalization**: Per-foreground anchor normalization

#### Box Regression Loss
- **Type**: `ciou` (Complete IoU)
- **Implementation**: Custom CIoU with epsilon guards
- **Stability Safeguards**:
  - Width/height clamped to `EPS = 1e-7`
  - Area/union clamped to `EPS`
  - Enclosing diagonal clamped to `EPS`
  - `torch.nan_to_num()` applied to CIoU outputs before IoU target computation
  - IoU values clamped to `[0, 1]`
- **Why CIoU**: Considers overlap, distance, and aspect ratio; more stable than GIoU/DIoU

#### Objectness Loss
- **Type**: `bce` (Binary Cross-Entropy with Logits)
- **Target**: IoU-based quality scores from CIoU
- **Stability**: Targets clamped to `[0, 1]` and cast to prediction dtype/device under AMP

### Segmentation Losses

#### Mask BCE Loss
- **Type**: `bce` (Binary Cross-Entropy with Logits)
- **Implementation**: `F.binary_cross_entropy_with_logits`
- **Reduction**: `mean` over all mask pixels
- **Stability**: Logit-space computation, empty mask handling

#### Mask Dice Loss
- **Type**: `dice` (Soft Dice Loss)
- **Implementation**: `1 - (2 * intersection + EPS) / (pred + target + EPS)`
- **Stability Safeguards**:
  - Epsilon `1e-6` in numerator and denominator
  - Empty mask early return with zero loss
  - Per-instance computation with mean reduction

#### Combined Mask Loss
- **Formula**: `loss_mask = bce_weight * loss_bce + dice_weight * loss_dice`
- **Default Weights**: `bce_weight=1.0`, `dice_weight=1.0`

### Configuration
```yaml
loss:
  cls: "bce"
  box: "ciou"
  obj: "bce"
  seg: "bce_dice"
```

## Numerical Stability Safeguards

### AMP (Automatic Mixed Precision)
- **API**: Modernized to `torch.amp` (no deprecation warnings)
- **GradScaler**: `torch.amp.GradScaler("cuda", enabled=amp_enabled)`
- **Autocast**: `torch.amp.autocast("cuda", enabled=amp_enabled)`
- **Dtype Casting**: All GT-derived tensors cast to prediction dtype/device via `_cast_like()` helper

### Box Operations
- **Decode**: ReLU clamping on ltrb distances before box construction
- **IoU**: Width/height/area/union all clamped to `EPS = 1e-7`
- **CIoU**: Enclosing diagonal, aspect ratio denominator clamped to `EPS`
- **NaN Guards**: `torch.nan_to_num(ciou_values, nan=1.0, posinf=1.0, neginf=1.0)`

### Mask Operations
- **Crop Regions**: Box coordinates clamped to valid prototype grid bounds
- **Minimum Crop Size**: Enforced 1-pixel minimum width/height
- **Dice Denominator**: Epsilon `1e-6` prevents division by zero
- **Empty Masks**: Early return with zero loss

### Loss Normalization
- **Foreground Normalizer**: `total_fg.clamp(min=1.0)` prevents division by zero
- **Empty Batch Handling**: Box loss returns `pred_box.sum() * 0.0` when no foregrounds

## Loss Debugging

### Debug Mode
Enable detailed loss component diagnostics:
```bash
python train.py --config configs/chimera_s_512.yaml --debug-loss
```

### Output Format
When non-finite loss detected:
```
non-finite loss components: loss_mask_dice=inf, loss_total=inf
  num_fg=12.0, num_mask_pos=8.0
```

### Loss Breakdown
All training runs log:
- `loss_total`: Combined weighted loss
- `loss_cls`: Classification loss
- `loss_box`: Box regression loss (CIoU)
- `loss_obj`: Objectness loss
- `loss_mask`: Combined mask loss
- `loss_mask_bce`: Mask BCE component
- `loss_mask_dice`: Mask Dice component
- `num_fg`: Foreground anchor count
- `num_mask_pos`: Positive mask instance count

## Why DFL Was Not Adopted

**Distribution Focal Loss (DFL)** is a modern box regression technique used in YOLOv8+ that predicts a distribution over discrete box offset bins instead of direct regression.

### Reasons for Exclusion
1. **Head Architecture Incompatibility**: Current model uses direct ltrb regression head (4 channels). DFL requires redesigning the box head to output `4 * (reg_max + 1)` channels (e.g., 64 channels for `reg_max=15`).
2. **Increased Complexity**: DFL adds softmax distribution prediction, integral computation, and bin-based loss calculation.
3. **Memory Overhead**: 16x increase in box head output channels (4 → 64) is prohibitive on 4GB VRAM.
4. **Stability First**: Current CIoU baseline is proven stable. DFL can be explored after baseline stability is confirmed.
5. **Minimal Gains for Small Models**: DFL benefits are most pronounced in large models; marginal for lightweight architectures.

### Future Consideration
If box regression instability persists after baseline tuning, DFL can be explored with:
- Head redesign to output `[B, 4*(reg_max+1), H, W]`
- Softmax + integral decoding
- DFL loss implementation
- Careful memory profiling

## Training Recommendations

### For GTX 1650 Ti 4GB
```yaml
train:
  img_size: 512
  batch_size: 8
  lr: 0.002
  weight_decay: 0.05
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_epochs: 3
  amp: true
  grad_accum: 1
  vram_cap: 0.80
```

### For Debugging Instability
1. Enable debug mode: `--debug-loss`
2. Disable AMP temporarily: `amp: false`
3. Reduce batch size: `batch_size: 4`
4. Increase warmup: `warmup_epochs: 5`
5. Lower learning rate: `lr: 0.001`

### For Faster Convergence
1. Use SGD: `optimizer: "sgd"`
2. Increase LR: `lr: 0.01` (for SGD)
3. Reduce warmup: `warmup_epochs: 1`
4. Enable EMA: `--ema`

## Summary

### Optimizer Defaults
- **AdamW** with `lr=0.002`, `weight_decay=0.05`
- **Cosine scheduler** with 3-epoch warmup
- **AMP enabled** with modernized API

### Loss Defaults
- **Classification**: BCE with logits
- **Box Regression**: CIoU with epsilon guards and NaN sanitization
- **Objectness**: BCE with IoU-based targets
- **Segmentation**: BCE + Dice with epsilon guards

### Stability Safeguards
- All box operations use `EPS = 1e-7` clamping
- CIoU outputs sanitized with `torch.nan_to_num()`
- Mask crop regions validated and clamped
- Dice denominator protected with epsilon
- AMP dtype/device consistency enforced via `_cast_like()`
- Foreground normalizer clamped to `min=1.0`

### Production-Ready
- Lightweight and memory-efficient
- No experimental losses or complex architectures
- Proven stable on GTX 1650 Ti 4GB
- Clear debugging and logging
- Configurable via YAML without code changes


