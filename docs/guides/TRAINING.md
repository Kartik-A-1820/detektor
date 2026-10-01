# Training

## Configuration

Training is **auto-configured by default** from:
- current free GPU VRAM or free system RAM
- CPU count
- dataset size and class metadata from `--data-yaml`

You can start with no training config file at all:

```bash
python train.py --data-yaml F:/data/data.yaml
```

`--config` remains optional and acts as a base YAML. CLI flags override both the auto-selected values and anything in the YAML.

## Architecture Profiles

Use `--model <name>` to override the auto-selected architecture profile.

| Profile | Display Name | Typical Use |
| --- | --- | --- |
| `firefly` | Firefly | CPU / ultra-low-memory fallback |
| `comet` | Comet | 4 GB GPUs and small datasets |
| `nova` | Nova | Higher-capacity small/medium runs |
| `pulsar` | Pulsar | Mid-tier GPUs with more memory |
| `quasar` | Quasar | Larger GPUs and heavier runs |
| `supernova` | Supernova | Highest-capacity profile |

Example:

```bash
python train.py --data-yaml F:/data/data.yaml --model nova
```

## Example YAML

Example base YAML in `configs/chimera_s_512.yaml`:

```yaml
train:
  img_size: 512
  epochs: 50
  batch_size: 8
  lr: 0.002
  weight_decay: 0.05
  optimizer: "adamw"      # or "sgd"
  scheduler: "cosine"     # or "none"
  warmup_epochs: 3
  momentum: 0.937         # for SGD only
  amp: true
  grad_accum: 1
  vram_cap: 0.95
  auto_tune: true
  maximize_batch_size: true
  batch_size_multiple: 4
  max_batch_probe: 64

augment:
  enabled: true
  hsv_h: 0.015
  hsv_s: 0.70
  hsv_v: 0.40
  fliplr: 0.50
  flipud: 0.05
  translate: 0.10
  scale: 0.30
  mosaic: 0.00
  cutmix: 0.00
  random_cut: 0.00
  random_cut_holes: 1
  random_cut_scale: 0.25

loss:
  cls: "bce"              # classification loss
  box: "ciou"             # box regression loss
  obj: "bce"              # objectness loss
  seg: "bce_dice"         # segmentation loss

model:
  profile: "comet"
  proto_k: 24
```

## Training CLI

Core run selection:

| Flag | Purpose |
| --- | --- |
| `--config` | Optional base training YAML |
| `--data-yaml` | Dataset YAML with train/val paths and class names |
| `--model` | Override architecture profile: `firefly`, `comet`, `nova`, `pulsar`, `quasar`, `supernova` |
| `--out-dir` | Override output run directory |
| `--device` | Override device: `auto`, `cpu`, `cuda` |
| `--seed` | Override random seed |
| `--deterministic` | Enable deterministic training where possible |

Training hyperparameters:

| Flag | YAML Key | Purpose |
| --- | --- | --- |
| `--img-size` | `train.img_size` | Input image size |
| `--epochs` | `train.epochs` | Number of epochs |
| `--batch-size` | `train.batch_size` | Physical batch size |
| `--num-workers` | `train.num_workers` | Data loader worker count |
| `--lr` | `train.lr` | Learning rate |
| `--weight-decay` | `train.weight_decay` | Weight decay |
| `--optimizer` | `train.optimizer` | `adamw` or `sgd` |
| `--scheduler` | `train.scheduler` | `cosine` or `none` |
| `--warmup-epochs` | `train.warmup_epochs` | Warmup length |
| `--momentum` | `train.momentum` | SGD momentum |
| `--amp` / `--no-amp` | `train.amp` | Explicitly enable or disable AMP |
| `--grad-accum` | `train.grad_accum` | Gradient accumulation steps |
| `--vram-cap` | `train.vram_cap` | Target fraction of currently free VRAM for this process |
| `--conf-thresh` | `train.conf_thresh` | Validation/in-training confidence threshold |
| `--iou-thresh` | `train.iou_thresh` | Validation/in-training IoU threshold |
| `--auto-tune` / `--no-auto-tune` | `train.auto_tune` | Enable or disable hardware-aware auto config |
| `--maximize-batch-size` / `--no-maximize-batch-size` | `train.maximize_batch_size` | Enable or disable CUDA batch probing |
| `--batch-size-multiple` | `train.batch_size_multiple` | Batch probe multiple, usually `4` |
| `--max-batch-probe` | `train.max_batch_probe` | Upper bound for batch probing |

Augmentation overrides:

| Flag | YAML Key | Purpose |
| --- | --- | --- |
| `--augment` / `--no-augment` | `augment.enabled` | Enable or disable training augmentation |
| `--hsv-h` | `augment.hsv_h` | HSV hue jitter |
| `--hsv-s` | `augment.hsv_s` | HSV saturation jitter |
| `--hsv-v` | `augment.hsv_v` | HSV value jitter |
| `--fliplr` | `augment.fliplr` | Horizontal flip probability |
| `--flipud` | `augment.flipud` | Vertical flip probability |
| `--translate` | `augment.translate` | Translation strength |
| `--scale` | `augment.scale` | Scale jitter |
| `--mosaic` | `augment.mosaic` | Mosaic probability |
| `--cutmix` | `augment.cutmix` | CutMix probability |
| `--random-cut` | `augment.random_cut` | Random cut probability |
| `--random-cut-holes` | `augment.random_cut_holes` | Random cut hole count |
| `--random-cut-scale` | `augment.random_cut_scale` | Random cut relative size |

Model and run-control overrides:

| Flag | YAML Key | Purpose |
| --- | --- | --- |
| `--proto-k` | `model.proto_k` | Prototype mask channel count |
| `--resume` | n/a | Resume optimizer, scaler, and epoch state |
| `--weights` | n/a | Initialize weights before training |
| `--ema` | n/a | Enable exponential moving average |
| `--grad-clip` | n/a | Gradient clipping max norm |
| `--debug-loss` | n/a | Print detailed non-finite loss diagnostics |
| `--run-val` | n/a | Run validation during training |
| `--val-freq` | n/a | Validation frequency in epochs |

Example override-heavy run:

```bash
python train.py \
  --data-yaml F:/data/data.yaml \
  --model nova \
  --epochs 10 \
  --batch-size 20 \
  --no-auto-tune \
  --mosaic 0.2 \
  --cutmix 0.1 \
  --out-dir runs/nova_manual
```

## Architecture Compatibility And Sweep

Use `model_matrix.py` to list which profiles fit on the current machine and optionally run a real training sweep across all architectures.

```bash
# Compatibility only
python model_matrix.py --data-yaml F:/data/data.yaml

# Real 3-epoch sweep across every profile
python model_matrix.py --data-yaml F:/data/data.yaml --run-train-sweep --epochs 3
```

Outputs:
- `runs/architecture_matrix/compatibility_matrix.json`
- `runs/architecture_matrix/train_sweep_summary.json`

The compatibility report includes CPU and CUDA support, approximate maximum batch size per profile, and the auto-recommended architecture for the current hardware.

## Optimizer Options

**AdamW (default):**
- Adaptive learning rates
- Better generalization
- Less sensitive to LR tuning
- Recommended for most cases

**SGD with Nesterov:**
- More stable convergence
- YOLO-style training
- Requires higher LR (e.g., 0.01)
- Set `optimizer: "sgd"` in config

## Learning Rate Schedule

**Cosine with Warmup (default):**
- Linear warmup: epochs 0-2 (configurable)
- Cosine decay: epochs 3+ to end
- Smooth learning rate transitions

**Constant LR:**
- Set `scheduler: "none"` in config
- Useful for debugging or short runs

## Loss Components

All loss components are logged to `runs/chimera/train_metrics.csv` and `.jsonl`:

- `loss_total`: Combined weighted loss
- `loss_cls`: Classification (BCE)
- `loss_box`: Box regression (CIoU)
- `loss_obj`: Objectness (BCE)
- `loss_mask`: Combined mask loss
- `loss_mask_bce`: Mask BCE component
- `loss_mask_dice`: Mask Dice component
- `num_fg`: Foreground anchor count
- `num_mask_pos`: Positive mask instances
- `lr`: Current learning rate

## Choosing the best checkpoint

* **With `--run-val`** (recommended): `chimera_best.pt` is the epoch with the highest validation mAP50 (checked every
  `--val-freq` epochs).
* **Without validation:** the epoch with the lowest *training* loss, **ignoring the first 3 warm-up epochs** (the box-loss
  weight and learning rate ramp up there, so early losses are artificially low). Training loss is only a weak proxy for
  quality — use `--run-val` whenever you have a validation split.

## Training Output

```
runs/chimera/
├── chimera_last.pt          # Latest checkpoint
├── chimera_best.pt          # Best checkpoint (highest val mAP50 with --run-val; otherwise lowest post-warm-up train loss)
├── chimera_final_weights.pt # Final model weights
├── train_metrics.csv        # Per-step metrics (CSV)
├── train_metrics.jsonl      # Per-step metrics (JSONL)
├── epoch_summaries.jsonl    # Per-epoch summaries
└── run_summary.json         # Training configuration
```

## Debugging Non-Finite Loss

If training produces NaN/Inf losses:

1. **Enable debug mode:**
   ```bash
   python train.py --config configs/chimera_s_512.yaml --data-yaml data.yaml --debug-loss
   ```

2. **Check which component fails:**
   ```
   non-finite loss components: loss_mask_dice=inf, loss_total=inf
     num_fg=12.0, num_mask_pos=8.0
   ```

3. **Try stability fixes:**
   - Disable AMP: `amp: false`
   - Lower LR: `lr: 0.001`
   - Increase warmup: `warmup_epochs: 5`
   - Reduce batch size: `batch_size: 4`

See `docs/internal/OPTIMIZER_LOSS_BASELINE.md` for detailed stability documentation.
