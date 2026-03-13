# Detektor

A production-ready object detection and instance segmentation framework optimized for rapid experimentation, stable training, and local deployment on modest hardware (GTX 1650 Ti 4GB).

This project is **fully vibe-coded**, meaning it was built through iterative collaboration between a human developer and AI coding assistants. In this repository, *vibe coding* means rapidly turning ideas into working code, refining them through repeated experimentation, and then hardening the useful workflows into a practical ML project.

## Quick Links

- **[Documentation Index](docs/README.md)** - Organized guides, references, and project notes
- **[Quick Start Guide](docs/guides/QUICKSTART.md)** - Complete workflow from training to deployment
- **[Tools Reference](docs/reference/TOOLS.md)** - Comprehensive CLI documentation
- **[Changelog](CHANGELOG.md)** - Version history and updates

## Recent Updates (March 2026)

### Production Hardening
- ✅ **Dataset Validation**: Pre-training validation tool to catch data issues
- ✅ **Comprehensive Testing**: 77+ unit, integration, and regression tests
- ✅ **Model Packaging**: Reproducible artifact packaging with full metadata
- ✅ **Production API**: Versioned FastAPI with validation, logging, and metrics
- ✅ **Gradio UI**: Lightweight web interface for local testing
- ✅ **Docker Deployment**: Production-ready containerization

### Training & Validation
- ✅ **Stable Training Baseline**: AdamW optimizer with cosine warmup scheduler
- ✅ **Loss Stability**: CIoU + BCE baseline with comprehensive numerical safeguards
- ✅ **Auto num_classes Detection**: Automatically detects class count from checkpoints
- ✅ **Architecture Matrix**: Lists CPU/GPU compatibility for every profile and can benchmark all profiles automatically
- ✅ **Detailed Loss Logging**: All loss components logged to CSV/JSONL
- ✅ **Production-Grade Validation**: AP50, AP50-95, confusion matrix, threshold sweep
- ✅ **Task-Aware System**: Auto-detects detection vs segmentation, supports both modes

### Deployment & Tooling
- ✅ **Folder Inference**: Batch process entire image directories
- ✅ **Modern FastAPI**: Updated to lifespan handlers (no deprecation warnings)
- ✅ **ONNX Benchmarking**: Compare PyTorch vs ONNX Runtime performance
- ✅ **Comprehensive Reporting**: Ultralytics-style plots and metrics summaries

## Project Overview

Detektor is built for local-first ML development and deployment:

- **Train** on YOLO-style datasets with stable, production-grade training loop
- **Validate** dataset quality before training with comprehensive checks
- **Evaluate** models with production-grade metrics (AP50, AP50-95, confusion matrix)
- **Infer** on single images or batch process entire folders
- **Serve** predictions through production FastAPI with versioned endpoints
- **Package** models with full reproducibility metadata
- **Deploy** with Docker for CPU or GPU environments
- **Export** to ONNX for optimized inference

## Task Modes

Detektor intelligently supports two task modes:

### 🎯 Detection Mode (`detect`)
- **Bounding box detection only**
- Faster training (no mask loss)
- Lower memory usage
- Returns: boxes, scores, labels

### 🎭 Segmentation Mode (`segment`)
- **Instance segmentation with masks**
- Full mask + box training
- Returns: boxes, scores, labels, masks
- Auto-generates boxes from masks if needed

### Auto-Detection

The system automatically detects your dataset type:

```
==========================================================
TASK DETECTION SUMMARY
==========================================================
Detected task mode: segment
Total label files: 150
Sampled files: 50
  - Bbox format: 0
  - Segment format: 50

Mode: INSTANCE SEGMENTATION
  - Training: Box + mask losses
  - Inference: Returns bounding boxes + masks
  - Boxes auto-generated from masks if needed
==========================================================
```

**Dataset Format Detection:**
- **Segmentation format**: `class_id x1 y1 x2 y2 x3 y3 ...` (polygon points)
- **Detection format**: `class_id x_center y_center width height` (bbox only)

The system samples your dataset and automatically chooses the appropriate mode.
- prepare for later runtime optimization such as TensorRT

The codebase is intentionally lightweight and practical, with a bias toward single-machine workflows and modest GPUs such as the GTX 1650 Ti 4GB.

## Features

### Training & Optimization
- **Configurable Optimizers**: AdamW (default) or SGD with Nesterov momentum
- **Learning Rate Scheduling**: Cosine annealing with warmup (3 epochs default)
- **Numerical Stability**: CIoU loss with epsilon guards, NaN sanitization, AMP dtype safety
- **Loss Components**: BCE classification, CIoU box regression, BCE objectness, BCE+Dice segmentation
- **Detailed Logging**: All loss components logged per step (CSV + JSONL)
- **Training Features**: AMP, gradient clipping, EMA, resume support, best checkpoint tracking

### Inference & Deployment
- **Auto-Detection**: Automatically detects `num_classes` from checkpoints
- **Folder Inference**: Batch process entire directories with progress tracking
- **FastAPI Service**: Modern async API with health checks and prediction endpoints
- **Class Name Support**: Load class names from dataset YAML for readable outputs
- **Flexible Output**: Bounding boxes with optional mask visualization

### Dataset & Export
- **YOLO/Roboflow Support**: Auto-configuration from dataset YAML
- **Multiple Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- **ONNX Export**: Stable tensor-only export path
- **Validation Metrics**: Lightweight benchmarking and smoke tests

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ with CUDA support (optional but recommended)
- 4GB+ GPU VRAM (GTX 1650 Ti or better)

### Install Dependencies

**Using pip:**
```bash
pip install -r requirements.txt
```

**Using uv (faster):**
```bash
uv pip install -r requirements.txt
```

**Development dependencies:**
```bash
pip install -r requirements-dev.txt
```

### Verify Installation

**Check CUDA:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Run smoke tests:**
```bash
python -m scripts.run_smoke_checks
```

## Quick Start

**New to Detektor?** See the **[Quick Start Guide](docs/guides/QUICKSTART.md)** for a complete walkthrough from dataset preparation to production deployment.

**Looking for specific tools?** Check the **[Tools Reference](docs/reference/TOOLS.md)** for comprehensive CLI documentation.

### Golden Path: 5-Minute Start

```bash
# 1. Validate your dataset
python check_dataset.py --data-yaml F:/data/data.yaml

# 2. Train your model
python train.py --data-yaml F:/data/data.yaml

# 3. Validate performance
python validate.py --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml

# 4. Run inference
python infer.py --weights runs/chimera/chimera_best.pt --source test.jpg --data-yaml F:/data/data.yaml

# 5. Start API server
python serve.py --weights runs/chimera/chimera_best.pt --device cuda
```

### 1. Training

**Basic training:**
```bash
python train.py --data-yaml F:/data/data.yaml
```

**Resume from checkpoint:**
```bash
python train.py --data-yaml F:/data/data.yaml --resume runs/chimera/chimera_last.pt
```

**Initialize from pretrained weights:**
```bash
python train.py --data-yaml F:/data/data.yaml --weights pretrained/chimera_coco.pt
```

**With debug mode:**
```bash
python train.py --data-yaml F:/data/data.yaml --debug-loss
```

**With EMA and gradient clipping:**
```bash
python train.py --data-yaml F:/data/data.yaml --ema --grad-clip 10.0
```

**Integrated Features:**
- **Auto-plot generation** after each epoch (loss curves, LR, metrics)
- **Optional validation** during training (`--run-val`)
- **Configurable validation frequency** (`--val-freq N`)
- **Comprehensive final report** at end of training
- **All plots saved** to `runs/chimera/plots/`
- **Training summary** saved to `runs/chimera/training_summary.txt`

### 2. Inference

### Task-Aware Inference

**Segmentation mode** (default - returns boxes + masks):
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source test.jpg --data-yaml F:/data/data.yaml --task segment
```

**Detection mode** (boxes only - faster):
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source test.jpg --data-yaml F:/data/data.yaml --task detect
```

### Single Image

Run inference on a single image:

```bash
python infer.py --weights runs/chimera/chimera_best.pt --source test.jpg --data-yaml F:/data/data.yaml
```

### Batch Inference

Batch inference on a folder:

```bash
python infer.py --weights runs/chimera/chimera_best.pt --source F:/data/test_images/ --data-yaml F:/data/data.yaml
```

**Custom output folder:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source F:/data/test/images --save-path my_results
```

### 3. API Server

**Start server:**
```bash
python serve.py --weights runs/chimera/chimera_best.pt --device cuda
```

**Test with curl:**
```bash
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" -F "image=@image.jpg"
```

**Access interactive docs:**
```
http://localhost:8000/docs
```

## Training

### Configuration

Training is **auto-configured by default** from:
- current free GPU VRAM or free system RAM
- CPU count
- dataset size and class metadata from `--data-yaml`

You can start with no training config file at all:

```bash
python train.py --data-yaml F:/data/data.yaml
```

`--config` remains optional and acts as a base YAML. CLI flags override both the auto-selected values and anything in the YAML.

### Architecture Profiles

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

### Example YAML

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

### Training CLI

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

### Architecture Compatibility And Sweep

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

### Optimizer Options

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

### Learning Rate Schedule

**Cosine with Warmup (default):**
- Linear warmup: epochs 0-2 (configurable)
- Cosine decay: epochs 3+ to end
- Smooth learning rate transitions

**Constant LR:**
- Set `scheduler: "none"` in config
- Useful for debugging or short runs

### Loss Components

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

### Training Output

```
runs/chimera/
├── chimera_last.pt          # Latest checkpoint
├── chimera_best.pt          # Best checkpoint (lowest loss)
├── chimera_final_weights.pt # Final model weights
├── train_metrics.csv        # Per-step metrics (CSV)
├── train_metrics.jsonl      # Per-step metrics (JSONL)
├── epoch_summaries.jsonl    # Per-epoch summaries
└── run_summary.json         # Training configuration
```

### Debugging Non-Finite Loss

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

## Inference

### Single Image

**Basic inference:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source image.jpg
```

**With class names:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source image.jpg --data-yaml F:/data/data.yaml
```

**Custom output:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source image.jpg --save-path my_output.jpg
```

### Folder Inference (Batch Processing)

**Process entire folder:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source F:/data/test/images --data-yaml F:/data/data.yaml
```

**Custom output directory:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source F:/data/test/images --save-path my_results
```

**Output:**
- Processes all `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` images
- Saves to `runs/inference/` by default
- Files named: `{original_name}_pred.jpg`
- Shows progress: `[1/25] processing: image.jpg`
- Displays detections with class names

## Dataset Validation

Before training, validate your dataset to catch common issues:

```bash
python check_dataset.py --data-yaml F:/data/data.yaml
```

### What It Checks

**File Integrity:**
- Missing image files
- Missing label files
- Corrupt or unreadable images
- Empty label files

**Label Format:**
- Malformed YOLO label rows (must have at least 5 values)
- Invalid class IDs (out of range)
- Invalid normalized coordinates (must be in [0, 1])
- Zero or negative box dimensions

**Dataset Quality:**
- Class distribution across dataset
- Image size distribution
- Duplicate filenames (potential data issues)

### Validation Output

The tool generates two report files in `reports/`:

**JSON Summary (`dataset_check.json`):**
```json
{
  "has_errors": false,
  "has_warnings": true,
  "total_issues": 3,
  "error_count": 0,
  "warning_count": 3,
  "stats": {
    "total_images": 150,
    "total_labels": 148,
    "total_annotations": 892,
    "empty_labels": 2,
    "corrupt_images": 0,
    "class_distribution": {
      "0": 234,
      "1": 312,
      "2": 198,
      "3": 148
    },
    "image_size_distribution": {
      "640x480": 120,
      "1280x720": 30
    },
    "duplicate_filenames": []
  }
}
```

**CSV Issues (`dataset_check.csv`):**
```csv
severity,category,message,file_path,line_number
warning,empty_label,Label file is empty,F:/data/train/labels/img_042.txt,
error,invalid_class_id,Class ID 5 out of range [0, 3],F:/data/train/labels/img_089.txt,3
```

### Exit Codes

- **Exit 0**: Validation passed (warnings allowed)
- **Exit 1**: Validation failed (errors found)

Use in CI/CD or pre-training hooks:

```bash
python check_dataset.py --data-yaml F:/data/data.yaml || exit 1
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml
```

### Validation Options

- `--data-yaml`: Dataset YAML file (required)
- `--output-dir`: Report output directory (default: `reports`)
- `--splits`: Dataset splits to validate (default: `train val`)

### Inference Options

- `--weights`: Model checkpoint path (required)
- `--source`: Image file or folder (required)
- `--data-yaml`: Dataset YAML for class names (optional)
- `--num-classes`: Override auto-detection (optional)
- `--img-size`: Input size (default: 512)
- `--conf-thresh`: Confidence threshold (default: 0.25)
- `--iou-thresh`: NMS IoU threshold (default: 0.6)
- `--max-det`: Max detections per image (default: 100)
- `--save-path`: Output file or folder (optional)

### Auto-Detection Features

**num_classes:** Automatically detected from checkpoint
```
auto-detected num_classes=4 from checkpoint
```

**Class names:** Loaded from `--data-yaml` if provided
```
loaded class names: ['ball', 'goalkeeper', 'player', 'referee']
detections: 24, labels: ['player', 'player', 'goalkeeper', ...]
```

## API Server

### Starting the Server

**Basic:**
```bash
python serve.py --weights runs/chimera/chimera_best.pt
```

**With CUDA:**
```bash
python serve.py --weights runs/chimera/chimera_best.pt --device cuda
```

**Custom host/port:**
```bash
python serve.py --weights runs/chimera/chimera_best.pt --host 0.0.0.0 --port 8080
```

### API Endpoints

> ✅ All endpoints emit `X-Request-ID` + `X-Response-Time` headers for traceability.

| Endpoint | Purpose | Notes |
| --- | --- | --- |
| `GET /health` | Liveness probe | Returns device + model load state |
| `GET /ready` | Readiness probe | Returns `ready=true` only after model + warmup |
| `GET /version` | Build info | Includes API + model metadata |
| `GET /metrics` | Lightweight stats | Latency percentiles, totals, error counts |
| `POST /v1/predict` | Single image inference | Structured JSON response with detections |
| `POST /v1/predict_batch` | Multi-image inference | Validates batch size + per-image errors |
| `POST /predict` | Legacy alias | Maps to `/v1/predict` (deprecated) |

**Health checks:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/version
curl http://localhost:8000/metrics
```

**Single prediction:**
```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Accept: application/json" \
  -F "image=@image.jpg" \
  -F "include_masks=true" \
  -F "conf_thresh=0.35" \
  -F "iou_thresh=0.5"
```

**Batch prediction:**
```bash
curl -X POST "http://localhost:8000/v1/predict_batch" \
  -F "images=@frame1.png" \
  -F "images=@frame2.png"
```

Responses now include:
```json
{
  "request_id": "7c6f...",
  "num_detections": 2,
  "detections": [
    {"box": [x1, y1, x2, y2], "score": 0.93, "label": 0, "mask": "..."},
    {"box": [x1, y1, x2, y2], "score": 0.88, "label": 3}
  ],
  "boxes": [[...]],         // legacy fields
  "scores": [0.93, 0.88],
  "labels": [0, 3],
  "image_width": 1024,
  "image_height": 768,
  "inference_time_ms": 42.3
}
```

Uploads are validated for MIME type, corrupt data, and size (`--max-upload-size-mb`).

### Interactive Documentation

FastAPI provides automatic interactive docs:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Configuration & Environment Variables

Every CLI flag has an `DETEKTOR_*` env var override so you can run via `uvicorn` or container orchestrators without editing the command line:

| Flag | Env Var | Default |
| --- | --- | --- |
| `--weights` | `DETEKTOR_WEIGHTS` | **required** |
| `--host` | `DETEKTOR_HOST` | `127.0.0.1` |
| `--port` | `DETEKTOR_PORT` | `8000` |
| `--device` | `DETEKTOR_DEVICE` | `auto` |
| `--num-classes` | `DETEKTOR_NUM_CLASSES` | auto-detect |
| `--proto-k` | `DETEKTOR_PROTO_K` | `24` |
| `--img-size` | `DETEKTOR_IMG_SIZE` | `512` |
| `--conf-thresh` | `DETEKTOR_CONF_THRESH` | `0.25` |
| `--iou-thresh` | `DETEKTOR_IOU_THRESH` | `0.6` |
| `--max-det` | `DETEKTOR_MAX_DET` | `100` |
| `--topk-pre-nms` | `DETEKTOR_TOPK_PRE_NMS` | `300` |
| `--mask-thresh` | `DETEKTOR_MASK_THRESH` | `0.5` |
| `--include-masks` | `DETEKTOR_INCLUDE_MASKS` | `false` |
| `--max-upload-size-mb` | `DETEKTOR_MAX_UPLOAD_SIZE_MB` | `10` |
| `--max-batch-size` | `DETEKTOR_MAX_BATCH_SIZE` | `16` |
| `--no-warmup` | `DETEKTOR_NO_WARMUP` | `false` |
| `--warmup-iterations` | `DETEKTOR_WARMUP_ITERATIONS` | `3` |
| `--log-level` | `DETEKTOR_LOG_LEVEL` | `INFO` |

Warmup is enabled by default and runs a few dummy passes to remove first-request latency spikes. Disable it with `--no-warmup` if you need instant start.

### Features

✅ **Auto num_classes detection** from checkpoint  
✅ **Modern FastAPI lifespan** handlers (no deprecation warnings)  
✅ **Async processing** for concurrent requests  
✅ **Structured logging** with request IDs  
✅ **Readiness + metrics endpoints** for orchestrators  
✅ **Optional mask output** via query parameter  
✅ **CUDA support** with automatic fallback

## Deployment (Docker & Compose)

Detektor ships with production-oriented Docker assets for local or on-prem deployments.

### Build the Image

```bash
docker build -t detektor:latest .
```

Weights are mounted at runtime—no large artifacts baked into the image by default.

### Run on CPU

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/artifacts/chimera_best:/artifacts:ro \
  -e DETEKTOR_WEIGHTS=/artifacts/model.pt \
  detektor:latest
```

### Run with NVIDIA GPU

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v $(pwd)/artifacts/chimera_best:/artifacts:ro \
  -e DETEKTOR_WEIGHTS=/artifacts/model.pt \
  -e DETEKTOR_DEVICE=cuda \
  detektor:latest
```

### docker-compose

```bash
docker compose up --build
```

The compose file includes both a CPU service and an optional GPU service (requires `runtime: nvidia`). Logs stream to stdout/stderr to integrate with container runtimes. Health checks hit `/ready` every 30 seconds.

### Relevant Environment Variables

| Variable | Purpose |
| --- | --- |
| `DETEKTOR_WEIGHTS` | Path to mounted model artifact (required) |
| `DETEKTOR_DEVICE` | `cpu`, `cuda`, or `auto` |
| `DETEKTOR_HOST` / `DETEKTOR_PORT` | Network binding |
| `DETEKTOR_CONF_THRESH` / `DETEKTOR_IOU_THRESH` / `DETEKTOR_MAX_DET` | Runtime thresholds |
| `DETEKTOR_INCLUDE_MASKS` | Default mask behavior |

Use `docker logs detektor-api` to inspect structured logs. For production, run behind a reverse proxy and mount packaged artifacts created via `scripts.package_model`.

## Dataset Format

### YOLO/Roboflow Format

Detektor supports standard YOLO-style datasets exported from Roboflow or similar tools.

**Dataset YAML (`data.yaml`):**
```yaml
train: F:/data/train/images
val: F:/data/valid/images
test: F:/data/test/images
nc: 4
names: ['ball', 'goalkeeper', 'player', 'referee']
```

**Directory Structure:**
```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Label Format

YOLO format: `class x_center y_center width height` (normalized 0-1)

**Example `labels/image1.txt`:**
```
0 0.5 0.5 0.3 0.4
2 0.7 0.3 0.2 0.3
```

### Auto-Configuration

When `--data-yaml` is provided, Detektor automatically:

✅ Resolves train/val paths  
✅ Converts `images/` paths to split roots  
✅ Sets `data.format = "yolo"`  
✅ Updates `data.num_classes` from `nc`  
✅ Loads class names from `names`  
✅ Validates directory structure  

### Supported Image Formats

- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.webp`

### Creating Your Dataset

1. **Export from Roboflow:**
   - Choose "YOLOv8" format
   - Download and extract

2. **Verify structure:**
   ```bash
   ls F:/data/train/images  # Should show images
   ls F:/data/train/labels  # Should show .txt files
   ```

3. **Check data.yaml:**
   ```bash
   cat F:/data/data.yaml
   ```

4. **Train:**
   ```bash
   python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml
   ```

## Validation

### Production-Grade Validation

Run comprehensive validation with detailed metrics and artifacts:

**Basic validation:**
```bash
python validate.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml
```

**With image saving:**
```bash
python validate.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml --save-images --max-images 50
```

**With AP50-95 (COCO-style):**
```bash
python validate.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml --compute-ap50-95
```

**Custom output directory:**
```bash
python validate.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml --output-dir my_validation
```

### Validation Metrics

The production-grade validation computes:

**Detection Metrics:**
- Precision, Recall, F1 score
- AP50 (Average Precision at IoU 0.5)
- mAP50 (Mean AP across all classes)
- AP50-95 (COCO-style, optional)
- Mean box IoU
- Per-class precision, recall, F1, AP50

**Segmentation Metrics:**
- Mean mask IoU
- Mean Dice coefficient
- Per-instance mask quality

**Analysis Tools:**
- Confusion matrix
- Confidence threshold sweep
- Precision-Recall curves
- Per-class breakdown

### Validation Outputs

```
runs/validate/<run_name>/
├── metrics.json              # Comprehensive metrics
├── per_class_metrics.csv     # Per-class performance
├── confusion_matrix.csv      # Confusion matrix
├── threshold_sweep.csv       # Threshold analysis
└── images/                   # Annotated images (optional)
```

### Metrics Summary Example

```json
{
  "overall": {
    "precision": 0.8542,
    "recall": 0.7891,
    "f1": 0.8203,
    "ap50": 0.8234,
    "map50": 0.8123,
    "mean_box_iou": 0.7456,
    "mean_mask_iou": 0.6789
  },
  "per_class": [
    {
      "class_name": "ball",
      "precision": 0.92,
      "recall": 0.85,
      "f1": 0.88,
      "ap50": 0.89
    }
  ],
  "threshold_sweep": {
    "best_threshold": 0.4,
    "best_f1": 0.8203
  }
}
```

### Validation Options

- `--config`: Path to config YAML (required)
- `--weights`: Path to model weights (required)
- `--data-yaml`: Dataset YAML for class names
- `--batch-size`: Validation batch size (default: 4)
- `--conf-thresh`: Confidence threshold (default: 0.25)
- `--iou-thresh`: IoU threshold for matching (default: 0.5)
- `--output-dir`: Custom output directory
- `--save-images`: Save annotated validation images
- `--max-images`: Max images to save (default: 20)
- `--compute-ap50-95`: Compute COCO-style AP (slower)

### Edge Cases Handled

✅ **Empty predictions** - Gracefully handles no detections  
✅ **Empty ground truth** - Handles images without annotations  
✅ **Missing classes** - Classes absent in validation split  
✅ **Segmentation disabled** - Works without masks  
✅ **NaN/Inf values** - Sanitizes invalid predictions  
✅ **Memory efficient** - Optimized for GTX 1650 Ti 4GB  

### Integration with Reporting

Validation outputs integrate seamlessly with the reporting module:

```bash
# Run validation
python validate.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml

# Generate visual report
python -m scripts.report --run-dir runs/validate/chimera_best
```

See `docs/reference/VALIDATION_OUTPUT_SCHEMA.md` for detailed output format documentation.

## Reporting

Detektor includes a comprehensive reporting module that **automatically generates** training visualizations and summaries during training.

### Automatic Integration

**Reports are generated automatically:**
- ✅ **After each epoch** - Loss curves, LR plot, epoch metrics updated
- ✅ **At end of training** - Comprehensive final report with all plots and summary
- ✅ **No separate command needed** - Everything happens during training

**Output location:** `runs/chimera/plots/`

### Features

- **Loss Curves**: Track all loss components over training
- **Learning Rate Schedule**: Visualize LR changes across epochs
- **Epoch Metrics**: Monitor training progress
- **Training Summary**: Detailed statistics and configuration info
- **Graceful Degradation**: Handles missing data elegantly

### Manual Report Generation (Optional)

If you need to regenerate reports manually:

```bash
python -m scripts.report --run-dir runs/chimera
```

**Specify custom output directory:**
```bash
python -m scripts.report --run-dir runs/chimera --output-dir my_reports
```

**Verbose output:**
```bash
python -m scripts.report --run-dir runs/chimera --verbose
```

### Generated Artifacts

**Plots (`runs/chimera/plots/`):**
- `loss_total.png` - Total training loss curve
- `loss_components.png` - Individual loss components (cls, box, obj, mask)
- `learning_rate.png` - Learning rate schedule
- `epoch_loss.png` - Per-epoch average loss
- `per_class_ap.png` - Per-class Average Precision bar chart (if validation data exists)
- `precision_recall_curve.png` - Precision-Recall curve (if validation data exists)
- `confusion_matrix.png` - Normalized confusion matrix heatmap (if validation data exists)

**Reports (`runs/chimera/reports/`):**
- `metrics_summary.json` - Machine-readable training and validation summary
- `per_class_metrics.csv` - Per-class AP metrics in CSV format
- `report_status.json` - Report generation status and warnings

### Metrics Summary Example

```json
{
  "training": {
    "total_steps": 190,
    "total_epochs": 5,
    "final_loss": 9.395267,
    "min_loss": 8.872341,
    "best_epoch": 5,
    "best_epoch_loss": 9.395267,
    "final_lr": 0.001,
    "avg_loss_cls": 2.456,
    "avg_loss_box": 3.123,
    "avg_loss_obj": 1.987,
    "avg_loss_mask": 1.829
  },
  "validation": {
    "map50": 0.8575,
    "per_class_ap": [0.85, 0.92, 0.78, 0.88],
    "class_names": ["ball", "goalkeeper", "player", "referee"]
  }
}
```

### Required Input Files

The report generator reads from your training run directory:

**Required:**
- `train_metrics.csv` or `train_metrics.jsonl` - Per-step training metrics
- `epoch_summaries.jsonl` - Per-epoch summaries

**Optional:**
- `val_metrics.json` - Validation results (for validation plots)

### Graceful Degradation

The report generator handles missing data gracefully:
- If validation metrics are missing, only training plots are generated
- If learning rate is not logged, LR plot is skipped
- Warnings are logged for missing files
- Report generation continues even if some plots fail

### Integration with Training

Reports are automatically compatible with the training output format. After training:

```bash
# Train your model
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml

# Generate comprehensive report
python -m scripts.report --run-dir runs/chimera
```

All plots and reports will be saved in the run directory for easy access and version control.

## Exporting Models

Export ONNX:

```bash
python export.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --output exports/chimera_odis.onnx
```

Compatibility alias:

```bash
python -m scripts.export_onnx --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt
```

## Running Tests

Detektor includes comprehensive unit, integration, and regression tests.

### Fast Tests (Unit Tests Only)

Run lightweight unit tests for quick validation:

```bash
python -m unittest discover -s tests -p "test_box_ops.py"
python -m unittest discover -s tests -p "test_ciou.py"
python -m unittest discover -s tests -p "test_mask_ops.py"
python -m unittest discover -s tests -p "test_schemas.py"
python -m unittest discover -s tests -p "test_config.py"
```

Or run all unit tests:

```bash
python -m unittest tests.test_box_ops tests.test_ciou tests.test_mask_ops tests.test_schemas tests.test_config
```

### Full Test Suite

Run all tests including integration and regression tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Test Categories

**Unit Tests:**
- `test_box_ops.py` - Box decoding and flattening operations
- `test_ciou.py` - CIoU loss and IoU helpers
- `test_mask_ops.py` - Mask composition, cropping, and resizing
- `test_schemas.py` - API schema serialization and validation
- `test_config.py` - Configuration parsing and dataset YAML handling

**Integration Tests:**
- `test_integration.py` - End-to-end workflows (inference, reporting, validation)
- `test_api.py` - FastAPI endpoint testing

**Regression Tests:**
- `test_regression.py` - Schema stability and no-NaN guarantees

**Smoke Tests:**
- `test_model.py` - Model architecture smoke tests
- `test_predict.py` - Prediction format validation
- `test_export.py` - ONNX export smoke tests

### Running Specific Test Classes

```bash
python -m unittest tests.test_box_ops.TestBoxOps
python -m unittest tests.test_ciou.TestCIoU.test_ciou_loss_identical_boxes
```

### Test Coverage Notes

- **Unit tests** are fast (<1s each) and cover core helper functions
- **Integration tests** may take longer and test full workflows
- **Regression tests** ensure API stability and no-NaN guarantees
- All tests are designed to run locally without GPU requirements

## Documentation

### Getting Started
- **[docs/README.md](docs/README.md)**: Documentation index
- **[docs/guides/QUICKSTART.md](docs/guides/QUICKSTART.md)**: Complete workflow from dataset to deployment
- **[docs/reference/TOOLS.md](docs/reference/TOOLS.md)**: Comprehensive CLI tool reference
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and updates

### Technical Documentation
- **[docs/internal/OPTIMIZER_LOSS_BASELINE.md](docs/internal/OPTIMIZER_LOSS_BASELINE.md)**: Optimizer and loss stability guide
- **[docs/reference/REPORTING.md](docs/reference/REPORTING.md)**: Reporting module documentation
- **[docs/reference/VALIDATION_OUTPUT_SCHEMA.md](docs/reference/VALIDATION_OUTPUT_SCHEMA.md)**: Validation metrics format
- **[docs/status/PROJECT_STATUS.md](docs/status/PROJECT_STATUS.md)**: Project roadmap and status
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Contribution guidelines

### Configuration Examples
- **`configs/chimera_s_512.yaml`**: Default training configuration
- **`examples/sample_val_metrics.json`**: Example validation output

## Troubleshooting

### Training Issues

**Non-finite loss:**
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml data.yaml --debug-loss
```
Check which component fails and see `docs/internal/OPTIMIZER_LOSS_BASELINE.md`.

**Out of memory:**
- Reduce `batch_size` in config
- Lower `img_size` (e.g., 416 or 384)
- Disable AMP: `amp: false`
- Increase `grad_accum` for gradient accumulation

**Checkpoint mismatch:**
- Use `--num-classes` to override auto-detection
- Ensure checkpoint matches model architecture

### Inference Issues

**Wrong num_classes:**
```bash
python infer.py --weights checkpoint.pt --source image.jpg --num-classes 4
```

**No detections:**
- Lower `--conf-thresh` (default: 0.25)
- Check if model was trained on similar data
- Verify image format is supported

### API Issues

**Port already in use:**
```bash
python serve.py --weights checkpoint.pt --port 8080
```

**CUDA out of memory:**
```bash
python serve.py --weights checkpoint.pt --device cpu
```

## Performance Tips

### For GTX 1650 Ti 4GB

**Recommended config:**
```yaml
train:
  img_size: 512
  batch_size: 8
  amp: true
  grad_accum: 1
  vram_cap: 0.80
```

**For faster training:**
- Use SGD: `optimizer: "sgd"`, `lr: 0.01`
- Reduce warmup: `warmup_epochs: 1`
- Disable EMA (saves memory)

**For better accuracy:**
- Use AdamW: `optimizer: "adamw"`, `lr: 0.002`
- Enable EMA: `--ema`
- Increase epochs: `epochs: 100`

## Contributing

Contributions are welcome, especially around:

- Model architecture improvements
- TensorRT export and optimization
- Additional dataset loaders
- Training stability enhancements
- Performance benchmarking
- Documentation improvements

See `CONTRIBUTING.md` for contribution guidelines.

## License

Detektor is released under the MIT License. See `LICENSE`.

## Credits

- Developed through iterative human-AI collaboration
- Built as a practical vibe-coded ML engineering project
- Optimized for lightweight experimentation and local deployment
- Tested on GTX 1650 Ti 4GB (4GB VRAM)
- Production-ready with stable training and inference pipelines

## Citation

If you use Detektor in your research or project, please cite:

```bibtex
@software{detektor2026,
  title={Detektor: Lightweight Object Detection and Instance Segmentation},
  author={Vibe-Coded Collaboration},
  year={2026},
  url={https://github.com/Kartik-A-1820/detektor}
}
```


