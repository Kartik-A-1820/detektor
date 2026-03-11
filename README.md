# Detektor

A production-ready object detection and instance segmentation framework optimized for rapid experimentation, stable training, and local deployment on modest hardware (GTX 1650 Ti 4GB).

This project is **fully vibe-coded**, meaning it was built through iterative collaboration between a human developer and AI coding assistants. In this repository, *vibe coding* means rapidly turning ideas into working code, refining them through repeated experimentation, and then hardening the useful workflows into a practical ML project.

## Recent Updates (March 2026)

- ✅ **Stable Training Baseline**: AdamW optimizer with cosine warmup scheduler
- ✅ **Loss Stability**: CIoU + BCE baseline with comprehensive numerical safeguards
- ✅ **Auto num_classes Detection**: Automatically detects class count from checkpoints
- ✅ **Folder Inference**: Batch process entire image directories
- ✅ **Modern FastAPI**: Updated to lifespan handlers (no deprecation warnings)
- ✅ **Detailed Loss Logging**: All loss components logged to CSV/JSONL
- ✅ **AMP Modernization**: Updated to `torch.amp` API
- ✅ **Ultralytics-Style Reporting**: Comprehensive plots and metrics summaries
- ✅ **Production-Grade Validation**: AP50, AP50-95, confusion matrix, threshold sweep
- ✅ **Task-Aware System**: Auto-detects detection vs segmentation, supports both modes

## Project Overview

Detektor is built for local-first ML development and deployment:

- train on YOLO-style datasets
- validate with lightweight metrics
- run image inference locally
- serve predictions through FastAPI
- export stable tensor outputs to ONNX

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
python run_smoke_checks.py
```

## Quick Start

### 1. Training

**Basic training** (with automatic plot generation):
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml
```

**Training with validation** (runs validation every epoch):
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml --run-val
```

**Training with periodic validation** (every 5 epochs):
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml --run-val --val-freq 5
```

**Resume from checkpoint:**
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml --resume runs/chimera/chimera_last.pt
```

**Initialize from pretrained weights:**
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml --weights pretrained/chimera_coco.pt
```

**With debug mode:**
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml --debug-loss
```

**With EMA and gradient clipping:**
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml --ema --grad-clip 10.0
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

Training is configured via YAML files in `configs/`. Example `chimera_s_512.yaml`:

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
  vram_cap: 0.80

loss:
  cls: "bce"              # classification loss
  box: "ciou"             # box regression loss
  obj: "bce"              # objectness loss
  seg: "bce_dice"         # segmentation loss

model:
  proto_k: 24
```

### CLI Arguments

- `--config`: Path to training config YAML (required)
- `--data-yaml`: Dataset YAML with train/val paths and class names
- `--resume`: Checkpoint path to resume training (restores optimizer, scheduler, epoch)
- `--weights`: Model weights to initialize from (model only, no optimizer state)
- `--ema`: Enable exponential moving average
- `--grad-clip`: Gradient clipping max norm (0 disables)
- `--debug-loss`: Enable detailed loss component debugging

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

See `OPTIMIZER_LOSS_BASELINE.md` for detailed stability documentation.

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
python validate_v2.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml
```

**With image saving:**
```bash
python validate_v2.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml --save-images --max-images 50
```

**With AP50-95 (COCO-style):**
```bash
python validate_v2.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml --compute-ap50-95
```

**Custom output directory:**
```bash
python validate_v2.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml --output-dir my_validation
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
python validate_v2.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml

# Generate visual report
python report.py --run-dir runs/validate/chimera_best
```

See `VALIDATION_OUTPUT_SCHEMA.md` for detailed output format documentation.

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
python report.py --run-dir runs/chimera
```

**Specify custom output directory:**
```bash
python report.py --run-dir runs/chimera --output-dir my_reports
```

**Verbose output:**
```bash
python report.py --run-dir runs/chimera --verbose
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
python report.py --run-dir runs/chimera
```

All plots and reports will be saved in the run directory for easy access and version control.

## Exporting Models

Export ONNX:

```bash
python export.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --output exports/chimera_odis.onnx
```

Compatibility alias:

```bash
python export_onnx.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt
```

## Running Tests

Run smoke checks:

```bash
python run_smoke_checks.py
```

Run the unittest suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Documentation

- **`README.md`** (this file): Quick start and usage guide
- **`OPTIMIZER_LOSS_BASELINE.md`**: Detailed optimizer and loss stability documentation
- **`REPORTING.md`**: Reporting module documentation and API reference
- **`VALIDATION_OUTPUT_SCHEMA.md`**: Validation metrics output format specification
- **`PROJECT_STATUS.md`**: Project roadmap and status
- **`CONTRIBUTING.md`**: Contribution guidelines

## Troubleshooting

### Training Issues

**Non-finite loss:**
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml data.yaml --debug-loss
```
Check which component fails and see `OPTIMIZER_LOSS_BASELINE.md`.

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
