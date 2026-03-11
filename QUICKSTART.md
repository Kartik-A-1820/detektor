# Detektor Quick Start Guide

This guide walks you through the complete workflow from dataset preparation to production deployment.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Golden Path: Training to Serving

### Step 1: Prepare Your Dataset

Organize your dataset in YOLO format:

```
F:/data/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

**data.yaml:**
```yaml
train: F:/data/train/images
val: F:/data/val/images
nc: 4
names: ['ball', 'goalkeeper', 'player', 'referee']
```

### Step 2: Validate Your Dataset

Before training, check for common issues:

```bash
python check_dataset.py --data-yaml F:/data/data.yaml
```

**Expected output:**
```
✅ VALIDATION PASSED - No issues found

Dataset Statistics:
  Total images: 150
  Total labels: 150
  Total annotations: 892
  Class distribution:
    Class 0: 234 annotations (26.2%)
    Class 1: 312 annotations (35.0%)
    Class 2: 198 annotations (22.2%)
    Class 3: 148 annotations (16.6%)
```

Fix any errors before proceeding to training.

### Step 3: Train Your Model

Start training with the default configuration:

```bash
python train.py \
  --config configs/chimera_s_512.yaml \
  --data-yaml F:/data/data.yaml
```

**Monitor training:**
```
Epoch 1/50 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:15
  loss: 12.345  cls: 3.456  box: 5.678  obj: 2.345  mask: 0.866
  lr: 0.000667  fg: 1234  mask_pos: 456

Epoch 5/50 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:10
  loss: 8.234  cls: 2.123  box: 3.456  obj: 1.789  mask: 0.866
  lr: 0.002000  fg: 1456  mask_pos: 512
```

**Training outputs:**
```
runs/chimera/
├── chimera_best.pt          # Best checkpoint (use this for inference)
├── chimera_last.pt          # Latest checkpoint
├── train_metrics.csv        # Detailed metrics
├── epoch_summaries.jsonl    # Per-epoch summaries
└── plots/                   # Training curves
```

### Step 4: Validate Your Model

Run validation to get detailed metrics:

```bash
python validate.py \
  --weights runs/chimera/chimera_best.pt \
  --data-yaml F:/data/data.yaml \
  --split val
```

**Expected output:**
```
Validation Results:
  mAP50: 0.8575
  mAP50-95: 0.6234
  Precision: 0.8234
  Recall: 0.7891

Per-Class AP50:
  ball: 0.8512
  goalkeeper: 0.9234
  player: 0.7823
  referee: 0.8731
```

### Step 5: Test Inference

Run inference on a single image:

```bash
python infer.py \
  --weights runs/chimera/chimera_best.pt \
  --source test_image.jpg \
  --data-yaml F:/data/data.yaml
```

**Batch inference on a folder:**
```bash
python infer.py \
  --weights runs/chimera/chimera_best.pt \
  --source F:/data/test/images/ \
  --data-yaml F:/data/data.yaml
```

**Output:**
```
runs/inference/
├── test_image_pred.jpg
├── img_001_pred.jpg
└── ...
```

### Step 6: Generate Training Report

Create comprehensive training analysis:

```bash
python report.py --run-dir runs/chimera
```

**Generates:**
```
runs/chimera/
├── plots/
│   ├── loss_curves.png
│   ├── learning_rate.png
│   ├── class_distribution.png
│   └── ...
└── metrics_summary.json
```

### Step 7: Package Your Model

Create a reproducible model artifact:

```bash
python package_model.py \
  --weights runs/chimera/chimera_best.pt \
  --config configs/chimera_s_512.yaml \
  --data-yaml F:/data/data.yaml \
  --output artifacts/chimera_v1
```

**Creates:**
```
artifacts/chimera_v1/
├── model.pt                 # Model weights
├── config.yaml              # Training config
├── data.yaml                # Dataset config
├── class_names.json         # Class names
├── metrics_summary.json     # Training metrics
├── environment.txt          # Python/CUDA versions
├── git_commit.txt           # Git commit hash
└── package_manifest.json    # Complete metadata
```

### Step 8: Deploy with FastAPI

Start the production API server:

```bash
python serve.py \
  --weights runs/chimera/chimera_best.pt \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000
```

**Test the API:**
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/v1/predict" \
  -F "image=@test.jpg" \
  -F "conf_thresh=0.25"

# Interactive docs
open http://localhost:8000/docs
```

### Step 9: Deploy with Docker (Optional)

Build and run with Docker:

```bash
# Build image
docker build -t detektor:latest .

# Run CPU service
docker-compose up detektor-cpu

# Run GPU service
docker-compose up detektor-gpu
```

### Step 10: Launch UI (Optional)

Start the Gradio UI for local testing:

```bash
python ui/app.py
```

Access at `http://localhost:7860`

---

## Common Workflows

### Resume Training

```bash
python train.py \
  --config configs/chimera_s_512.yaml \
  --data-yaml F:/data/data.yaml \
  --resume runs/chimera/chimera_last.pt
```

### Export to ONNX

```bash
python export.py \
  --config configs/chimera_s_512.yaml \
  --weights runs/chimera/chimera_best.pt \
  --output exports/chimera.onnx
```

### Benchmark Performance

```bash
python benchmark.py \
  --weights runs/chimera/chimera_best.pt \
  --onnx exports/chimera.onnx \
  --source F:/data/test/images
```

### Run Tests

```bash
# Fast unit tests only
python -m unittest tests.test_box_ops tests.test_ciou tests.test_mask_ops

# Full test suite
python -m unittest discover -s tests -p "test_*.py"
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```yaml
train:
  batch_size: 4  # Reduce from 8
  vram_cap: 0.70  # Lower VRAM cap
```

### NaN/Inf Loss

Enable debug mode:
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml --debug-loss
```

Try stability fixes:
- Disable AMP: `amp: false`
- Lower learning rate: `lr: 0.001`
- Increase warmup: `warmup_epochs: 5`

### Dataset Validation Errors

Fix issues reported by `check_dataset.py`:
- Remove corrupt images
- Fix malformed labels
- Ensure class IDs are in range [0, nc-1]
- Verify coordinates are normalized [0, 1]

---

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [OPTIMIZER_LOSS_BASELINE.md](OPTIMIZER_LOSS_BASELINE.md) for training tips
- See [REPORTING.md](REPORTING.md) for metrics documentation
- Review [VALIDATION_OUTPUT_SCHEMA.md](VALIDATION_OUTPUT_SCHEMA.md) for output formats

## Support

For issues and questions:
- Check existing documentation
- Review test cases for usage examples
- Examine example configs in `configs/`
