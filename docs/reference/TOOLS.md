# Detektor Tools Reference

Complete reference for all Detektor command-line tools.

---

## Training

### `train.py`

Train a ChimeraODIS model on YOLO-format datasets.

**Basic Usage:**
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml
```

**Arguments:**
- `--config PATH` (required): Path to training configuration YAML
- `--data-yaml PATH`: Dataset YAML with train/val paths and class names
- `--resume PATH`: Resume from checkpoint (restores full training state)
- `--weights PATH`: Initialize model weights only (no optimizer state)
- `--ema`: Enable exponential moving average
- `--grad-clip FLOAT`: Gradient clipping max norm (0 disables, default: 0)
- `--debug-loss`: Enable detailed loss component debugging
- `--run-val`: Run validation after each epoch
- `--val-freq INT`: Validation frequency in epochs (default: 1)

**Outputs:**
```
runs/{run_name}/
├── {run_name}_best.pt          # Best checkpoint
├── {run_name}_last.pt          # Latest checkpoint
├── {run_name}_final_weights.pt # Final weights
├── train_metrics.csv           # Per-step metrics
├── train_metrics.jsonl         # Per-step metrics (JSON)
├── epoch_summaries.jsonl       # Per-epoch summaries
├── run_summary.json            # Training config
└── plots/                      # Training curves
```

**Examples:**
```bash
# Basic training
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml

# Resume training
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml \
  --resume runs/chimera/chimera_last.pt

# Training with EMA and gradient clipping
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml \
  --ema --grad-clip 10.0

# Debug mode for loss issues
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml \
  --debug-loss
```

---

## Validation

### `validate.py`

Compute comprehensive validation metrics on a dataset split.

**Basic Usage:**
```bash
python validate.py --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml
```

**Arguments:**
- `--weights PATH` (required): Path to model checkpoint
- `--data-yaml PATH` (required): Dataset YAML file
- `--split NAME`: Dataset split to validate (default: `val`)
- `--img-size INT`: Input image size (default: 512)
- `--conf-thresh FLOAT`: Confidence threshold (default: 0.25)
- `--iou-thresh FLOAT`: NMS IoU threshold (default: 0.6)
- `--max-det INT`: Maximum detections per image (default: 100)
- `--batch-size INT`: Batch size for validation (default: 8)
- `--device NAME`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--output PATH`: Output JSON file (default: `runs/validation/val_metrics.json`)

**Outputs:**
```
runs/validation/
├── val_metrics.json            # Detailed metrics
├── confusion_matrix.png        # Confusion matrix plot
└── pr_curves.png               # Precision-recall curves
```

**Metrics Computed:**
- mAP50: Mean Average Precision at IoU=0.50
- mAP50-95: Mean Average Precision at IoU=0.50:0.95
- Precision, Recall, F1
- Per-class AP50
- Confusion matrix

**Examples:**
```bash
# Validate on validation set
python validate.py --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml

# Validate on test set
python validate.py --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml \
  --split test

# Custom thresholds
python validate.py --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml \
  --conf-thresh 0.5 --iou-thresh 0.7
```

---

## Inference

### `infer.py`

Run inference on images or folders with visualization.

**Basic Usage:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source image.jpg
```

**Arguments:**
- `--weights PATH` (required): Path to model checkpoint
- `--source PATH` (required): Image file or folder path
- `--data-yaml PATH`: Dataset YAML for class names (optional)
- `--num-classes INT`: Override auto-detected number of classes
- `--img-size INT`: Input image size (default: 512)
- `--conf-thresh FLOAT`: Confidence threshold (default: 0.25)
- `--iou-thresh FLOAT`: NMS IoU threshold (default: 0.6)
- `--max-det INT`: Maximum detections per image (default: 100)
- `--mask-thresh FLOAT`: Mask threshold (default: 0.5)
- `--device NAME`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--save-path PATH`: Custom output path

**Outputs:**
```
runs/inference/
├── image_pred.jpg
├── img_001_pred.jpg
└── ...
```

**Examples:**
```bash
# Single image
python infer.py --weights runs/chimera/chimera_best.pt --source test.jpg

# Folder inference
python infer.py --weights runs/chimera/chimera_best.pt --source F:/data/test/images/

# With class names
python infer.py --weights runs/chimera/chimera_best.pt --source test.jpg \
  --data-yaml F:/data/data.yaml

# Custom output
python infer.py --weights runs/chimera/chimera_best.pt --source test.jpg \
  --save-path my_output.jpg
```

---

## API Server

### `serve.py`

Start the FastAPI inference service (optionally with the web console).

**Basic Usage:**
```bash
python serve.py --weights runs/chimera/chimera_best.pt
python serve.py --weights runs/chimera/chimera_best.pt --ui          # + web console at /ui
```

**Key arguments** (every flag also has a `DETEKTOR_*` environment variable — see
[CONFIGURATION.md](CONFIGURATION.md) for the complete list):
- `--weights PATH` (required): checkpoint; sibling `chimera_best.pt` / `chimera_last.pt` are switchable at runtime
- `--device NAME`: `auto` (default), `cpu` or `cuda`
- `--host IP` / `--port INT`: bind address (default `127.0.0.1:8000`)
- `--img-size INT`: network input size (default 512)
- `--conf-thresh FLOAT` / `--iou-thresh FLOAT` / `--max-det INT`: default inference thresholds (0.25 / 0.6 / 100)
- `--max-upload-size-mb INT` (10), `--max-batch-size INT` (16), `--max-concurrency INT` (1)
- `--api-key KEY`: require `X-API-Key` / Bearer auth on inference, runtime and metrics endpoints
- `--cors-origins LIST`: comma-separated allowed origins (CORS off by default)
- `--ui`, `--ui-path PATH`, `--ui-auth user:password`: web console

**Endpoints** (full reference: [API.md](API.md)):
- `GET /health`, `GET /ready`, `GET /version` - probes (always open)
- `GET /metrics`, `GET /metrics/prometheus` - JSON / Prometheus metrics
- `GET /runtime`, `POST /runtime/select_model` - run metadata and checkpoint switching
- `POST /v1/predict`, `POST /v1/predict_batch` - inference; `POST /predict` is a deprecated alias

**Examples:**
```bash
python serve.py --weights runs/chimera/chimera_best.pt --device cuda
python serve.py --weights runs/chimera/chimera_best.pt --host 0.0.0.0 --api-key "$(openssl rand -hex 32)"

curl http://localhost:8000/ready
curl -X POST "http://localhost:8000/v1/predict?conf_thresh=0.3" -H "X-API-Key: $KEY" -F "image=@test.jpg"
open http://localhost:8000/docs
```

---

## Reporting

### `scripts.report`

Generate comprehensive training analysis and visualizations.

**Basic Usage:**
```bash
python -m scripts.report --run-dir runs/chimera
```

**Arguments:**
- `--run-dir PATH` (required): Training run directory
- `--output-dir PATH`: Output directory for plots (default: same as run-dir)

**Generates:**
```
runs/chimera/
├── plots/
│   ├── loss_curves.png         # Training loss over time
│   ├── learning_rate.png       # Learning rate schedule
│   ├── loss_components.png     # Individual loss components
│   └── epoch_metrics.png       # Per-epoch summaries
└── metrics_summary.json        # Comprehensive metrics
```

**Examples:**
```bash
# Generate report
python -m scripts.report --run-dir runs/chimera

# Custom output location
python -m scripts.report --run-dir runs/chimera --output-dir reports/chimera_analysis
```

---

## Model Packaging

### `scripts.package_model`

Package model artifacts for reproducibility and deployment.

**Basic Usage:**
```bash
python -m scripts.package_model \
  --weights runs/chimera/chimera_best.pt \
  --config configs/chimera_s_512.yaml \
  --data-yaml F:/data/data.yaml
```

**Arguments:**
- `--weights PATH` (required): Path to model checkpoint
- `--config PATH` (required): Path to training config YAML
- `--data-yaml PATH` (required): Path to dataset YAML
- `--output PATH`: Output directory (default: `artifacts/{model_name}`)
- `--include-onnx PATH`: Include ONNX export in package

**Creates:**
```
artifacts/chimera_v1/
├── model.pt                    # Model weights
├── model.onnx                  # ONNX export (if provided)
├── config.yaml                 # Training config
├── data.yaml                   # Dataset config
├── class_names.json            # Class names
├── metrics_summary.json        # Training metrics
├── environment.txt             # Python/CUDA versions
├── git_commit.txt              # Git commit hash
└── package_manifest.json       # Complete metadata
```

**Examples:**
```bash
# Basic packaging
python -m scripts.package_model \
  --weights runs/chimera/chimera_best.pt \
  --config configs/chimera_s_512.yaml \
  --data-yaml F:/data/data.yaml

# Include ONNX export
python -m scripts.package_model \
  --weights runs/chimera/chimera_best.pt \
  --config configs/chimera_s_512.yaml \
  --data-yaml F:/data/data.yaml \
  --include-onnx exports/chimera.onnx

# Custom output location
python -m scripts.package_model \
  --weights runs/chimera/chimera_best.pt \
  --config configs/chimera_s_512.yaml \
  --data-yaml F:/data/data.yaml \
  --output artifacts/production_v1
```

---

## Benchmarking

### `python -m benchmarks`

Unified benchmark framework: complexity, latency, throughput, memory, training speed, cold start, ONNX Runtime, HTTP
API load, synthetic end-to-end quality, robustness and accuracy. Full documentation: [BENCHMARKS.md](../BENCHMARKS.md).

```bash
python -m benchmarks list
python -m benchmarks run --suites fast --profiles firefly,comet,nova
python -m benchmarks run --suites all --profiles all --tag full
python -m benchmarks run --suites accuracy,robustness --weights runs/chimera/chimera_best.pt --data-yaml data.yaml
python -m benchmarks report runs/benchmarks/<run>/results.json -o report.md
python -m benchmarks compare base.json new.json --threshold 15 --fail-on-regression
python -m benchmarks.synthetic --out data/synthetic        # dataset-free demo data
```

### `scripts.benchmark` (legacy)

Benchmark PyTorch and ONNX Runtime inference performance.

**Basic Usage:**
```bash
python -m scripts.benchmark \
  --weights runs/chimera/chimera_best.pt \
  --onnx exports/chimera.onnx \
  --source F:/data/test/images
```

**Arguments:**
- `--weights PATH` (required): Path to PyTorch checkpoint
- `--onnx PATH`: Path to ONNX model (optional)
- `--source PATH` (required): Image folder for benchmarking
- `--num-samples INT`: Number of images to benchmark (default: 100)
- `--warmup INT`: Warmup iterations (default: 10)
- `--img-size INT`: Input image size (default: 512)
- `--output-dir PATH`: Output directory (default: `runs/benchmarks`)

**Outputs:**
```
runs/benchmarks/
├── benchmark_summary.json      # Detailed results
└── benchmark_table.csv         # Comparison table
```

**Metrics:**
- Average latency (ms)
- P50/P95 latency (ms)
- Throughput (images/sec)
- Warmup vs steady-state timing

**Examples:**
```bash
# Benchmark PyTorch and ONNX
python -m scripts.benchmark \
  --weights runs/chimera/chimera_best.pt \
  --onnx exports/chimera.onnx \
  --source F:/data/test/images

# PyTorch only
python -m scripts.benchmark \
  --weights runs/chimera/chimera_best.pt \
  --source F:/data/test/images

# Custom sample count
python -m scripts.benchmark \
  --weights runs/chimera/chimera_best.pt \
  --source F:/data/test/images \
  --num-samples 50
```

---

## Dataset Validation

### `check_dataset.py`

Validate YOLO-format datasets before training.

**Basic Usage:**
```bash
python check_dataset.py --data-yaml F:/data/data.yaml
```

**Arguments:**
- `--data-yaml PATH` (required): Path to dataset YAML
- `--output-dir PATH`: Report output directory (default: `reports`)
- `--splits NAMES`: Splits to validate (default: `train val`)

**Validates:**
- Missing image/label files
- Corrupt or unreadable images
- Malformed YOLO label format
- Invalid class IDs (out of range)
- Invalid normalized coordinates (not in [0, 1])
- Zero or negative box dimensions
- Empty label files

**Outputs:**
```
reports/
├── dataset_check.json          # Detailed validation results
└── dataset_check.csv           # Issue list
```

**Exit Codes:**
- 0: Validation passed (warnings allowed)
- 1: Validation failed (errors found)

**Examples:**
```bash
# Validate dataset
python check_dataset.py --data-yaml F:/data/data.yaml

# Validate specific splits
python check_dataset.py --data-yaml F:/data/data.yaml --splits train val test

# Use in CI/CD
python check_dataset.py --data-yaml F:/data/data.yaml || exit 1
```

---

## ONNX Export

### `export.py`

Export trained model to ONNX format.

**Basic Usage:**
```bash
python export.py \
  --config configs/chimera_s_512.yaml \
  --weights runs/chimera/chimera_best.pt \
  --output exports/chimera.onnx
```

**Arguments:**
- `--config PATH` (required): Path to training config YAML
- `--weights PATH` (required): Path to model checkpoint
- `--output PATH`: Output ONNX file path (default: `exports/model.onnx`)
- `--img-size INT`: Input image size (default: 512)
- `--opset INT`: ONNX opset version (default: 12)

**Examples:**
```bash
# Export to ONNX
python export.py \
  --config configs/chimera_s_512.yaml \
  --weights runs/chimera/chimera_best.pt \
  --output exports/chimera.onnx

# Custom image size
python export.py \
  --config configs/chimera_s_512.yaml \
  --weights runs/chimera/chimera_best.pt \
  --output exports/chimera_640.onnx \
  --img-size 640
```

---

## UI

### Integrated console (`serve.py --ui`)

```bash
python serve.py --weights runs/chimera/chimera_best.pt --ui --ui-auth admin:change-me
# → http://127.0.0.1:8000/ui
```

Tabs: **Detect** (upload / folder, presets, annotated gallery, JSON export), **Model** (checkpoint switching, class
map, dataset + training metadata), **Training** (curves, validation history), **Benchmark** (live latency/throughput of
the loaded model, saved benchmark reports) and **API & help**. Light and dark themes follow the OS setting.

### Remote console (`python -m ui.app`)

Runs the same look against an API that is already running elsewhere.

```bash
DETEKTOR_UI_BACKEND=http://192.168.1.100:8000 python -m ui.app      # http://127.0.0.1:7860
```

Enter the API key in *Backend connection* if the server sets `DETEKTOR_API_KEY`.

---

## Testing

### Run Tests

**Fast unit tests:**
```bash
python -m unittest tests.test_box_ops tests.test_ciou tests.test_mask_ops
```

**Full test suite:**
```bash
python -m unittest discover -s tests -p "test_*.py"
```

**Specific test class:**
```bash
python -m unittest tests.test_box_ops.TestBoxOps
```

**Specific test method:**
```bash
python -m unittest tests.test_ciou.TestCIoU.test_ciou_loss_identical_boxes
```

---

## Docker Deployment

### Build and Run

**Build image:**
```bash
docker build -t detektor:latest .
```

**Run with Docker Compose:**
```bash
# CPU service
docker-compose up detektor-cpu

# GPU service
docker-compose up detektor-gpu
```

**Manual run:**
```bash
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/artifacts \
  -e DETEKTOR_WEIGHTS=/artifacts/model.pt \
  -e DETEKTOR_DEVICE=cpu \
  detektor:latest
```

---

## Common Patterns

### Full Training Pipeline

```bash
# 1. Validate dataset
python check_dataset.py --data-yaml F:/data/data.yaml

# 2. Train model
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml

# 3. Validate model
python validate.py --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml

# 4. Generate report
python -m scripts.report --run-dir runs/chimera

# 5. Package model
python -m scripts.package_model \
  --weights runs/chimera/chimera_best.pt \
  --config configs/chimera_s_512.yaml \
  --data-yaml F:/data/data.yaml
```

### Production Deployment

```bash
# 1. Export to ONNX
python export.py \
  --config configs/chimera_s_512.yaml \
  --weights runs/chimera/chimera_best.pt \
  --output exports/chimera.onnx

# 2. Benchmark performance
python -m scripts.benchmark \
  --weights runs/chimera/chimera_best.pt \
  --onnx exports/chimera.onnx \
  --source F:/data/test/images

# 3. Deploy with Docker
docker-compose up detektor-gpu
```

---

For more details, see the main [README](../../README.md) and [Quick Start Guide](../guides/QUICKSTART.md).



