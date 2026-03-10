# Detektor

A lightweight object detection and instance segmentation framework designed for rapid experimentation, edge deployment, and local inference.

This project is **fully vibe-coded**, meaning it was built through iterative collaboration between a human developer and AI coding assistants. In this repository, *vibe coding* means rapidly turning ideas into working code, refining them through repeated experimentation, and then hardening the useful workflows into a practical ML project.

## Project Overview

Detektor is built for local-first ML development and deployment:

- train on YOLO-style datasets
- validate with lightweight metrics
- run image inference locally
- serve predictions through FastAPI
- export stable tensor outputs to ONNX
- prepare for later runtime optimization such as TensorRT

The codebase is intentionally lightweight and practical, with a bias toward single-machine workflows and modest GPUs such as the GTX 1650 Ti 4GB.

## Features

- object detection and instance segmentation model scaffold
- hardened training loop with checkpoints, resume support, AMP, gradient clipping, EMA, and structured logging
- lightweight validation metrics and benchmarking
- ONNX export path with parity utilities
- local FastAPI inference service
- CLI image prediction workflow
- dataset YAML auto-configuration for YOLO / Roboflow-style exports
- CPU-runnable smoke tests and quick verification scripts

## Installation

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Optional CUDA check:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Quick Start

### Training

```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml dataset.yaml
```

### Inference

```bash
python infer.py --weights best.pt --source image.jpg
```

### API

```bash
curl -X POST http://localhost:8000/predict -F "image=@image.jpg"
```

## Training

The main training entrypoint is `train.py`.

Example:

```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml dataset.yaml
```

Supported core options include:

- `--config`
- `--data-yaml`
- `--resume`
- `--weights`
- `--ema`
- `--grad-clip`

## Inference

Run direct image inference:

```bash
python infer.py --weights best.pt --source image.jpg --save-path outputs/pred.jpg
```

Or use the CLI wrapper:

```bash
python cli.py --weights best.pt --source image.jpg --save-path outputs/pred.jpg
```

## API Server

Start the local FastAPI inference service:

```bash
python serve.py --weights best.pt
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction request:

```bash
curl -X POST "http://localhost:8000/predict" -F "image=@image.jpg"
```

## Dataset Format

Detektor supports YOLO / Roboflow-style dataset YAML files without requiring manual edits to the main config.

Example:

```yaml
train: F:/data/train/images
val: F:/data/valid/images
test: F:/data/test/images
nc: 1
names: ['object']
```

When `--data-yaml` is provided, Detektor automatically:

- resolves `train` and `val`
- converts split paths ending in `images` to split roots
- sets `data.format = "yolo"`
- updates `data.num_classes`
- stores class names in memory when present

Expected internal split layout:

```text
train/
  images/
  labels/
valid/
  images/
  labels/
```

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

## Contributing

Contributions are welcome, especially around:

- model optimization
- TensorRT export
- dataset loaders
- training improvements
- performance benchmarking

See `CONTRIBUTING.md` for contribution guidelines.

## License

Detektor is released under the MIT License. See `LICENSE`.

## Credits

- developed through iterative human-AI collaboration
- built as a practical vibe-coded ML engineering project
- optimized for lightweight experimentation and local deployment
