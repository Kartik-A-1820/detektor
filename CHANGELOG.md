# Changelog

All notable changes to Detektor are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/)
and the project aims to follow [Semantic Versioning](https://semver.org/).

## [1.1.0] - Production hardening release

### Added
- **Benchmark framework** (`python -m benchmarks`): eleven suites — complexity, latency (with decode/forward/postprocess
  breakdown), batch throughput, memory, training throughput, cold start, ONNX Runtime vs PyTorch, HTTP API load test,
  synthetic end-to-end train+validate, robustness to image corruptions, and accuracy on user data — plus a Markdown
  reporter and a regression `compare` command. Published baseline in `benchmarks/results/`, methodology in
  `docs/BENCHMARKS.md`.
- **Redesigned web console**: tabs for Detect / Model / Training / Benchmark / API help, light and dark themes,
  class-consistent colour-blind-safe annotations, presets (Balanced / High precision / High recall), KPI cards, JSON
  export, live latency benchmark, saved-report viewer, remote-backend mode with API-key support.
- **API security & operations**: optional API-key auth (`X-API-Key` / Bearer, constant-time compare), CORS allow-list,
  request-body limit (413), security headers, client `X-Request-ID` propagation, Prometheus endpoint
  (`/metrics/prometheus`) with a latency histogram, `--max-concurrency`, console login (`--ui-auth`).
- Synthetic dataset generator (`python -m benchmarks.synthetic`) for trying the full pipeline without data.
- SHA-256 checksums in packaged-model manifests.
- Docs: architecture, deployment (Docker, nginx, systemd, Kubernetes), configuration reference, security policy,
  split task-focused guides; GitHub Actions CI (lint, tests, benchmark smoke, Docker build), issue/PR templates,
  Dependabot, pre-commit, Makefile, `pyproject.toml` tooling config, code of conduct.

### Changed
- **Performance:** the inference service skips mask composition and full-resolution resizing unless `include_masks` is
  requested (the default is off). On a 1280×720 image with 100 detections this cuts post-processing from ~230 ms to
  ~25 ms (see `benchmarks/results/`).
- **Dockerfile**: multi-stage, CPU-only PyTorch by default (GPU via build arg), non-root user, healthcheck,
  `opencv-python-headless`. **docker-compose**: env-file driven, read-only filesystem, dropped capabilities, GPU profile.
- Inference now runs in a worker thread (the event loop is no longer blocked by model execution).
- Multipart uploads without a part `Content-Type` (or `application/octet-stream`) are accepted; content is verified by
  decoding. Explicitly wrong types are still rejected.
- The UI console binds to `127.0.0.1` by default (was `0.0.0.0`) and the in-process UI may only serve files from the run
  directory (previously the whole working directory).
- `requirements.txt` now pins sensible lower bounds and `gradio>=4.44,<7`; added `pillow`, `pandas`, `psutil`.
- Python source cleaned with `ruff` (whitespace, import order, unused names).

### Fixed
- **Instance masks covered almost the whole image.** `ChimeraODIS.predict` cropped mask *logits* to 0 outside the box,
  then applied `sigmoid` (→ 0.5) and thresholded with `>= 0.5`, so every pixel outside every box counted as foreground
  (mean mask IoU 0.04 on a trained model; **0.75** after the fix, same checkpoint). Masks are now cropped in probability
  space. Affects `/v1/predict?include_masks=true`, `infer.py`, `validate.py` mask metrics and the console's mask overlay.
- **`train.py` best-checkpoint selection (default, no `--run-val`):** `chimera_best.pt` was selected by epoch training
  loss, which is not comparable across epochs — the box-loss weight ramps over the first 3 epochs, so epoch 1 always
  "won" and the best checkpoint stayed at an essentially untrained snapshot. Warm-up epochs are now excluded from
  loss-based selection (found by the new synthetic end-to-end benchmark, which scored 0 mAP on it).
- Models for profiles with a non-default `proto_k` (e.g. `firefly`) could not be loaded from checkpoints that embed
  `model_config`, because the loader's default `proto_k=24` overrode the embedded value.
- ONNX export failed on PyTorch ≥ 2.9 (default dynamo exporter needs `onnxscript`); export now pins the TorchScript path.
- Decompression-bomb guard: oversized images are rejected from the header before pixel decoding.
- `PredictionResponse.masks` rejected `null` entries; `compute_ap50` was missing from `utils.metrics_helpers`;
  report generation crashed when the epoch summary used `avg_loss` instead of `epoch_loss`.
- Ten previously failing tests (stale expectations and the issues above); the suite is green (190+ tests).
- `.gitignore` ignored the `datasets/` Python package; compiled `__pycache__` files were tracked.

## [Unreleased]

_Nothing yet._

## [1.0.0-rc] - March 2026 (production cleanup & hardening)

### Production Cleanup & Hardening

**CLI Improvements:**
- Standardized help messages across all scripts
- Consistent argument naming and defaults
- Improved error messages with actionable guidance
- Unified output folder naming conventions

**Documentation:**
- Added comprehensive tool reference section
- Created "golden path" example from training to serving
- Improved README organization and clarity
- Added dataset validation documentation

**Code Quality:**
- Consistent logging format across all modules
- Improved error handling and validation
- Better default configurations for safety
- Enhanced backwards compatibility

**New Features:**
- Dataset validation tool (`check_dataset.py`)
- Comprehensive test suite (unit, integration, regression)
- Model artifact packaging (`scripts.package_model`)
- ONNX Runtime benchmarking (`scripts.benchmark`)
- Production-grade FastAPI serving layer
- Gradio UI for local testing

**Testing:**
- 77+ new tests covering core functionality
- Unit tests for box ops, CIoU, mask ops, schemas, config
- Integration tests for inference, API, reporting, validation
- Regression tests for schema stability and no-NaN guarantees

## [1.0.0] - March 2026

### Core Features

**Training & Optimization:**
- AdamW optimizer with cosine warmup scheduler (default)
- SGD with Nesterov momentum (alternative)
- CIoU + BCE loss baseline with numerical safeguards
- AMP support with `torch.amp` API
- Gradient clipping and EMA support
- Resume training with full state restoration
- Detailed loss component logging (CSV + JSONL)

**Inference & Deployment:**
- Auto-detection of `num_classes` from checkpoints
- Folder inference for batch processing
- FastAPI service with versioned endpoints
- Class name support from dataset YAML
- ONNX export with stable tensor outputs

**Validation & Reporting:**
- Production-grade validation metrics (AP50, AP50-95)
- Confusion matrix and threshold sweep
- Ultralytics-style reporting with plots
- Comprehensive metrics summaries

**Task Modes:**
- Detection mode (bounding boxes only)
- Segmentation mode (boxes + masks)
- Auto-detection from dataset format

**Dataset Support:**
- YOLO/Roboflow format compatibility
- Auto-configuration from dataset YAML
- Multiple image formats (jpg, png, bmp, webp)
- Task-aware training (detect vs segment)

### Architecture

**Model:**
- ChimeraODIS: Lightweight detection + segmentation
- Optimized for modest hardware (GTX 1650 Ti 4GB)
- Prototype-based instance segmentation
- Anchor-free detection head

**Loss Components:**
- Classification: BCE
- Box regression: CIoU
- Objectness: BCE
- Segmentation: BCE + Dice

### Documentation

- Comprehensive README with examples
- Optimizer and loss stability guide
- Reporting module documentation
- Validation output schema specification
- Project status and roadmap
- Contributing guidelines

---

## Version History

- **v1.0.0** (March 2026): Initial production release
- **Unreleased**: Ongoing improvements and cleanup

