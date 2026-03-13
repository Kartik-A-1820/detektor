# Changelog

All notable changes to Detektor are documented in this file.

## [Unreleased] - March 2026

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

