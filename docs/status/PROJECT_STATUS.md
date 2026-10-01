# Project status

_Last reviewed with release 1.1.0._

## Maturity

Detektor is a **solid foundation for local and small-team deployments** with production plumbing around it: a tested,
hardened serving layer, reproducible packaging, CI, containerisation and a benchmark framework. It is **not** a
drop-in replacement for large, pretrained detector families — see *Known limitations*.

| Area | State | Evidence |
| --- | --- | --- |
| Training pipeline | Stable; hardware-aware auto-config, retries, EMA/AMP, resume | `tests/test_auto_train_config.py`, `tests/test_smart_training.py`, synthetic end-to-end benchmark |
| Evaluation | P/R/F1, AP50, mAP50, AP50-95, IoU, confusion matrix, threshold sweep | `tests/test_metrics_helpers.py`, `validate.py` |
| Serving API | Auth, limits, metrics, hot-swap, probes | `tests/test_api.py`, `tests/test_api_security.py`, `tests/test_upload_validation.py` |
| Web console | Redesigned; unit-tested render layer; verified in a real browser (light + dark) | `tests/test_ui.py` |
| Packaging / export | Manifest + SHA-256, ONNX export with parity check | `tests/test_export.py`, `tests/test_run_artifacts.py` |
| Benchmarks | 11 suites, report + regression compare | `tests/test_benchmarks.py`, [BENCHMARKS](../BENCHMARKS.md) |
| CI | Lint, tests (3.11/3.12), benchmark smoke, Docker build | `.github/workflows/ci.yml` |
| Docker / Kubernetes | Image + compose + manifests provided; image built in CI | [DEPLOYMENT](../guides/DEPLOYMENT.md) |

## Good for today

- Training and serving custom detectors/segmenters on a workstation or a small server
- Rapid experimentation with a hardware-aware training loop
- Single-GPU or CPU inference behind an authenticated API, with Prometheus monitoring
- Reproducible benchmarking of architecture/hardware trade-offs

## Known limitations

- **Model quality** — recall/precision on difficult real-world datasets is the main open problem; no pretrained weights
  or model zoo are shipped. Always evaluate on your own data first.
- No TensorRT runtime; ONNX Runtime export is supported (and markedly faster than eager PyTorch on CPU).
- No distributed / multi-GPU training.
- Single model execution per process; scale horizontally. No built-in TLS or rate limiting.
- Published benchmark numbers are CPU-only from one shared VM; GPU baselines are welcome contributions.
- Class-aware NMS only (no class-agnostic option yet): a weak model can emit overlapping boxes of different classes on
  one object.
- When masks are requested, post-processing cost grows with the number of detections because each mask is resized to the
  original image resolution (see the *dense* scenarios in [BENCHMARKS](../BENCHMARKS.md)); the API skips masks unless
  `include_masks=true`. Cropped per-instance resizing is a planned optimisation.

## Roadmap

Near term
- Pretrained checkpoints / model zoo and a documented fine-tuning recipe
- Class-agnostic NMS option and faster (cropped, per-instance) mask post-processing
- GPU benchmark baselines (CUDA, ONNX Runtime CUDA)
- Optional rate limiting and per-key identities in the API

Later
- TensorRT runtime
- Multi-GPU / distributed training
- Richer dataset loaders (COCO JSON, Pascal VOC)
- Quantisation (INT8) benchmarks
