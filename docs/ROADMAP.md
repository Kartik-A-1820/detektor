# Roadmap

Detektor's primary use case is **a person training and validating detectors on their own laptop, with good visuals**.
Priorities follow from that: first make training faster and models more accurate on modest hardware, then make results
easier to *see* and understand. Effort: **S** days · **M** 1–2 weeks · **L** several weeks. Impact figures are
estimates, not measurements — every accuracy item is to be validated against the real-data baseline below before it is
claimed.

Legend: ✅ done · 🚧 in progress · ⬜ planned

## 0. Foundations

| | Item | Notes |
| --- | --- | --- |
| ✅ | **COCO-faithful metrics** | `validate.py` reports pooled mAP50 / mAP75 / mAP50-95, size-bucketed AP and AR@100 via `pycocotools`. The old per-image AP50-95 overstated by ~6 % on the synthetic benchmark (kept as `*_legacy`). |
| ✅ | **Real-data baseline harness** | `python -m benchmarks run --suites real` downloads a public dataset on demand (checksum-pinned, safe extraction) and trains/validates on it. Reference result (from scratch, 400 photos, 20 CPU epochs): mAP50 0.218 / mAP50-95 0.091 — see [BENCHMARKS.md](BENCHMARKS.md). Pass your own `--data-yaml` to benchmark private data. |
| ✅ | **Stable box regression** | The box head predicted raw pixel distances through a ReLU and collapsed on datasets with large objects (mAP exactly 0 on the real baseline). Distances are now predicted in stride units via `softplus`; the same run reaches mAP50 0.218, and the synthetic benchmark's AP50-95 rose from 0.785 to 0.952. Legacy checkpoints are unaffected. |
| ✅ | **Both YOLO dataset layouts** | Roboflow (`<split>/images`) and YOLOv5/v8 (`images/<split>`), the YAML `path:` key, and YAMLs with `names` but no `nc`. |
| ⬜ S | **Experiment log** | One JSON per run (config, metrics, timings, environment) so runs can be compared; prerequisite for the comparison UI. |

## A. Performance (training and validation speed on a laptop)

| | Item | Why it matters | Effort |
| --- | --- | --- | --- |
| ⬜ | Apple Silicon (MPS) support and CPU tuning | `device=auto` only knows CUDA or CPU, so every Mac falls back to slow CPU. Includes `channels_last`, CPU bf16 autocast, thread settings. | M |
| ⬜ | Dataset caching (RAM / disk of resized images) | JPEG decode + resize every epoch is a large share of CPU-bound training time (≈2× or more for small datasets). | S |
| ⬜ | Faster data loading | Persistent workers, per-OS worker defaults (Windows is fragile), cheaper augmentation order. | S |
| ⬜ | Rectangular training/inference with letterbox | Images are stretched to a square today; letterboxing keeps aspect ratio and avoids padding compute. | M |
| ⬜ | Lower-memory modes | Gradient checkpointing, 4 GB-GPU preset, automatic image-size fallback after OOM. | M |
| ⬜ | Early stopping and time-budget training | "Train for 30 minutes" / "stop when validation mAP stalls". | S |
| ⬜ | Training-time estimate before you start | Use the benchmark framework to say "≈40 min on this machine". | S |

## B. Accuracy

| | Item | Expected impact | Effort |
| --- | --- | --- | --- |
| ⬜ | **Pretrained backbones (transfer learning)** | Biggest lever for small datasets; permissively licensed ImageNet backbones (e.g. torchvision MobileNet / EfficientNet) into our neck and head. | M–L |
| ⬜ | Stronger default augmentation | Mosaic / copy-paste / scale jitter are off by default; turn mosaic off for the last epochs. | M |
| ⬜ | EMA on by default, better LR recipe | Cheap and reliable. | S |
| ⬜ | Better label assignment and classification loss | Task-aligned assigner + quality-/varifocal loss. Largest model-side gain, riskiest. | L |
| ⬜ | Small-object support | Optional stride-4 head, tiled (sliced) inference, test-time augmentation. | M |
| ⬜ | Class-imbalance tools | Class weights, rare-class oversampling, focal-loss option. | S–M |
| ⬜ | Recommended threshold | Best-F1 confidence stored in the checkpoint and used as the inference default (fixes "too many boxes"). | S |
| ⬜ | Dataset quality checks | Likely-mislabeled images, duplicates, degenerate boxes, using model disagreement. | M |
| ⬜ | K-fold cross-validation, stratified split tool | Small datasets give noisy single-split scores. | M |
| ⬜ | COCO JSON / Pascal VOC import | Fewer conversion mistakes for non-YOLO data. | S–M |

## C. UI and visual features

| | Item | Effort |
| --- | --- | --- |
| ⬜ | Live training dashboard: curves, ETA, resource use, per-epoch prediction previews on fixed validation images | M |
| ⬜ | Validation report viewer: interactive PR / F1-confidence curves, clickable confusion matrix, per-class AP, size-bucket AP | M |
| ⬜ | False-positive / false-negative galleries with ground truth overlay, filterable by class and confidence | M |
| ⬜ | Live threshold sliders that recompute precision/recall and redraw without rerunning inference | S–M |
| ⬜ | Run comparison: overlaid curves, metric diffs, checkpoint A/B on the same image | M |
| ⬜ | Dataset explorer: class balance, box size/aspect histograms, box-location heatmap, labelled gallery with `check_dataset` issues | M |
| ⬜ | Augmentation preview | S |
| ⬜ | Start training from the UI with hardware-aware recommendations; desktop notification on completion | M |
| ⬜ | Exports: annotated images, CSV / COCO JSON predictions, HTML/PDF validation report with an auto-generated model card | S–M |
| ⬜ | First-run wizard (dataset → check → train → validate) with a one-click sample dataset | S–M |

## D. Enablers

| | Item | Effort |
| --- | --- | --- |
| ⬜ | `detektor/` package, `pip install detektor`, one `detektor` CLI (train / val / predict / ui) | M–L |
| ⬜ | Windows and macOS in CI (only Linux is tested today) | S–M |
| ⬜ | Notebook-friendly Python API (`Detector("best.pt").predict(...)`) | M |

## Suggested order

1. ✅ Foundations: COCO-faithful metrics and the real-data baseline.
2. Apple MPS, dataset caching, letterbox (A).
3. Pretrained backbones, stronger default augmentation, EMA by default (B) — expect the largest accuracy gain.
4. Recommended threshold, the validation report viewer, FP/FN galleries.
5. Live training dashboard and run comparison, then packaging (D).

## Later / out of scope for now

Server-side concerns (multi-user auth, fleet management, TensorRT, non-detection tasks such as pose or oriented boxes)
matter for a hosted product but not for a single-laptop workflow; they are deliberately deferred. See the discussion of
competitors in the project history for the longer product view.
