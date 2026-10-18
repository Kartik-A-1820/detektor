# Benchmarks

Detektor ships a benchmark framework (`benchmarks/`) covering speed, memory, size, quality, robustness and serving
capacity. Everything is one CLI, writes machine-readable JSON plus a Markdown report, and can diff two runs to catch
regressions.

```bash
python -m benchmarks list                                   # suites + profiles
python -m benchmarks run --suites fast --profiles firefly,comet,nova
python -m benchmarks run --suites all --profiles all --tag full
python -m benchmarks report runs/benchmarks/<run>/results.json -o report.md
python -m benchmarks compare base/results.json new/results.json --threshold 15 --fail-on-regression
```

## Published baseline (CPU)

Produced by `python -m benchmarks run --suites all --profiles all` — see [Reproducing the published baseline](#reproducing-the-published-baseline). Raw data: [`benchmarks/results/baseline-cpu.json`](../benchmarks/results/baseline-cpu.json).

### Environment

| Key | Value |
|---|---|
| timestamp_utc | 2026-10-01T18:07:43+00:00 |
| git_revision | 58a9631 |
| cpu_model | Intel(R) Xeon(R) Processor @ 2.10GHz |
| cpu_cores_physical | 4 |
| cpu_cores_logical | 4 |
| ram_gb | 15.72 |
| platform | Linux-6.18.44-fc-v50-x86_64-with-glibc2.39 |
| python | 3.11.15 |
| torch | 2.14.1+cu130 |
| torch_threads | 4 |
| onnxruntime | 1.30.0 |
| device | cpu |

### Configuration

```json
{
  "suites": [
    "complexity",
    "latency",
    "throughput",
    "memory",
    "training",
    "startup",
    "onnx",
    "api",
    "e2e",
    "robustness",
    "accuracy",
    "real"
  ],
  "profiles": [
    "firefly",
    "comet",
    "nova",
    "pulsar",
    "quasar",
    "supernova"
  ],
  "device": "cpu",
  "img_sizes": [
    320,
    512
  ],
  "img_size": 512,
  "batch_sizes": [
    1,
    2,
    4,
    8
  ],
  "warmup": 5,
  "runs": 30,
  "seed": 0,
  "quick": false,
  "note": "performance suites (complexity..api) from the 1.1 baseline run; e2e, robustness and real re-run on the final 1.1 code (COCO-style metrics, stride box decoding) on the same machine"
}
```

### Model complexity

_Parameters, GFLOPs and model size per architecture profile and input size_ — ran in 3.0 s

| Profile | Input | Params (M) | GFLOPs | FP32 size (MB) | FP16 size (MB) |
|---|---|---|---|---|---|
| firefly | 320 | 1.40 | 1.70 | 5.32 | 2.66 |
| firefly | 512 | 1.40 | 4.36 | 5.32 | 2.66 |
| comet | 320 | 2.47 | 2.83 | 9.43 | 4.71 |
| comet | 512 | 2.47 | 7.24 | 9.43 | 4.71 |
| nova | 320 | 4.70 | 5.69 | 17.93 | 8.96 |
| nova | 512 | 4.70 | 14.57 | 17.93 | 8.96 |
| pulsar | 320 | 9.13 | 10.28 | 34.81 | 17.41 |
| pulsar | 512 | 9.13 | 26.32 | 34.81 | 17.41 |
| quasar | 320 | 9.61 | 10.38 | 36.66 | 18.33 |
| quasar | 512 | 9.61 | 26.56 | 36.66 | 18.33 |
| supernova | 320 | 15.00 | 16.72 | 57.22 | 28.61 |
| supernova | 512 | 15.00 | 42.80 | 57.22 | 28.61 |

### Inference latency

_Batch-1 latency with preprocess / forward / postprocess breakdown (p50/p95/p99)_ — ran in 269.4 s

| Profile | Input | Decode+resize p50 | Forward p50 | Predict p50 | Predict p95 | Predict p99 | Dense + masks p50 | Dense, boxes only p50 | FPS |
|---|---|---|---|---|---|---|---|---|---|
| firefly | 320 | 6.32 | 29.48 | 33.82 | 50.84 | 67.63 | 272.84 | 50.43 | 27.34 |
| firefly | 512 | 5.44 | 47.33 | 44.90 | 55.99 | 62.98 | 313.96 | 69.07 | 21.90 |
| comet | 320 | 5.25 | 36.66 | 30.68 | 37.03 | 47.43 | 263.25 | 58.23 | 31.03 |
| comet | 512 | 5.35 | 63.69 | 61.76 | 90.43 | 116.28 | 351.05 | 86.49 | 15.08 |
| nova | 320 | 5.35 | 50.60 | 50.58 | 62.56 | 65.17 | 274.94 | 72.30 | 19.57 |
| nova | 512 | 5.37 | 87.10 | 86.12 | 114.71 | 125.15 | 366.79 | 111.17 | 10.93 |
| pulsar | 320 | 5.09 | 58.92 | 64.65 | 84.81 | 109.44 | 283.74 | 93.40 | 14.71 |
| pulsar | 512 | 6.10 | 117.70 | 118.39 | 140.83 | 153.39 | 402.89 | 168.46 | 8.29 |
| quasar | 320 | 5.43 | 61.62 | 63.54 | 87.82 | 88.56 | 290.88 | 84.27 | 14.93 |
| quasar | 512 | 5.33 | 129.86 | 124.02 | 170.70 | 181.20 | 394.54 | 136.88 | 7.61 |
| supernova | 320 | 6.25 | 89.15 | 92.20 | 117.35 | 135.34 | 324.73 | 119.44 | 10.64 |
| supernova | 512 | 5.38 | 175.79 | 178.06 | 208.47 | 211.47 | 473.10 | 218.90 | 5.55 |

_All latencies in ms, batch 1, 1280×720 source image. “Dense” lowers the confidence threshold to 0.001 so top‑k, NMS and (for the masks column) full‑resolution mask composition run at their worst‑case workload — 100 detections; a randomly initialised model emits no confident detections, so the default column shows the idle case. The API skips mask computation unless `include_masks=true`._

### Batch throughput

_Images/second versus batch size (forward pass)_ — ran in 101.9 s

| Profile | Input | Batch | Batch latency (ms) | ms / image | Images / s |
|---|---|---|---|---|---|
| firefly | 512 | 1 | 55.85 | 55.85 | 17.91 |
| firefly | 512 | 2 | 72.79 | 36.40 | 27.48 |
| firefly | 512 | 4 | 120.85 | 30.21 | 33.10 |
| firefly | 512 | 8 | 271.84 | 33.98 | 29.43 |
| comet | 512 | 1 | 58.58 | 58.58 | 17.07 |
| comet | 512 | 2 | 82.10 | 41.05 | 24.36 |
| comet | 512 | 4 | 168.86 | 42.22 | 23.69 |
| comet | 512 | 8 | 332.95 | 41.62 | 24.03 |
| nova | 512 | 1 | 92.49 | 92.49 | 10.81 |
| nova | 512 | 2 | 137.36 | 68.68 | 14.56 |
| nova | 512 | 4 | 267.57 | 66.89 | 14.95 |
| nova | 512 | 8 | 556.25 | 69.53 | 14.38 |
| pulsar | 512 | 1 | 118.84 | 118.84 | 8.41 |
| pulsar | 512 | 2 | 196.85 | 98.42 | 10.16 |
| pulsar | 512 | 4 | 446.04 | 111.51 | 8.97 |
| pulsar | 512 | 8 | 853.80 | 106.73 | 9.37 |
| quasar | 512 | 1 | 130.53 | 130.53 | 7.66 |
| quasar | 512 | 2 | 194.63 | 97.32 | 10.28 |
| quasar | 512 | 4 | 415.20 | 103.80 | 9.63 |
| quasar | 512 | 8 | 850.35 | 106.29 | 9.41 |
| supernova | 512 | 1 | 167.10 | 167.10 | 5.98 |
| supernova | 512 | 2 | 324.42 | 162.21 | 6.16 |
| supernova | 512 | 4 | 667.97 | 166.99 | 5.99 |
| supernova | 512 | 8 | 1,465.08 | 183.13 | 5.46 |

### Memory footprint

_Peak RAM (CPU, RSS) or VRAM (CUDA, allocated) for inference and a training step, one fresh process per profile_ — ran in 41.0 s

| Profile | Input | Weights (MB) | Inference peak (MB) | Inference Δ (MB) | Train batch | Train peak (MB) | Train Δ (MB) |
|---|---|---|---|---|---|---|---|
| firefly | 512 | 5.32 | 900.50 | 155.30 | 4 | 1,470.20 | 696.30 |
| comet | 512 | 9.43 | 913.40 | 158.50 | 4 | 1,624.20 | 840.60 |
| nova | 512 | 17.93 | 928.90 | 157.90 | 4 | 1,837.20 | 1,038.00 |
| pulsar | 512 | 34.81 | 952.60 | 158.00 | 4 | 2,114.90 | 1,302.70 |
| quasar | 512 | 36.66 | 954.30 | 159.80 | 4 | 2,128.10 | 1,313.60 |
| supernova | 512 | 57.22 | 980.00 | 161.90 | 4 | 2,483.50 | 1,651.90 |

_Metric: process RSS (MB), fresh process per profile. “Peak” includes the interpreter and PyTorch libraries (~703 MB); “Δ” is the increase over the level just before the measured work._

### Training throughput

_Training step time and images/second (synthetic batch, AdamW)_ — ran in 47.2 s

| Profile | Input | Batch | Step p50 (ms) | Step p95 (ms) | Images / s | Loss finite |
|---|---|---|---|---|---|---|
| firefly | 320 | 4 | 307.85 | 322.52 | 13.08 | True |
| comet | 320 | 4 | 370.07 | 434.33 | 10.46 | True |
| nova | 320 | 4 | 570.36 | 739.05 | 6.83 | True |
| pulsar | 320 | 4 | 748.18 | 866.48 | 5.24 | True |
| quasar | 320 | 4 | 698.82 | 776.50 | 5.66 | True |
| supernova | 320 | 4 | 1,140.91 | 1,197.90 | 3.58 | True |

### Cold start

_Checkpoint load time, first-request penalty and warm latency_ — ran in 7.9 s

| Profile | Checkpoint (MB) | Load (ms) | 1st inference (ms) | 2nd inference (ms) | Warm‑up penalty (ms) |
|---|---|---|---|---|---|
| firefly | 5.51 | 95.29 | 70.01 | 59.69 | 10.32 |
| comet | 9.62 | 172.02 | 94.80 | 72.50 | 22.30 |
| nova | 18.14 | 106.61 | 123.68 | 101.22 | 22.46 |
| pulsar | 35.08 | 148.17 | 149.91 | 159.21 | -9.31 |
| quasar | 36.92 | 180.66 | 142.08 | 145.43 | -3.35 |
| supernova | 57.54 | 176.34 | 215.92 | 196.93 | 18.99 |

### ONNX Runtime vs PyTorch

_ONNX export, numerical parity and ONNX Runtime vs PyTorch latency_ — ran in 45.4 s

| Profile | Input | Export (s) | ONNX (MB) | Max |Δ| vs PyTorch | PyTorch p50 (ms) | ORT p50 (ms) | Speed‑up |
|---|---|---|---|---|---|---|---|
| firefly | 512 | 0.63 | 5.36 | 6.7e-08 | 55.11 | 17.48 | 3.15 |
| comet | 512 | 0.61 | 9.46 | 6.7e-08 | 60.71 | 23.19 | 2.62 |
| nova | 512 | 0.68 | 17.95 | 6.7e-08 | 84.69 | 39.93 | 2.12 |
| pulsar | 512 | 0.86 | 34.81 | 6.7e-08 | 137.32 | 75.38 | 1.82 |
| quasar | 512 | 0.98 | 36.65 | 6.7e-08 | 130.92 | 73.65 | 1.78 |
| supernova | 512 | 1.35 | 57.20 | 4.8e-07 | 186.63 | 118.41 | 1.58 |

### HTTP API load test

_Requests/second and latency percentiles of /v1/predict under concurrent clients_ — ran in 44.0 s

Profile `firefly`, input 512px, device `cpu`, payload synthetic 1280x720 JPEG, server concurrency 1.

| Clients | Requests | Errors | Req / s | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|---|
| 1 | 80 | 0 | 8.91 | 110.16 | 132.49 | 159.75 |
| 2 | 80 | 0 | 9.88 | 200.73 | 233.81 | 259.30 |
| 4 | 80 | 0 | 9.55 | 411.03 | 467.65 | 473.58 |
| 8 | 80 | 0 | 9.47 | 829.23 | 966.09 | 991.23 |
| 16 | 80 | 0 | 10.02 | 1,538.97 | 1,645.18 | 1,710.18 |

### End-to-end quality (synthetic data)

_Train + validate on a seeded synthetic dataset (pipeline correctness and learning signal)_ — ran in 678.7 s

Profile `firefly`, 192 train / 48 val synthetic images at 256px, 30 epochs on `cpu` — 672.1 s total (8.57 img/s).

| Precision | Recall | F1 | mAP50 | mAP75 | mAP50‑95 | Mean box IoU | Mean mask IoU | Images |
|---|---|---|---|---|---|---|---|---|
| 0.5938 | 1.00 | 0.7451 | 1.00 | 0.9866 | 0.9517 | 0.9548 | 0.7586 | 48 |

_mAP is COCO‑style (pycocotools): pooled over the dataset with 101‑point interpolation._

### Robustness to corruptions

_mAP50 retention under Gaussian noise, blur, brightness/contrast, JPEG and downscaling_ — ran in 14.8 s

| Perturbation | mAP50 | Retention vs clean | Mean detections |
|---|---|---|---|
| clean | 1.00 | 1.00 | 4.06 |
| gaussian_noise_s10 | 1.00 | 1.00 | 4.19 |
| gaussian_noise_s25 | 1.00 | 1.00 | 4.19 |
| blur_k5 | 1.00 | 1.00 | 4.12 |
| blur_k9 | 1.00 | 1.00 | 4.10 |
| brightness_x0.6 | 1.00 | 1.00 | 4.25 |
| brightness_x1.4 | 1.00 | 1.00 | 4.31 |
| contrast_x0.5 | 1.00 | 1.00 | 4.25 |
| jpeg_q20 | 1.00 | 1.00 | 4.06 |
| downscale_x0.5 | 1.00 | 1.00 | 4.10 |

_48 images at 256px. mAP50 is COCO-style (pycocotools) at IoU 0.5, computed in-process._

### Accuracy on user dataset

_Precision/recall/mAP50/mAP50-95/IoU on a dataset split via validate.py (needs --weights and --data-yaml)_ — ran in 0.0 s

_Skipped: pass --weights and --data-yaml to enable the accuracy suite_

### Real-data baseline

_Train on a public (or your own) real dataset and report COCO-style mAP, per-class AP and training time_ — ran in 567.9 s

Dataset `african-wildlife` (AGPL-3.0 (as distributed by Ultralytics)), profile `firefly`, 400 train / 225 val images at 320px, 20 epochs on `cpu` — 558.2 s. trained from scratch (no pretrained weights); a short CPU run, so treat as a relative baseline

| Precision | Recall | F1 | mAP50 | mAP75 | mAP50‑95 | Mean box IoU | Mean mask IoU | Images |
|---|---|---|---|---|---|---|---|---|
| 0.1809 | 0.504 | 0.2662 | 0.218 | 0.0519 | 0.0912 | 0.7208 | 0 | 225 |

_mAP is COCO‑style (pycocotools): pooled over the dataset with 101‑point interpolation._

| Class | AP50 | Precision | Recall |
|---|---|---|---|
| buffalo | 0.2943 | 0.2624 | 0.5955 |
| elephant | 0.0304 | 0.1667 | 0.0659 |
| rhino | 0.226 | 0.1739 | 0.6588 |
| zebra | 0.3214 | 0.1532 | 0.6667 |

## Suites

| Suite | Measures | Notes |
| --- | --- | --- |
| `complexity` | Parameters, GFLOPs/GMACs, FP32/FP16 size | Exact FLOP counts from `torch.utils.flop_counter` |
| `latency` | Batch‑1 p50/p90/p95/p99, FPS; **decode+resize / forward / postprocess** breakdown | Plus a *dense* worst‑case scenario (see below) |
| `throughput` | Images/s and ms/image vs batch size | Forward pass |
| `memory` | Peak RSS (CPU) or VRAM (CUDA) for inference and a training step | Sampled every 4 ms / `max_memory_allocated` |
| `training` | Step time and images/s (forward + loss + backward + AdamW) | Synthetic batch; also asserts the loss stays finite |
| `startup` | Checkpoint load time, first‑request penalty, file size | Cold‑start planning |
| `onnx` | Export time, ONNX size, numerical parity, ONNX Runtime vs PyTorch latency | CPU provider |
| `api` | Req/s and p50/p95/p99 vs concurrent clients against a **real in‑process uvicorn server** | Includes auth + the full HTTP/decode/serialise path |
| `e2e` | Seeded synthetic dataset → real `train.py` → real `validate.py` | Proves the whole pipeline learns; not a real‑world accuracy claim |
| `real` | **Real photographs**: downloads a public dataset on demand (`python -m benchmarks datasets list`), trains from scratch with the real `train.py`, validates with COCO‑style metrics | The baseline to measure accuracy changes against; `--real-dataset`, `--real-epochs`, `--real-max-train`, or your own `--data-yaml` |
| `robustness` | mAP50 retention under noise, blur, brightness/contrast, JPEG, downscaling | Uses the `e2e` model or `--weights/--data-yaml` |
| `accuracy` | P/R/F1, COCO‑style mAP50 / mAP75 / mAP50‑95, size‑bucketed AP and IoU on **your** dataset via `validate.py` | `--weights` + `--data-yaml` required |

`fast` = every suite except `api`, `e2e`, `robustness`, `accuracy`, `real`. `all` = everything (`accuracy` skips itself without `--weights`/`--data-yaml`; `real` downloads its dataset and skips cleanly when offline).

### Common options

| Option | Default | |
| --- | --- | --- |
| `--profiles` | `firefly,comet,nova` | Comma list or `all` (six profiles) |
| `--device` | `auto` | `cpu`, `cuda` |
| `--img-sizes` / `--img-size` | `320,512` / largest | Latency/complexity sizes; primary size for throughput/memory |
| `--batch-sizes` | `1,2,4,8` | Throughput sweep |
| `--warmup` / `--runs` | `5` / `30` | Untimed / timed iterations |
| `--threads` | library default | `torch.set_num_threads` |
| `--quick` | off | Tiny workloads for CI smoke runs |
| `--weights`, `--data-yaml` | – | For `accuracy` / `robustness` on your own data |
| `--e2e-epochs` | 30 (4 with `--quick`) | Synthetic training length |

Output goes to `runs/benchmarks/<timestamp>_<tag>/{results.json,report.md}`; `results.json` is rewritten after every
suite, so a crash never loses finished work. A failing suite is recorded (`status: error`) and the rest still run.

## Methodology (read before quoting numbers)

* **Timing:** `time.perf_counter` around the call, `torch.cuda.synchronize()` on GPU, untimed warm‑up, `gc.collect()`
  before the timed loop. Percentiles use linear interpolation over all timed iterations.
* **Random weights vs trained weights:** speed, size and memory depend on the architecture, not on the learned values,
  so those suites use freshly initialised models (seeded) — you can benchmark any profile without training it.
* **Dense scenarios:** a random network emits near‑uniform scores, so at the default threshold (0.25) *no* candidate
  survives and post‑processing looks free. `predict_dense` lowers the threshold to 0.001 so top‑k, class‑aware NMS and
  full‑resolution mask composition run at their worst‑case workload (100 detections); `predict_dense_boxes` is the same
  workload without masks — what the API does unless `include_masks=true`; `predict_default` is the realistic idle case.
  A trained model on a busy scene sits in between. (This benchmark is how the mask post‑processing cost was found and
  the box‑only fast path justified.)
* **Source image:** latency uses a deterministic textured 1280×720 JPEG so decode + resize cost is realistic and
  reproducible.
* **Shared hosts are noisy.** Run on an idle machine, pin `--threads`, repeat, and compare *relative* changes on the
  same box. Absolute numbers are only comparable across identical hardware.
* **Accuracy metrics are COCO‑style.** `validate.py` and the suites use `pycocotools`: detections are pooled over the
  whole dataset with 101‑point interpolated precision, so numbers are comparable with published results. (Before 1.1 the
  AP50‑95 was an average of per‑image APs that skipped images without predictions or ground truth; it overstated the
  score by ~6 % on the synthetic set. The old values remain available as `*_legacy` fields.) Size buckets (small/medium/
  large = box area < 32², 32²–96², > 96² px) are relative to the evaluated image resolution.
* **Real‑data baseline:** the public datasets used by the `real` suite are downloaded on demand and checksum‑verified;
  they are **not** redistributed here. The registry records each dataset's license (the bundled entries are AGPL‑3.0 as
  distributed by Ultralytics) so you can judge suitability for your own use. Training is from scratch on CPU, so the
  absolute numbers are modest — the point is a *stable reference* for measuring improvements.
* **Synthetic quality benchmark:** three shape classes on noisy textured backgrounds, generated from a seed
  (`benchmarks/synthetic.py`). It validates that data loading, loss, optimisation, checkpointing and evaluation work
  together. Scores say nothing about performance on real data.

## Regression gating

`compare` flattens each run into headline metrics (latency p50/p95, images/s, memory Δ, load time, ORT p50, API req/s
and p95, mAP/recall/precision, robustness mAP) and flags any that moved the wrong way by more than `--threshold`
percent. With `--fail-on-regression` it exits non‑zero, so it can gate a release pipeline on a dedicated, stable
runner. (The bundled GitHub Actions workflow runs a smoke benchmark only — shared runners are too noisy to gate on.)

## Reproducing the published baseline

```bash
python -m benchmarks run --suites all --profiles all --img-sizes 320,512 --batch-sizes 1,2,4,8 \
    --runs 30 --warmup 5 --e2e-epochs 30 --tag baseline-cpu
```

The committed raw data lives in [`benchmarks/results/`](../benchmarks/results/).

## Adding a suite

1. Create `benchmarks/suites/<name>.py` exposing `DESCRIPTION: str` and `run(ctx: BenchContext) -> dict`
   (return `{"skipped": "reason", "rows": []}` when prerequisites are missing).
2. Register it in `benchmarks/suites/__init__.py` (`SUITES`, and `HEAVY` if slow).
3. Add a renderer in `benchmarks/report.py` (`RENDERERS`, `TITLES`) and, if it has headline numbers, extend
   `flatten_metrics` so `compare` can gate on them.
4. Add a fast test in `tests/test_benchmarks.py`.

## Legacy tools

`scripts/benchmark.py` (PyTorch vs ONNX Runtime on real images) and `model_matrix.py` (architecture compatibility and
training sweep per profile) remain available; the new `onnx` and `training` suites cover similar ground with a
uniform report format.
