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

__RESULTS__

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
| `robustness` | mAP50 retention under noise, blur, brightness/contrast, JPEG, downscaling | Uses the `e2e` model or `--weights/--data-yaml` |
| `accuracy` | P/R/F1/mAP50/mAP50‑95/IoU on **your** dataset via `validate.py` | `--weights` + `--data-yaml` required |

`fast` = every suite except `api`, `e2e`, `robustness`, `accuracy`. `all` = everything.

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
