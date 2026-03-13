# VibeCode Handoff

## Operating Constraints

- Activate Python with `cmd /c ".venv\Scripts\activate.bat && python ..."` or `.\.venv\Scripts\python.exe ...`.
- `Activate.ps1` is blocked by local execution policy.
- Push only to `beta`.
- `F:/data/data.yaml` is the active dataset.
- Do not commit generated `artifacts/`, `reports/`, or ad hoc `runs/` outputs unless explicitly asked.

## Verified Baseline Before This Update

- Quickstart steps `2-8` and `10` were verified and pushed.
- Docker step `9` is still not runnable here because `docker` is not installed.
- Training is now numerically stable on this machine in `fp32` on CUDA.
- Auto-training already existed:
  - `python train.py --data-yaml F:/data/data.yaml`
  - hardware-aware runtime selection
  - architecture auto-selection
  - stronger training augmentations
- Last major verified 5-epoch run before this update:
  - GPU: `NVIDIA GeForce GTX 1650 Ti` (`~4 GB`)
  - resolved profile: `Comet`
  - precision: `0.4894`
  - recall: `0.0784`
  - mAP50: `0.0438`
  - mean IoU: `0.7144`
- Earlier commits already pushed to `origin/beta`:
  - `7daca64` `Verify step 2 dataset validation`
  - `9178358` `Verify quickstart steps 3-8 and 10`

## Latest Update: Smart Training Orchestrator

Date:
- March 13, 2026

User request:
- make training self-heal for common runtime failures
- retry with safer settings instead of aborting immediately
- test by running training once
- push to `beta`
- compact this file

Files changed:
- [train.py](F:\detektor\train.py)
  - split training into single-attempt execution plus retry orchestrator
  - added recoverable failure classification for:
    - AMP instability
    - non-finite loss
    - non-finite gradients
    - CUDA / runtime OOM
  - retries now restart cleanly in a new run directory instead of continuing from a contaminated attempt
- [utils/auto_train_config.py](F:\detektor\utils\auto_train_config.py)
  - added `smart_training` defaults
  - added retry planner and attempt-specific output directory generation
  - added resolved-runtime summarizer for retry attempts
- [tests/test_auto_train_config.py](F:\detektor\tests\test_auto_train_config.py)
  - added retry-planner coverage for AMP instability and OOM downgrades
- [tests/test_smart_training.py](F:\detektor\tests\test_smart_training.py)
  - added orchestrator test proving a failed attempt is retried with downgraded config

Smart retry behavior now implemented:
- `amp_instability`
  - disable AMP and restart from scratch in a new attempt directory
- `oom`
  - force `num_workers=0`
  - halve `batch_size`
  - increase `grad_accum` to preserve effective batch as much as possible
  - if already at minimum batch, reduce `img_size`
- `non_finite_loss` / `non_finite_grad`
  - disable AMP when applicable
  - reduce LR
  - soften aggressive augmentations
  - for gradient failures, also reduce batch size
- retries stop when no logical downgrade remains or `max_attempts` is reached

Default smart-training policy now lives in resolved config:

```yaml
smart_training:
  enabled: true
  max_attempts: 4
  min_batch_size: 1
  min_img_size: 256
  min_lr: 0.00025
  non_finite_patience: 2
```

## Verification For This Update

Unit tests:

```powershell
cmd /c ".venv\Scripts\activate.bat && python -m unittest tests.test_auto_train_config tests.test_smart_training"
```

Status:
- verified

Real training run:

```powershell
cmd /c ".venv\Scripts\activate.bat && python train.py --config reports/auto_verify_smoke.yaml --data-yaml F:/data/data.yaml --run-val --val-freq 1"
```

Status:
- verified
- completed on attempt `1/4`
- no smart retry was needed on this smoke run

Observed results:
- epoch 1 loss: `3.324238`
- validation precision: `0.0000`
- validation recall: `0.0000`
- validation mAP50: `0.0000`

Artifacts written:
- `runs/auto_verify_smoke/`

## Current State

- Training orchestration is now materially more fault-tolerant for common runtime failures.
- The major remaining problem is still model quality, not training startup or crash handling.
- If accuracy work continues next, focus on:
  - target assignment quality
  - head calibration / loss weighting
  - longer verified runs after the smart orchestrator changes
