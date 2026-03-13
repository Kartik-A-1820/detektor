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

## Latest Update: Validation-Driven Best Checkpoints

Date:
- March 13, 2026

User request:
- fix validation/checkpoint mismatch across architecture profiles
- save self-contained `.pt` files with weights plus model architecture
- verify longer training and improve auto-training defaults
- push to `beta`

Files changed:
- [train.py](F:\detektor\train.py)
  - `chimera_best.pt` now tracks validation `mAP50` when `--run-val` is enabled
  - final weights now save as a self-contained checkpoint payload, not raw `state_dict`
- [validate.py](F:\detektor\validate.py)
  - validation now rebuilds the model from checkpoint metadata instead of assuming the YAML architecture still matches
- [utils/checkpoints.py](F:\detektor\utils\checkpoints.py)
  - added checkpoint payload builder with `model_state`, `model_config`, `config`, and training state
- [models/factory.py](F:\detektor\models\factory.py)
  - model reconstruction now prefers embedded checkpoint architecture metadata
- [models/chimera.py](F:\detektor\models\chimera.py)
  - detection-mode predictions now keep a stable `masks` key for validation compatibility
- [utils/auto_train_config.py](F:\detektor\utils\auto_train_config.py)
  - auto-config now chooses longer epoch counts by dataset size and device class

Verification:
- targeted tests passed:
  - `python -m unittest tests.test_auto_train_config tests.test_smart_training tests.test_checkpoints tests.test_predict`
- real 5-epoch verification run:
  - config: `reports/auto_verify_5epoch.yaml`
  - output: `runs/auto_verify_5epoch/`
- epoch-by-epoch validation trend from `runs/auto_verify_5epoch/val_metrics.jsonl`:
  - epoch 1: `P=0.0000` `R=0.0000` `mAP50=0.0000`
  - epoch 2: `P=0.0000` `R=0.0000` `mAP50=0.0000`
  - epoch 3: `P=0.0000` `R=0.0000` `mAP50=0.0000`
  - epoch 4: `P=0.0396` `R=0.0392` `mAP50=0.0020`
  - epoch 5: `P=0.5970` `R=0.1337` `mAP50=0.0848`
- standalone validation after the run:
  - `chimera_best.pt`: still `0.0000` metrics because it was created before the selection fix
  - `chimera_final_weights.pt`: `P=0.7636` `R=0.1678` `mAP50=0.1260`

Git:
- pushed to `origin/beta`
- commit: `077d3f1` `Track best checkpoints by val mAP50 and lengthen auto training`

Current state:
- training is verified to run and save self-contained architecture-aware `.pt` bundles
- validation is now loading the correct architecture from checkpoints
- longer training materially improves detection metrics on the current dataset
- next accuracy work should focus on recall and early-epoch instability, not checkpoint plumbing


