# VibeCode Handoff

## Operating Constraints

- Activate Python with `cmd /c ".venv\Scripts\activate.bat && python ..."` or `.\.venv\Scripts\python.exe ...`.
- `Activate.ps1` is blocked by local execution policy.
- Push only to `beta`.
- `F:/data/data.yaml` is the active dataset.
- Do not commit generated `artifacts/`, `reports/`, or ad hoc `runs/` outputs unless explicitly asked.

- Common activation commands:
  - cmd /c ".venv\Scripts\activate.bat && python train.py --data-yaml F:/data/data.yaml"
  - .\.venv\Scripts\python.exe train.py --data-yaml F:/data/data.yaml
- Git push command, when explicitly requested:
  - git push origin beta

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



## Latest Update: Repo Refactor, Validation Consolidation, and Verified 5-Epoch Run

Date:
- March 13, 2026

User request:
- clean the project layout and move root clutter into a more production-like structure
- ignore generated outputs in git
- remove the duplicate validation path
- run one round of train / validate / test
- push to `beta`

Files changed:
- [README.md](F:\detektor\README.md)
  - updated top-level navigation to the new `docs/` layout
  - updated command examples for moved utility entrypoints
- [docs/README.md](F:\detektor\docs\README.md)
  - added a documentation index
- [docs/guides/QUICKSTART.md](F:\detektor\docs\guides\QUICKSTART.md)
- [docs/reference/TOOLS.md](F:\detektor\docs\reference\TOOLS.md)
- [docs/reference/VALIDATION_OUTPUT_SCHEMA.md](F:\detektor\docs\reference\VALIDATION_OUTPUT_SCHEMA.md)
- [docs/internal/INTEGRATION_SUMMARY.md](F:\detektor\docs\internal\INTEGRATION_SUMMARY.md)
- [docs/internal/REPORTING_IMPLEMENTATION_SUMMARY.md](F:\detektor\docs\internal\REPORTING_IMPLEMENTATION_SUMMARY.md)
  - updated internal documentation links and command references
- [.gitignore](F:\detektor\.gitignore)
  - now ignores generated outputs broadly:
    - `runs/`
    - `artifacts/`
    - `reports/`
    - `.tmp_testdata/`
- [scripts](F:\detektor\scripts)
  - moved utility entrypoints out of root:
    - `benchmark.py`
    - `cli.py`
    - `export_onnx.py`
    - `package_model.py`
    - `report.py`
    - `run_smoke_checks.py`
- [validate.py](F:\detektor\validate.py)
  - absorbed the useful comprehensive features from the old `validate_v2.py`
  - now supports:
    - `--output-dir`
    - `--save-images`
    - `--max-images`
    - `--compute-ap50-95`
  - remains the single public validation command
- [utils/auto_train_config.py](F:\detektor\utils\auto_train_config.py)
  - fixed a bug where auto-tuning overwrote explicit user config values like `train.epochs`
- [tests/test_auto_train_config.py](F:\detektor\tests\test_auto_train_config.py)
  - added regression coverage proving explicit epoch overrides are preserved

Removed:
- old duplicate validator `validate_v2.py`

Important behavioral fix:
- before this update, a config file containing `train.epochs: 5` was still being auto-expanded to `36` epochs
- that bug is now fixed
- resolved configs now preserve explicit overrides and only auto-tune unspecified values

Verification:

Unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_auto_train_config
```

Status:
- verified

Smoke checks:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_smoke_checks
```

Status:
- verified
- `7/7` checks passed

Real training run:

```powershell
.\.venv\Scripts\python.exe train.py --config reports/refactor_verify_5epoch.yaml --data-yaml F:/data/data.yaml --run-val --val-freq 1
```

Status:
- verified
- completed as an actual `5` epoch run after the auto-config override fix

Resolved config:
- output: `runs/refactor_verify_5epoch/`
- epochs: `5`
- warmup_epochs: `1`
- profile: `Comet`
- img_size: `512`
- batch_size: `4`
- grad_accum: `2`

Observed training validation trend:
- epoch 1:
  - validation failed before first checkpoint because `chimera_last.pt` did not exist yet
- epoch 2:
  - `P=0.0000` `R=0.0000` `mAP50=0.0000`
- epoch 3:
  - `P=0.0000` `R=0.0000` `mAP50=0.0000`
- epoch 4:
  - `P=0.0000` `R=0.0000` `mAP50=0.0000`
- epoch 5:
  - `P=0.5247` `R=0.2530` `mAP50=0.1463`

Standalone validation after training:

```powershell
.\.venv\Scripts\python.exe validate.py --config reports/refactor_verify_5epoch.yaml --weights runs/refactor_verify_5epoch/chimera_final_weights.pt --data-yaml F:/data/data.yaml --output-dir runs/validate/refactor_verify_5epoch
```

Status:
- verified

Observed standalone validation results:
- precision: `0.7808`
- recall: `0.2913`
- F1: `0.4243`
- AP50: `0.2363`
- mAP50: `0.0713`
- mean box IoU: `0.6867`

Per-class observation:
- detections are still concentrated almost entirely in class `player`
- classes `ball`, `goalkeeper`, and `referee` remained at `0.0` recall in this verified run

Git:
- pushed to `origin/beta`
- commit: `2fd8830` `Refactor project layout and consolidate validation`

Current state:
- repository layout is cleaner and more production-like
- generated outputs are no longer intended for git tracking
- validation now has one canonical entrypoint: `validate.py`
- explicit training config overrides are now honored correctly
- the main remaining issue is model quality imbalance across classes, not project structure

## Latest Update: Auto Overrides, Resource-Aware Batch Probing, And Architecture Matrix

Date:
- March 13, 2026

User request:
- keep recommended config and architecture selection automatic
- allow CLI overrides for model and training/augmentation settings
- maximize batch size from current free resources
- add a feature to list architecture compatibility on CPU and GPU
- run all model profiles for at least `3` epochs on this machine

Files changed:
- [train.py](F:\detektor\train.py)
  - added direct CLI overrides for device, model profile, training args, and augmentation args
  - training now resolves overrides after auto-tune
  - runtime logs now include free VRAM, free RAM, CPU count, and effective VRAM cap
  - CUDA batch probing now pushes to the largest safe multiple of `4`
- [utils/auto_train_config.py](F:\detektor\utils\auto_train_config.py)
  - auto-config now uses current free VRAM, free RAM, and CPU count
  - config overrides now apply cleanly after auto-tune
  - fixed model-profile override so `--model <profile>` rebuilds the full selected profile
- [utils/vram.py](F:\detektor\utils\vram.py)
  - `vram_cap: 0.95` now means `95%` of currently free VRAM, not total VRAM
- [utils/architecture_compatibility.py](F:\detektor\utils\architecture_compatibility.py)
  - new compatibility probe for all architecture profiles on CPU and CUDA
- [model_matrix.py](F:\detektor\model_matrix.py)
  - new user-facing command to print architecture compatibility and optionally run a real training sweep
- [tests/test_auto_train_config.py](F:\detektor\tests\test_auto_train_config.py)
  - added coverage for free-VRAM/free-RAM tuning and explicit override behavior
- [tests/test_architecture_compatibility.py](F:\detektor\tests\test_architecture_compatibility.py)
  - added coverage for compatibility matrix collection
- [README.md](F:\detektor\README.md)
  - training docs now describe auto-by-default behavior, architecture names, CLI model override, full training args, and augmentation args

New commands:

```powershell
.\.venv\Scripts\python.exe train.py --data-yaml F:/data/data.yaml
.\.venv\Scripts\python.exe train.py --data-yaml F:/data/data.yaml --model nova --no-auto-tune --batch-size 8
.\.venv\Scripts\python.exe model_matrix.py --data-yaml F:/data/data.yaml
.\.venv\Scripts\python.exe model_matrix.py --data-yaml F:/data/data.yaml --run-train-sweep --epochs 3
```

Verification:
- unit tests passed:
  - `.\.venv\Scripts\python.exe -m unittest tests.test_auto_train_config tests.test_architecture_compatibility`
- CLI help verified:
  - `.\.venv\Scripts\python.exe train.py --help`
  - `.\.venv\Scripts\python.exe model_matrix.py --help`
- override smoke run verified:
  - `.\.venv\Scripts\python.exe train.py --data-yaml F:/data/data.yaml --epochs 1 --batch-size 8 --no-auto-tune --model nova --no-augment --out-dir runs/cli_override_smoke_2`
- compatibility smoke run verified:
  - `.\.venv\Scripts\python.exe model_matrix.py --data-yaml F:/data/data.yaml --profiles firefly --max-batch-probe 8 --cpu-max-batch-probe 4 --output-dir runs/architecture_matrix_smoke`

Live machine results:
- compatibility summary:
  - `runs/architecture_matrix_live/compatibility_matrix.json`
- `3`-epoch sweep summary:
  - `runs/architecture_matrix_live/train_sweep_summary.json`
- measured `3`-epoch sweep times and resolved CUDA batch sizes:
  - `firefly`: `20` batch, `1.87 min`
  - `comet`: `24` batch, `1.31 min`
  - `nova`: `16` batch, `1.65 min`
  - `pulsar`: `12` batch, `1.57 min`
  - `quasar`: `12` batch, `1.33 min`
  - `supernova`: `8` batch, `1.97 min`
- approximate wall time:
  - training sweep only: `9.70 min`
  - compatibility probe only: `13.64 min`
  - full end-to-end run here: about `23.6 min`

Current state:
- training is now auto-configured by default and overrideable from CLI without editing YAML
- recommended profile selection is automatic, but all profile names are exposed to users through `--model`
- resource tuning is based on current free VRAM/RAM, not static hardware buckets
- the repo now includes a machine-specific architecture compatibility and benchmark feature
- the next major area of work is still actual model-accuracy improvement, not training automation



## Latest Update: In-Process Serving UI With Checkpoint Switching

Date:
- March 13, 2026

User request:
- read latest status from this handoff
- work on serving next
- add a `serve` flag to launch a user-friendly GUI
- support switching between `best` and `last`, defaulting to `best`
- allow image uploads plus folder-based inference
- show annotated predictions in the UI
- surface dataset, training, and performance details from the selected run
- keep the UI fast and lightweight

Files changed:
- [serve.py](F:\detektor\serve.py)
  - added `--ui` and `--ui-path`
  - UI now mounts into the same FastAPI process instead of requiring a second server
  - startup now discovers sibling checkpoints and defaults to `chimera_best.pt` when available
  - added runtime checkpoint switching without process restart
  - added runtime metadata endpoints:
    - `/runtime`
    - `/runtime/select_model`
- [ui/app.py](F:\detektor\ui\app.py)
  - replaced the old thin backend client with a run-aware dashboard for in-process serving mode
  - batch inference now uses the in-memory service directly for lower overhead
  - UI now supports:
    - checkpoint selection between `best` and `last`
    - drag/drop image uploads
    - folder path inference
    - annotated gallery output
    - dataset details
    - training summary
    - validation history
    - training and validation curve plots
    - saved run plot previews
  - preserved a fallback standalone remote-backend mode for `python ui/app.py`
- [api/run_artifacts.py](F:\detektor\api\run_artifacts.py)
  - added run artifact discovery and parsing for:
    - checkpoint siblings
    - resolved config
    - run summary
    - training summary
    - validation history
    - saved plot paths
- [tests/test_api.py](F:\detektor\tests\test_api.py)
  - added coverage for runtime metadata and checkpoint switching endpoints
- [tests/test_run_artifacts.py](F:\detektor\tests\test_run_artifacts.py)
  - added coverage for checkpoint discovery and run artifact parsing

New command:

```powershell
.\.venv\Scripts\python.exe serve.py --weights runs/refactor_verify_5epoch/chimera_last.pt --ui
```

Behavior:
- if both `chimera_best.pt` and `chimera_last.pt` exist in the same run directory, serving now starts on `best`
- the UI is mounted at `/ui` on the same host/port as the API
- batch UI inference is chunked by `max_batch_size` and avoids the extra HTTP round-trip path in mounted mode

Verification:
- unit tests passed:
  - `.\.venv\Scripts\python.exe -m unittest tests.test_run_artifacts tests.test_api`
- Python compile check passed:
  - `.\.venv\Scripts\python.exe -m py_compile serve.py ui/app.py api/run_artifacts.py`
- Gradio interface construction smoke checks passed in both:
  - standalone mode
  - mounted runtime mode

Status:
- verified for code path, tests, and interface construction
- not yet verified through a full live browser session against a real model in this update

Current state:
- serving now has an integrated GUI path suitable for local interactive use
- the UI can inspect a training run and infer on images without manually wiring class maps or plot paths
- `best` vs `last` is now a runtime choice instead of a restart-time choice
- next work on serving should focus on live UX polish and real-model browser validation, not basic plumbing
