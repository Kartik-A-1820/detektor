# VibeCode Handoff

## Operating Constraints

- Use `.\.venv\Scripts\python.exe ...` or `cmd /c ".venv\Scripts\activate.bat && python ..."`; `Activate.ps1` is blocked.
- Work on branch `beta`; if missing, create it from the current integrated baseline before starting.
- Push only to `origin beta`.
- Active dataset: `F:/data/data.yaml`.
- Do not commit generated `artifacts/`, `reports/`, or ad hoc `runs/` outputs unless explicitly requested.

## Canonical Rules

- This file tracks only active tasks, open issues, current constraints, and the next priority order.
- Once a Story or Bug is solved cleanly and verified, remove it from this file instead of keeping historical detail here.
- If a new defect is found, add it here immediately as the next available `Z-*` item.
- Keep each active item concise but complete:
  - type
  - problem or scope
  - target outcome
  - concrete verification
- For every completed `Z-*` item:
  - make a git commit whose title starts with that item ID
  - push the result to `beta`
  - then remove the solved item from this file

## Current Verified State

- Full automated unittest suite is green in this environment.
- Latest full verification:
  - `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"`
  - result: `175` passed path, `1` skipped
- Latest contract verification:
  - `.\.venv\Scripts\python.exe -m unittest tests.test_api tests.test_schemas tests.test_regression`
  - result: `35` passed
- Local serving contract:
  - current contract version: `v1`
  - docs aligned in `README.md`, `docs/reference/TOOLS.md`, and `docs/guides/QUICKSTART.md`
- Latest live UI verification: `2026-03-18`
  - run: `F:/detektor/runs/auto_verify_5epoch`
  - serve command:
    - `.\.venv\Scripts\python.exe serve.py --weights runs/auto_verify_5epoch/chimera_best.pt --device cpu --host 127.0.0.1 --port 8000 --no-warmup --ui`
  - verified:
    - mounted `/ui/` loads and `GET /version` plus `GET /runtime` expose `contract_version = v1`
    - dashboard shows run metadata, class map, validation history, and saved plots
    - upload flow works on `F:/data/test/images/08fd33_3_6_png.rf.261781c58b95436fb40e6afc0495bc57.jpg` with active checkpoint `last`
    - folder inference works on `F:/detektor/.tmp_testdata/ui_verify` with active checkpoint `best`
    - runtime checkpoint switching works end to end: `best -> last -> best`
- Phase 3 diagnostic baseline: `2026-03-18`
  - baseline checkpoint: `F:/detektor/runs/refactor_verify_5epoch/chimera_best.pt`
  - baseline validation artifacts: `F:/detektor/runs/z8_baseline_validate`
  - baseline assignment audit: `F:/detektor/runs/z8_phase3_baseline/phase3_diagnostics.json`
  - commands:
    - `.\.venv\Scripts\python.exe validate.py --weights runs/refactor_verify_5epoch/chimera_best.pt --data-yaml F:/data/data.yaml --output-dir runs/z8_baseline_validate`
    - `.\.venv\Scripts\python.exe scripts/phase3_diagnostics.py --data-yaml F:/data/data.yaml --output runs/z8_phase3_baseline/phase3_diagnostics.json`
    - `.\.venv\Scripts\python.exe check_dataset.py --data-yaml F:/data/data.yaml`
  - verified baseline results:
    - detection precision `0.7808`, recall `0.2913`, AP50 `0.2363`, mean box IoU `0.6867`
    - per-class recall is `0.0000` for `ball`, `goalkeeper`, and `referee`; only `player` is detected (`0.3515` recall)
    - dataset remains heavily skewed toward `player` annotations (`83.4%` overall) with `ball` at `3.6%`
    - at `512` input resolution with the current `CenterPriorAssigner(center_radius=2.5)`, `70.54%` of training `ball` boxes and `60.0%` of validation `ball` boxes receive zero positive points
    - median `ball` box size is only `3.11 x 5.78 px` on train and `2.67 x 5.33 px` on val, so the current `inside_box` rule is the dominant recall blocker for tiny objects; class imbalance is secondary
- Latest Phase 3 controlled probe: `2026-03-18`
  - candidate: stride-aware minimum effective target size (`8 px`) inside `CenterPriorAssigner`
  - diagnostic artifacts: `F:/detektor/runs/z9_effective_box_phase3/phase3_diagnostics.json`
  - training run: `F:/detektor/runs/z9_effective_box_5epoch`
  - standalone validation artifacts: `F:/detektor/runs/z9_effective_box_5epoch_validate`
  - commands:
    - `.\.venv\Scripts\python.exe scripts/phase3_diagnostics.py --data-yaml F:/data/data.yaml --output runs/z9_effective_box_phase3/phase3_diagnostics.json`
    - `.\.venv\Scripts\python.exe train.py --config runs/refactor_verify_5epoch/resolved_train_config.yaml --data-yaml F:/data/data.yaml --device cuda --img-size 512 --epochs 5 --batch-size 4 --grad-accum 2 --lr 0.002 --num-workers 0 --vram-cap 0.8 --no-maximize-batch-size --out-dir runs/z9_effective_box_5epoch --run-val --val-freq 1`
    - `.\.venv\Scripts\python.exe validate.py --weights runs/z9_effective_box_5epoch/chimera_best.pt --data-yaml F:/data/data.yaml --output-dir runs/z9_effective_box_5epoch_validate`
  - verified results versus baseline:
    - assignment audit: `ball` zero-positive rate improved from `70.54%` to `0.0%` on train and from `60.0%` to `0.0%` on val
    - standalone validation improved overall recall from `0.2913` to `0.4310` and AP50 from `0.2363` to `0.3543`
    - precision dropped from `0.7808` to `0.6156`
    - per-class recall is still `0.0000` for `ball`, `goalkeeper`, and `referee`; gains are currently isolated to `player`
- Main remaining risks:
  - model quality and recall outside the dominant `player` class
  - deployment-path verification gaps such as Docker

## Project Z

### Phase 3: Training Quality And Model Performance

Goal:
- improve real detection quality after the engineering base was stabilized

Items:
- `Z-9` `Story` - Improve recall via target-assignment and head/loss calibration changes
  - scope:
    - investigate assignment thresholds, positive matching, confidence calibration, and loss weighting
    - current blocker is no longer zero assignment; it is zero minority-class recall after assignment was fixed
  - current best probe:
    - `2026-03-18`: stride-aware minimum effective target size (`8 px`) inside `CenterPriorAssigner`
    - diagnostic artifacts: `F:/detektor/runs/z9_effective_box_phase3/phase3_diagnostics.json`
    - training run: `F:/detektor/runs/z9_effective_box_5epoch`
    - standalone validation artifacts: `F:/detektor/runs/z9_effective_box_5epoch_validate`
    - result versus baseline:
      - baseline standalone validate: precision `0.7808`, recall `0.2913`, AP50 `0.2363`
      - current candidate: precision `0.6156`, recall `0.4310`, AP50 `0.3543`
      - assignment audit improved `ball` zero-positive rate from `70.54%` to `0.0%` on train and from `60.0%` to `0.0%` on val
      - per-class recall is still `0.0000` for `ball`, `goalkeeper`, and `referee`, so the dominant non-player failure mode remains open
  - rejected probe:
    - `2026-03-18`: per-GT fallback assignment plus fallback objectness floor `0.2`
    - training run: `F:/detektor/runs/z9_assigner_fallback_objfloor_5epoch`
    - standalone validation artifacts: `F:/detektor/runs/z9_assigner_fallback_objfloor_5epoch_validate`
    - result: precision `0.6781`, recall `0.2692`, AP50 `0.1878`
  - next-agent execution order:
    - first, fix `Z-14` so every fresh `--run-val` training run records a valid epoch-1 checkpoint evaluation
    - second, keep the current assigner change as the baseline and do not revert it unless a new controlled probe beats `runs/z9_effective_box_5epoch_validate`
    - third, run one calibration-focused probe that targets class discrimination rather than assignment count
    - recommended first probe: minority-class-aware classification/objectness weighting or thresholding that is measured against per-class recall, not just aggregate AP50
    - avoid starting long runs until a 5-epoch probe produces non-zero recall for at least one of `ball`, `goalkeeper`, or `referee`
  - exit criteria:
    - materially better recall and AP50 than the Z-8 baseline
    - non-zero standalone validation recall for at least one currently dead minority class
    - exact commands, run paths, and per-class metrics recorded here
  - verification:
    - before/after run comparison logged here with exact commands and paths
    - standalone `validate.py` evidence must include `per_class_metrics.csv`

- `Z-10` `Story` - Run longer controlled experiments after calibration changes
  - scope:
    - starts only after `Z-9` has a clean short-run candidate that meets the `Z-9` exit criteria
    - compare short smoke runs against longer verification runs
    - confirm gains persist past early epochs without minority-class regression
  - entry gate:
    - do not start from a candidate that improves only aggregate `player` metrics while leaving `ball`, `goalkeeper`, and `referee` at `0.0000` recall
  - target outcome:
    - stable longer-run improvement, not a short-run artifact
    - checkpoint choice backed by both aggregate metrics and per-class recall
  - verification:
    - epoch-by-epoch metrics summary recorded here
    - selected checkpoint and rationale documented here
    - standalone validation artifacts for the long run recorded here

- `Z-14` `Bug` - Fix training-time validation hook ordering for fresh runs
  - problem:
    - `train.py --run-val` attempts validation at epoch 1 before `runs/.../chimera_last.pt` exists, producing a missing-file warning and skipping the first validation point
  - target outcome:
    - every requested validation epoch evaluates a real checkpoint without relying on a prior run artifact
  - verification:
    - fresh `--run-val` training run records epoch-1 validation metrics without a missing-file warning

- `Z-15` `Story` - Add per-class Phase 3 promotion gates to validation reporting
  - problem:
    - aggregate AP50 improved in `Z-9`, but three classes still have `0.0000` recall and that can be missed when comparing only top-line metrics
  - target outcome:
    - Phase 3 comparisons and checkpoint promotion decisions explicitly surface minority-class recall and dead-class status
  - verification:
    - validation summary or handoff log clearly reports per-class recall deltas and flags classes that remain at zero recall

Phase 3 exit:
- recall and mAP improve materially on the active dataset
- improvements are reproducible
- checkpoint selection and validation reports agree

### Phase 4: Deployment Readiness For Local Use

Goal:
- verify the delivery paths that the repo currently claims to support

Items:
- `Z-11` `Bug` - Verify Docker workflow when Docker is available
  - problem:
    - Docker is documented but not verified on this machine
  - target outcome:
    - either verify Docker end to end or downgrade unsupported claims
  - verification:
    - exact command log and result recorded here

- `Z-12` `Story` - Validate ONNX export and benchmark path against real checkpoints
  - scope:
    - export a real model
    - confirm benchmark path and exported artifact usability
  - target outcome:
    - verified export path with real checkpoint evidence
  - verification:
    - commands, artifacts, and benchmark summary recorded here

Phase 4 exit:
- documented deployment and export flows are verified or explicitly downgraded

### Phase 5: Release Readiness Review

Goal:
- decide what the repo can honestly claim after the remaining verification work

Items:
- `Z-13` `Story` - Produce final Project Z readiness review
  - scope:
    - summarize remaining technical debt
    - classify the repo as internal prototype, local-use release candidate, or production-ready for narrow use
  - target outcome:
    - evidence-based readiness classification
  - verification:
    - final assessment written here and reflected in public docs if needed

Phase 5 exit:
- project claims match verified evidence
- unresolved risks are explicitly documented

## Priority Order

1. `Z-14`
2. `Z-9`
3. `Z-15`
4. `Z-10`
5. `Z-11`
6. `Z-12`
7. `Z-13`
