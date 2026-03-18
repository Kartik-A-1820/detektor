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
- Main remaining risks:
  - model quality and recall
  - deployment-path verification gaps such as Docker

## Project Z

### Phase 3: Training Quality And Model Performance

Goal:
- improve real detection quality after the engineering base was stabilized

Items:
- `Z-8` `Story` - Audit training signal quality and dominant recall failure mode on `F:/data/data.yaml`
  - scope:
    - inspect class balance, target quality, assignment behavior, and current validation curves
    - identify the main cause of weak recall
  - target outcome:
    - written diagnosis with a reproducible baseline
  - verification:
    - concise diagnostic summary recorded here
    - baseline metrics recorded with exact run paths

- `Z-9` `Story` - Improve recall via target-assignment and head/loss calibration changes
  - scope:
    - investigate assignment thresholds, positive matching, confidence calibration, and loss weighting
  - target outcome:
    - materially better recall and mAP on controlled comparisons
  - verification:
    - before/after run comparison logged here with exact commands and paths

- `Z-10` `Story` - Run longer controlled experiments after calibration changes
  - scope:
    - compare short smoke runs against longer verification runs
    - confirm gains persist past early epochs
  - target outcome:
    - stable longer-run improvement, not a short-run artifact
  - verification:
    - epoch-by-epoch metrics summary recorded here
    - selected checkpoint and rationale documented here

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

1. `Z-8`
2. `Z-9`
3. `Z-10`
4. `Z-11`
5. `Z-12`
6. `Z-13`
