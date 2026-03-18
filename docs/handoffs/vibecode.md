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
  - result: `172` passed path, `1` skipped
- Main remaining risks:
  - model quality and recall
  - real browser-level serving verification against a trained run
  - deployment-path verification gaps such as Docker

## Project Z

### Phase 2: Serving And Contract Stabilization

Goal:
- make local serving reliable and contract-stable now that the base suite is trustworthy

Items:
- `Z-6` `Story` - Lock the public local API contract
  - scope:
    - version the response contract deliberately
    - document legacy-field compatibility rules
    - align server responses, schemas, and tests
  - target outcome:
    - one stable documented contract for local API consumers
  - verification:
    - `.\.venv\Scripts\python.exe -m unittest tests.test_api tests.test_schemas tests.test_regression`
    - docs and implementation agree

- `Z-7` `Bug` - Verify integrated UI and runtime checkpoint switching against a real trained run
  - problem:
    - UI construction is tested, but a real end-to-end browser flow has not been verified against an actual checkpointed run
  - target outcome:
    - verify upload flow, folder inference, plot display, run metadata, and `best`/`last` switching end to end
  - verification:
    - live manual verification notes recorded here
    - any newly discovered defects logged as new `Z-*` items

Phase 2 exit:
- local API responses are stable and documented
- mounted UI workflow is verified with a real trained run

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

1. `Z-6`
2. `Z-7`
3. `Z-8`
4. `Z-9`
5. `Z-10`
6. `Z-11`
7. `Z-12`
8. `Z-13`
