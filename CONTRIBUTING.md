# Contributing to Detektor

Thanks for your interest! Detektor is a lightweight, practical object detection and instance segmentation project for
rapid experimentation and single-machine deployment. This repository is also explicitly **vibe-coded**: it was built
through iterative collaboration between a human developer and AI coding assistants, so expect pragmatic decisions and a
preference for improving working paths incrementally over large rewrites.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Security issues go through
[SECURITY.md](SECURITY.md), not public issues.

## Where contributions are especially welcome

- Model quality (recall/precision), architecture and training improvements
- TensorRT / ONNX Runtime export and runtime integration
- Dataset loaders and augmentation
- **Performance benchmarking** — new suites, GPU results, more hardware baselines
- Deployment and runtime tooling, documentation

## Development setup

```bash
git clone https://github.com/Kartik-A-1820/detektor && cd detektor
python -m venv .venv && source .venv/bin/activate

# CPU-only PyTorch is enough for the whole test suite
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

make test          # or: python -m pytest
make lint          # or: ruff check .
make bench-quick   # ≈1 min benchmark smoke run
```

Optional: `pip install pre-commit && pre-commit install` to run lint on every commit.

## Project conventions

- Keep changes focused, production-minded and **backward compatible** (CLI flags, checkpoint format, API schema).
- Avoid redesigning working core model/training logic without a strong reason and measurements.
- Prefer explicit code over clever abstractions; add type hints where useful.
- Match the surrounding style; `ruff check .` must pass (config in `pyproject.toml`).
- Don't add dependencies unless clearly justified; keep optional ones optional.
- Pure logic goes in importable modules with unit tests (see `ui/render.py` vs `ui/app.py` for the pattern).

## Tests

```bash
python -m pytest                          # everything (≈1–2 min on CPU)
python -m pytest tests/test_api.py -x     # a single file, stop on first failure
python -m scripts.run_smoke_checks        # lightweight model smoke checks
```

Every bug fix should come with a regression test; every new feature with tests for its main path and edge cases.
Tests must run without a GPU or network access. See [docs/guides/TESTING.md](docs/guides/TESTING.md).

## Benchmarks

Performance claims need numbers. For anything that touches the model, post-processing, data path or serving layer:

```bash
python -m benchmarks run --suites fast --profiles firefly,comet --tag before     # on main
python -m benchmarks run --suites fast --profiles firefly,comet --tag after      # on your branch
python -m benchmarks compare runs/benchmarks/<before>/results.json runs/benchmarks/<after>/results.json
```

Run both on the same idle machine and include the comparison table in your PR. See
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) for methodology and how to add a suite.

## Pull requests

- Keep scope focused and explain the motivation; link the issue.
- Include test/benchmark results and mention any CLI, config, dependency or behaviour changes (with migration notes).
- Update docs and `CHANGELOG.md` (under *Unreleased*) for user-visible changes.
- CI runs lint, tests (Python 3.11/3.12), a benchmark smoke run and a Docker build — all must be green.

PRs that are easier to review touch fewer unrelated files, include a short design note for non-obvious choices and
keep commits logically separated.

## Reporting issues

Please include OS, Python/PyTorch versions, GPU/CUDA details if relevant, the exact command, the full traceback and
minimal reproduction steps. The issue forms in `.github/ISSUE_TEMPLATE` guide you.

## Release process (maintainers)

1. Update `api/__init__.py::__version__` and `CHANGELOG.md`.
2. Run `make test lint` and `make bench` on the reference machine; refresh `benchmarks/results/` if numbers moved.
3. Tag `vX.Y.Z` and publish release notes from the changelog.
