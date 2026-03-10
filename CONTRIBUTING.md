# Contributing to Detektor

## Project Philosophy

Detektor is a lightweight, practical object detection and instance segmentation project built for rapid experimentation, local deployment, and straightforward iteration. The codebase favors small reusable modules, clear CLI workflows, and developer-friendly defaults over heavy abstractions.

This repository is also explicitly **vibe-coded**: it was built through iterative collaboration between a human developer and AI coding assistants. In practice, that means contributors should expect pragmatic decisions, quick iteration, and a preference for improving working paths incrementally rather than performing large rewrites.

## Where Contributions Are Especially Welcome

- model optimization
- TensorRT export and runtime integration
- dataset loaders
- training improvements
- performance benchmarking
- deployment/runtime tooling

## Coding Style

- Keep changes lightweight and production-minded.
- Avoid redesigning working core model or training logic unless there is a strong reason.
- Prefer explicit code over clever abstractions.
- Add type hints where useful.
- Keep CLI and utility behavior backward compatible when possible.
- Do not add dependencies unless they are clearly justified.

## Development Setup

Install runtime requirements:

```bash
pip install -r requirements.txt
```

Install development requirements:

```bash
pip install -r requirements-dev.txt
```

## Running Tests

Run the lightweight smoke suite:

```bash
python run_smoke_checks.py
```

Run unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Pull Requests

When opening a PR:

- keep scope focused
- explain the motivation clearly
- include local test/smoke results
- mention any CLI, config, or dependency changes
- include migration notes if behavior changes

## Reporting Issues

When filing an issue, please include:

- operating system
- Python version
- GPU / CUDA availability if relevant
- exact command used
- full traceback or error logs
- minimal reproduction steps

## Review Expectations

PRs that are easier to review tend to:

- touch fewer unrelated files
- avoid unnecessary formatting churn
- preserve existing workflows
- include concise documentation updates where needed
