# Documentation

The repository root stays focused on runnable entry points and project metadata; everything else lives here.

## Guides — how to do things

| Guide | What it covers |
| --- | --- |
| [QUICKSTART](guides/QUICKSTART.md) | End-to-end walk-through: dataset → train → validate → serve |
| [DATASETS](guides/DATASETS.md) | YOLO/Roboflow format, auto-configuration, `check_dataset.py` preflight validation |
| [TASK_MODES](guides/TASK_MODES.md) | Detection vs instance segmentation and auto-detection |
| [TRAINING](guides/TRAINING.md) | Auto-tuned config, profiles, every training flag, loss components, debugging |
| [EVALUATION](guides/EVALUATION.md) | `validate.py` metrics and artifacts; the reporting module |
| [INFERENCE](guides/INFERENCE.md) | Single-image and folder inference |
| [EXPORT](guides/EXPORT.md) | ONNX export |
| [DEPLOYMENT](guides/DEPLOYMENT.md) | Docker, Compose, reverse proxy/TLS, systemd, Kubernetes, production checklist |
| [TESTING](guides/TESTING.md) | Running and writing tests |
| [TROUBLESHOOTING](guides/TROUBLESHOOTING.md) | Common training / inference / API problems and performance tips |

## Reference

| Document | What it covers |
| --- | --- |
| [reference/TOOLS.md](reference/TOOLS.md) | Every CLI tool and its flags |
| [reference/API.md](reference/API.md) | REST endpoints, auth, limits, metrics, error format |
| [reference/CONFIGURATION.md](reference/CONFIGURATION.md) | All `serve.py` flags and `DETEKTOR_*` environment variables |
| [reference/REPORTING.md](reference/REPORTING.md) | Reporting module |
| [reference/VALIDATION_OUTPUT_SCHEMA.md](reference/VALIDATION_OUTPUT_SCHEMA.md) | Validation output format |

## Concepts, performance and status

| Document | What it covers |
| --- | --- |
| [ARCHITECTURE](ARCHITECTURE.md) | System design, model, checkpoints, serving layer, repository layout |
| [BENCHMARKS](BENCHMARKS.md) | Benchmark suites, methodology, published baseline, regression gating |
| [status/PROJECT_STATUS.md](status/PROJECT_STATUS.md) | Maturity, limitations and roadmap |
| [../CHANGELOG.md](../CHANGELOG.md) | Release notes |
| [../SECURITY.md](../SECURITY.md) | Security policy and hardening guidance |

## Internal notes

Implementation write-ups kept for maintainers: [OPTIMIZER_LOSS_BASELINE](internal/OPTIMIZER_LOSS_BASELINE.md),
[TRAINING_STABILITY](internal/TRAINING_STABILITY.md), [ROBUSTNESS](internal/ROBUSTNESS.md),
[INTEGRATION_SUMMARY](internal/INTEGRATION_SUMMARY.md),
[REPORTING_IMPLEMENTATION_SUMMARY](internal/REPORTING_IMPLEMENTATION_SUMMARY.md), and the collaboration
[handoff log](handoffs/vibecode.md).
