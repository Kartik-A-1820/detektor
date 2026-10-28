# Published benchmark results

| File | Contents |
| --- | --- |
| [`baseline-cpu.json`](baseline-cpu.json) | Raw results (`python -m benchmarks run --suites all --profiles all`) — machine readable, diff-able with `python -m benchmarks compare` |
| [`baseline-cpu.md`](baseline-cpu.md) | The same data rendered as a report |

The environment block at the top of each file records the exact hardware and library versions. These are **CPU-only
numbers from a shared 4‑vCPU virtual machine**; absolute values will differ on your hardware — compare relative changes
on the same machine. Contributions of results from other hardware (especially CUDA GPUs) are welcome: add
`<hardware>.json` / `.md` here via a pull request.

Check your own machine against the baseline:

```bash
python -m benchmarks run --suites fast --profiles firefly,comet,nova --tag mine
python -m benchmarks compare benchmarks/results/baseline-cpu.json runs/benchmarks/<your-run>/results.json --threshold 25
```

A timing-only reproduction can use an architecture profile without trained weights. Accuracy/robustness claims require the matching trained checkpoint, dataset, split, and evaluation settings. Label those inputs explicitly when contributing a result; do not infer model quality from random-weight latency measurements.
