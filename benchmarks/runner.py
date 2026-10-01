"""Command-line entry point: ``python -m benchmarks {list,run,report,compare}``."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

from benchmarks import SCHEMA_VERSION
from benchmarks.common import ALL_PROFILES, BenchContext, seed_everything
from benchmarks.env import collect_environment
from benchmarks.report import compare, render_markdown
from benchmarks.suites import SUITES, load, resolve


def _csv_ints(text: str) -> List[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: --device cuda requested but CUDA is not available")
    return name


def cmd_list(_: argparse.Namespace) -> int:
    print("Available suites (use with `run --suites`; aliases: all, fast):\n")
    for name in SUITES:
        print(f"  {name:<11} {load(name).DESCRIPTION}")
    print(f"\nProfiles: {', '.join(ALL_PROFILES)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    profiles = list(ALL_PROFILES) if args.profiles == "all" else [p.strip() for p in args.profiles.split(",") if p.strip()]
    for profile in profiles:
        if profile not in ALL_PROFILES:
            raise SystemExit(f"error: unknown profile '{profile}'. Choose from {', '.join(ALL_PROFILES)}")
    try:
        suites = resolve(args.suites.split(","))
    except KeyError as exc:
        raise SystemExit(f"error: {exc.args[0]}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / (f"{stamp}_{args.tag}" if args.tag else stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_sizes = _csv_ints(args.img_sizes)
    ctx = BenchContext(
        profiles=profiles, device=device, img_sizes=img_sizes, img_size=args.img_size or max(img_sizes),
        batch_sizes=_csv_ints(args.batch_sizes), num_classes=args.num_classes,
        warmup=args.warmup if not args.quick else 1, runs=args.runs if not args.quick else 5,
        seed=args.seed, weights=args.weights or None, data_yaml=args.data_yaml or None,
        output_dir=out_dir, quick=args.quick,
    )
    if args.e2e_epochs:
        ctx.extras["e2e_epochs"] = args.e2e_epochs

    seed_everything(args.seed)
    results: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment": collect_environment(device, args.threads),
        "config": {
            "suites": suites, "profiles": profiles, "device": device, "img_sizes": ctx.img_sizes, "img_size": ctx.img_size,
            "batch_sizes": ctx.batch_sizes, "warmup": ctx.warmup, "runs": ctx.runs, "seed": ctx.seed, "quick": ctx.quick,
        },
        "suites": {},
    }

    print(f"Detektor benchmarks → {out_dir}\n  device={device} profiles={','.join(profiles)} suites={','.join(suites)}\n")
    failed = 0
    for name in suites:
        print(f"▶ {name} ...", flush=True)
        started = time.perf_counter()
        try:
            suite_result = load(name).run(ctx)
            suite_result["status"] = "skipped" if suite_result.get("skipped") else "ok"
        except Exception as exc:  # noqa: BLE001 - one broken suite must not discard the others
            traceback.print_exc()
            suite_result = {"description": load(name).DESCRIPTION, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            failed += 1
        suite_result["duration_s"] = round(time.perf_counter() - started, 1)
        results["suites"][name] = suite_result
        print(f"  {suite_result['status']} in {suite_result['duration_s']} s", flush=True)
        # Persist after every suite so a crash never loses finished work.
        (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    shutil.rmtree(out_dir / "_scratch", ignore_errors=True)
    report = render_markdown(results)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"\nResults: {out_dir / 'results.json'}\nReport:  {out_dir / 'report.md'}")
    return 1 if failed else 0


def cmd_report(args: argparse.Namespace) -> int:
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    md = render_markdown(results)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(md)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))
    md, regressions = compare(base, new, args.threshold)
    print(md)
    if regressions:
        print(f"{len(regressions)} regression(s) beyond {args.threshold:g}%:", file=sys.stderr)
        for line in regressions:
            print(f"  - {line}", file=sys.stderr)
        return 1 if args.fail_on_regression else 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks", description="Detektor benchmark suite")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List suites and profiles").set_defaults(func=cmd_list)

    run = sub.add_parser("run", help="Run benchmark suites")
    run.add_argument("--suites", default="fast", help="Comma list of suites, or 'fast' / 'all' (default: fast)")
    run.add_argument("--profiles", default="firefly,comet,nova", help="Comma list of architecture profiles or 'all'")
    run.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    run.add_argument("--img-sizes", default="320,512", help="Comma list of input sizes for latency/complexity")
    run.add_argument("--img-size", type=int, default=0, help="Primary size for throughput/memory (default: largest of --img-sizes)")
    run.add_argument("--batch-sizes", default="1,2,4,8")
    run.add_argument("--num-classes", type=int, default=4)
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--runs", type=int, default=30)
    run.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = library default)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--weights", default="", help="Checkpoint for the accuracy/robustness suites")
    run.add_argument("--data-yaml", default="", help="Dataset YAML for the accuracy/robustness suites")
    run.add_argument("--e2e-epochs", type=int, default=0, help="Epochs for the synthetic end-to-end suite")
    run.add_argument("--quick", action="store_true", help="Few iterations / small workloads (CI smoke run)")
    run.add_argument("--output-dir", default="runs/benchmarks")
    run.add_argument("--tag", default="", help="Suffix for the output folder name")
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="Render results.json as Markdown")
    rep.add_argument("results")
    rep.add_argument("--output", "-o", default="")
    rep.set_defaults(func=cmd_report)

    cmp_ = sub.add_parser("compare", help="Diff two results.json files and flag regressions")
    cmp_.add_argument("base")
    cmp_.add_argument("new")
    cmp_.add_argument("--threshold", type=float, default=15.0, help="Percent change treated as a regression")
    cmp_.add_argument("--fail-on-regression", action="store_true", help="Exit non-zero when a regression is found")
    cmp_.set_defaults(func=cmd_compare)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
