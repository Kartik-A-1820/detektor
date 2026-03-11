#!/usr/bin/env python
"""Benchmark PyTorch vs ONNX Runtime inference for Detektor models."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from api.utils import load_model
from utils.benchmarking import BenchmarkSample, compute_latency_stats, load_benchmark_samples

try:  # Optional dependency for ONNX runtime benchmarks
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


@dataclass
class BackendResult:
    name: str
    framework: str
    device: str
    status: str
    warmup_latencies: List[float]
    steady_latencies: List[float]
    error: Optional[str] = None

    def to_summary(self) -> Dict[str, object]:
        warmup_stats = compute_latency_stats(self.warmup_latencies)
        steady_stats = compute_latency_stats(self.steady_latencies)
        return {
            "backend": self.name,
            "framework": self.framework,
            "device": self.device,
            "status": self.status,
            "error": self.error,
            "warmup": {"runs": len(self.warmup_latencies), **warmup_stats},
            "steady": {"runs": len(self.steady_latencies), **steady_stats},
        }


def _run_torch_backend(
    model_path: Path,
    device_name: str,
    samples: List[BenchmarkSample],
    warmup_runs: int,
    benchmark_runs: int,
    proto_k: int,
) -> BackendResult:
    backend_name = f"pytorch-{device_name}"
    try:
        model, device = load_model(str(model_path), proto_k=proto_k, device_name=device_name)
    except Exception as exc:  # noqa: BLE001
        return BackendResult(
            name=backend_name,
            framework="pytorch",
            device=device_name,
            status="error",
            warmup_latencies=[],
            steady_latencies=[],
            error=str(exc),
        )

    model.eval()
    warmup_latencies: List[float] = []
    steady_latencies: List[float] = []
    total_iterations = warmup_runs + benchmark_runs
    if total_iterations == 0:
        return BackendResult(backend_name, "pytorch", device_name, "ok", [], [])

    with torch.no_grad():
        for idx in range(total_iterations):
            sample = samples[idx % len(samples)]
            tensor = sample.torch_tensor.to(device, non_blocking=True)
            start = time.perf_counter()
            _ = model.predict(
                tensor,
                original_sizes=[sample.original_size],
            )
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            latency_ms = (time.perf_counter() - start) * 1000.0
            if idx < warmup_runs:
                warmup_latencies.append(latency_ms)
            else:
                steady_latencies.append(latency_ms)

    return BackendResult(backend_name, "pytorch", device_name, "ok", warmup_latencies, steady_latencies)


def _prepare_ort_session(onnx_path: Path, providers: List[str]) -> "ort.InferenceSession":
    if ort is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "onnxruntime is not installed. `pip install onnxruntime` (or onnxruntime-gpu) to enable ORT benchmarks."
        )
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    available = set(ort.get_available_providers())
    missing = [provider for provider in providers if provider not in available]
    if missing and providers[0] == "CUDAExecutionProvider":
        raise RuntimeError(
            "CUDAExecutionProvider requested but not available. Install onnxruntime-gpu and ensure CUDA drivers are installed."
        )

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    return session


def _run_onnx_backend(
    onnx_path: Path,
    provider: str,
    samples: List[BenchmarkSample],
    warmup_runs: int,
    benchmark_runs: int,
) -> BackendResult:
    backend_name = f"onnx-{provider.replace('ExecutionProvider', '').lower()}"
    try:
        providers = [provider]
        if provider == "CUDAExecutionProvider":
            providers.append("CPUExecutionProvider")
        session = _prepare_ort_session(onnx_path, providers)
    except Exception as exc:  # noqa: BLE001
        return BackendResult(
            name=backend_name,
            framework="onnxruntime",
            device=provider,
            status="error",
            warmup_latencies=[],
            steady_latencies=[],
            error=str(exc),
        )

    input_name = session.get_inputs()[0].name
    warmup_latencies: List[float] = []
    steady_latencies: List[float] = []
    total_iterations = warmup_runs + benchmark_runs
    if total_iterations == 0:
        return BackendResult(backend_name, "onnxruntime", provider, "ok", [], [])

    for idx in range(total_iterations):
        sample = samples[idx % len(samples)]
        ort_inputs = {input_name: sample.numpy_array.astype("float32")}
        start = time.perf_counter()
        session.run(None, ort_inputs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        if idx < warmup_runs:
            warmup_latencies.append(latency_ms)
        else:
            steady_latencies.append(latency_ms)

    status = "ok"
    return BackendResult(backend_name, "onnxruntime", provider, status, warmup_latencies, steady_latencies)


def _write_outputs(
    output_dir: Path,
    results: List[BackendResult],
    samples: List[BenchmarkSample],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_summary.json"
    table_path = output_dir / "benchmark_table.csv"

    summary_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "num_samples": len(samples),
        "sample_paths": [str(sample.path) for sample in samples],
        "image_size": args.image_size,
        "warmup_runs": args.warmup,
        "benchmark_runs": args.runs,
        "results": [result.to_summary() for result in results],
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    with table_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "backend",
                "framework",
                "device",
                "status",
                "avg_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "throughput_img_s",
                "warmup_avg_ms",
                "warmup_p95_ms",
                "warmup_runs",
                "steady_runs",
                "error",
            ]
        )
        for result in results:
            warmup_stats = compute_latency_stats(result.warmup_latencies)
            steady_stats = compute_latency_stats(result.steady_latencies)
            writer.writerow(
                [
                    result.name,
                    result.framework,
                    result.device,
                    result.status,
                    steady_stats["avg_latency_ms"],
                    steady_stats["p50_latency_ms"],
                    steady_stats["p95_latency_ms"],
                    steady_stats["throughput_img_s"],
                    warmup_stats["avg_latency_ms"],
                    warmup_stats["p95_latency_ms"],
                    len(result.warmup_latencies),
                    len(result.steady_latencies),
                    result.error,
                ]
            )

    print(f"Benchmark summary saved to {summary_path}")
    print(f"Benchmark table saved to {table_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Detektor inference backends")
    parser.add_argument("--weights", required=True, type=str, help="Path to PyTorch weights (.pt)")
    parser.add_argument("--onnx", required=True, type=str, help="Path to ONNX model file")
    parser.add_argument("--source", required=True, type=str, help="Path to test image or directory")
    parser.add_argument("--image-size", type=int, default=512, help="Image resize for preprocessing")
    parser.add_argument("--max-images", type=int, default=32, help="Maximum number of images to load")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per backend")
    parser.add_argument("--runs", type=int, default=20, help="Benchmark iterations per backend")
    parser.add_argument("--proto-k", type=int, default=24, help="Prototype count for model creation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/benchmarks",
        help="Directory to store benchmark summary outputs",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA benchmarking even if GPU is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_benchmark_samples(args.source, image_size=args.image_size, max_images=args.max_images)
    if not samples:
        raise RuntimeError("No benchmark samples were loaded. Check the source path and extensions.")

    results: List[BackendResult] = []

    # PyTorch CPU
    results.append(
        _run_torch_backend(
            model_path=Path(args.weights),
            device_name="cpu",
            samples=samples,
            warmup_runs=args.warmup,
            benchmark_runs=args.runs,
            proto_k=args.proto_k,
        )
    )

    # PyTorch CUDA (if available)
    if torch.cuda.is_available() and not args.no_cuda:
        results.append(
            _run_torch_backend(
                model_path=Path(args.weights),
                device_name="cuda",
                samples=samples,
                warmup_runs=args.warmup,
                benchmark_runs=args.runs,
                proto_k=args.proto_k,
            )
        )
    else:
        results.append(
            BackendResult(
                name="pytorch-cuda",
                framework="pytorch",
                device="cuda",
                status="skipped",
                warmup_latencies=[],
                steady_latencies=[],
                error="CUDA not available or disabled",
            )
        )

    onnx_path = Path(args.onnx)

    # ONNX Runtime CPU
    results.append(
        _run_onnx_backend(
            onnx_path=onnx_path,
            provider="CPUExecutionProvider",
            samples=samples,
            warmup_runs=args.warmup,
            benchmark_runs=args.runs,
        )
    )

    # ONNX Runtime CUDA if available
    if not args.no_cuda:
        results.append(
            _run_onnx_backend(
                onnx_path=onnx_path,
                provider="CUDAExecutionProvider",
                samples=samples,
                warmup_runs=args.warmup,
                benchmark_runs=args.runs,
            )
        )
    else:
        results.append(
            BackendResult(
                name="onnx-cuda",
                framework="onnxruntime",
                device="CUDAExecutionProvider",
                status="skipped",
                warmup_latencies=[],
                steady_latencies=[],
                error="CUDA benchmarking disabled via --no-cuda",
            )
        )

    _write_outputs(Path(args.output_dir), results, samples, args)


if __name__ == "__main__":
    main()
