"""Render benchmark results as Markdown and flatten them for regression comparison."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "–"
        if isinstance(value, float):
            return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.4f}".rstrip("0").rstrip(".") or "0"
        return str(value)

    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def _get(d: Dict[str, Any], *path: str) -> Any:
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)  # type: ignore[assignment]
    return d


def _complexity(r: Dict[str, Any]) -> str:
    return _table(
        ["Profile", "Input", "Params (M)", "GFLOPs", "FP32 size (MB)", "FP16 size (MB)"],
        [[x["profile"], x["img_size"], x["params_m"], x["gflops"], x["size_fp32_mb"], x["size_fp16_mb"]] for x in r["rows"]],
    )


def _latency(r: Dict[str, Any]) -> str:
    rows = []
    for x in r["rows"]:
        rows.append([
            x["profile"], x["img_size"], _get(x, "preprocess", "p50_ms"), _get(x, "forward", "p50_ms"),
            _get(x, "predict_default", "p50_ms"), _get(x, "predict_default", "p95_ms"), _get(x, "predict_default", "p99_ms"),
            _get(x, "predict_dense", "p50_ms"), _get(x, "predict_dense", "postprocess_p50_ms"), _get(x, "predict_default", "fps"),
        ])
    return _table(
        ["Profile", "Input", "Decode+resize p50", "Forward p50", "Predict p50", "Predict p95", "Predict p99",
         "Dense predict p50", "Dense postproc.", "FPS"], rows) + (
        "\n\n_All latencies in ms, batch 1. “Dense” lowers the confidence threshold to 0.001 so top‑k, NMS and mask "
        "composition run at their worst‑case workload (a randomly initialised model emits no confident detections)._")


def _throughput(r: Dict[str, Any]) -> str:
    return _table(
        ["Profile", "Input", "Batch", "Batch latency (ms)", "ms / image", "Images / s"],
        [[x["profile"], x["img_size"], x["batch"], _get(x, "batch_latency", "mean_ms"), x.get("ms_per_image"),
          x.get("images_per_s") if "error" not in x else f"error: {x['error']}"] for x in r["rows"]],
    )


def _memory(r: Dict[str, Any]) -> str:
    return _table(
        ["Profile", "Input", "Inference peak (MB)", "Inference Δ (MB)", "Train batch", "Train peak (MB)", "Train Δ (MB)"],
        [[x["profile"], x["img_size"], x.get("inference_peak_mb"), x.get("inference_delta_mb"), x.get("train_batch"),
          x.get("train_peak_mb"), x.get("train_delta_mb")] for x in r["rows"]],
    ) + f"\n\n_Metric: {r.get('metric')}. “Δ” is the increase over the pre‑measurement baseline._"


def _training(r: Dict[str, Any]) -> str:
    return _table(
        ["Profile", "Input", "Batch", "Step p50 (ms)", "Step p95 (ms)", "Images / s", "Loss finite"],
        [[x["profile"], x["img_size"], x["batch"], _get(x, "step", "p50_ms"), _get(x, "step", "p95_ms"),
          x.get("images_per_s"), x.get("loss_finite")] for x in r["rows"]],
    )


def _startup(r: Dict[str, Any]) -> str:
    return _table(
        ["Profile", "Checkpoint (MB)", "Load (ms)", "1st inference (ms)", "2nd inference (ms)", "Warm‑up penalty (ms)"],
        [[x["profile"], x["checkpoint_mb"], _get(x, "load_ms", "mean_ms"), x["first_inference_ms"],
          x["second_inference_ms"], x["warmup_penalty_ms"]] for x in r["rows"]],
    )


def _onnx(r: Dict[str, Any]) -> str:
    if r.get("skipped"):
        return f"_Skipped: {r['skipped']}_"
    return _table(
        ["Profile", "Input", "Export (s)", "ONNX (MB)", "Max |Δ| vs PyTorch", "PyTorch p50 (ms)", "ORT p50 (ms)", "Speed‑up"],
        [[x["profile"], x["img_size"], x.get("export_s"), x.get("onnx_mb"),
          f"{x['max_abs_diff']:.1e}" if "max_abs_diff" in x else None,
          _get(x, "pytorch_cpu", "p50_ms"), _get(x, "onnxruntime_cpu", "p50_ms"),
          x.get("speedup_vs_pytorch") if "error" not in x else f"error: {x['error']}"] for x in r["rows"]],
    )


def _api(r: Dict[str, Any]) -> str:
    if r.get("error"):
        return f"_Error: {r['error']}_"
    head = (f"Profile `{r['profile']}`, input {r['img_size']}px, device `{r['device']}`, payload {r['payload']}, "
            f"server concurrency {r['server_max_concurrency']}.\n\n")
    return head + _table(
        ["Clients", "Requests", "Errors", "Req / s", "p50 (ms)", "p95 (ms)", "p99 (ms)"],
        [[x["concurrency"], x["requests"], x["errors"], x["rps"], _get(x, "latency", "p50_ms"),
          _get(x, "latency", "p95_ms"), _get(x, "latency", "p99_ms")] for x in r["rows"]],
    )


def _metrics_block(m: Dict[str, Any]) -> str:
    return _table(
        ["Precision", "Recall", "F1", "mAP50", "mAP50‑95", "Mean box IoU", "Mean mask IoU", "Images"],
        [[m["precision"], m["recall"], m["f1"], m["map50"], m["ap50_95"], m["mean_box_iou"], m["mean_mask_iou"], m["num_images"]]],
    )


def _e2e(r: Dict[str, Any]) -> str:
    head = (f"Profile `{r['profile']}`, {r['train_images']} train / {r['val_images']} val synthetic images at {r['img_size']}px, "
            f"{r['epochs']} epochs on `{r['device']}` — {r['train_wall_s']} s total ({r['train_images_per_s']} img/s).\n\n")
    return head + _metrics_block(r["metrics"])


def _robustness(r: Dict[str, Any]) -> str:
    if r.get("skipped"):
        return f"_Skipped: {r['skipped']}_"
    return _table(
        ["Perturbation", "mAP50", "Retention vs clean", "Mean detections"],
        [[x["perturbation"], x["map50"], x["retention"], x["mean_detections"]] for x in r["rows"]],
    ) + f"\n\n_{r.get('images')} images at {r.get('img_size')}px. {r.get('note', '')}_"


def _accuracy(r: Dict[str, Any]) -> str:
    if r.get("skipped"):
        return f"_Skipped: {r['skipped']}_"
    return _metrics_block(r["metrics"])


RENDERERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "complexity": _complexity, "latency": _latency, "throughput": _throughput, "memory": _memory,
    "training": _training, "startup": _startup, "onnx": _onnx, "api": _api, "e2e": _e2e,
    "robustness": _robustness, "accuracy": _accuracy,
}

TITLES = {
    "complexity": "Model complexity", "latency": "Inference latency", "throughput": "Batch throughput",
    "memory": "Memory footprint", "training": "Training throughput", "startup": "Cold start",
    "onnx": "ONNX Runtime vs PyTorch", "api": "HTTP API load test", "e2e": "End-to-end quality (synthetic data)",
    "robustness": "Robustness to corruptions", "accuracy": "Accuracy on user dataset",
}


def render_markdown(results: Dict[str, Any]) -> str:
    env = results.get("environment", {})
    lines = ["# Detektor benchmark report", ""]
    lines.append("## Environment")
    lines.append("")
    keys = ["timestamp_utc", "git_revision", "cpu_model", "cpu_cores_physical", "cpu_cores_logical", "ram_gb", "platform",
            "python", "torch", "torch_threads", "onnxruntime", "device", "gpu_name", "gpu_vram_gb"]
    lines.append(_table(["Key", "Value"], [[k, env[k]] for k in keys if env.get(k) is not None]))
    cfg = results.get("config", {})
    if cfg:
        lines += ["", "## Configuration", "", "```json", __import__("json").dumps(cfg, indent=2, default=str), "```"]
    for name, suite in results.get("suites", {}).items():
        lines += ["", f"## {TITLES.get(name, name)}", ""]
        status = suite.get("status")
        if status == "error":
            lines.append(f"**Suite failed:** `{suite.get('error')}`")
            continue
        lines.append(f"_{suite.get('description', '')}_ — ran in {suite.get('duration_s', '?')} s")
        lines.append("")
        renderer = RENDERERS.get(name)
        try:
            lines.append(renderer(suite) if renderer else "```json\n" + __import__("json").dumps(suite, indent=2) + "\n```")
        except Exception as exc:  # noqa: BLE001  - never lose the whole report to one bad suite
            lines.append(f"_Could not render suite: {exc}_")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Flattening for `compare`
# ---------------------------------------------------------------------------

Metric = Tuple[float, str]  # (value, "lower" | "higher" is better)


def flatten_metrics(results: Dict[str, Any]) -> Dict[str, Metric]:
    """Extract the comparable headline numbers from a results document."""
    out: Dict[str, Metric] = {}
    suites = results.get("suites", {})

    def put(key: str, value: Any, better: str) -> None:
        if isinstance(value, (int, float)) and value == value:
            out[key] = (float(value), better)

    for x in suites.get("latency", {}).get("rows", []):
        base = f"latency/{x['profile']}@{x['img_size']}"
        put(f"{base}/forward_p50_ms", _get(x, "forward", "p50_ms"), "lower")
        put(f"{base}/predict_p50_ms", _get(x, "predict_default", "p50_ms"), "lower")
        put(f"{base}/predict_p95_ms", _get(x, "predict_default", "p95_ms"), "lower")
    for x in suites.get("throughput", {}).get("rows", []):
        put(f"throughput/{x['profile']}@{x['img_size']}/b{x['batch']}/images_per_s", x.get("images_per_s"), "higher")
    for x in suites.get("memory", {}).get("rows", []):
        put(f"memory/{x['profile']}/inference_delta_mb", x.get("inference_delta_mb"), "lower")
    for x in suites.get("training", {}).get("rows", []):
        put(f"training/{x['profile']}/images_per_s", x.get("images_per_s"), "higher")
    for x in suites.get("startup", {}).get("rows", []):
        put(f"startup/{x['profile']}/load_ms", _get(x, "load_ms", "mean_ms"), "lower")
    for x in suites.get("onnx", {}).get("rows", []):
        put(f"onnx/{x['profile']}/ort_p50_ms", _get(x, "onnxruntime_cpu", "p50_ms"), "lower")
    for x in suites.get("api", {}).get("rows", []):
        put(f"api/c{x['concurrency']}/rps", x.get("rps"), "higher")
        put(f"api/c{x['concurrency']}/p95_ms", _get(x, "latency", "p95_ms"), "lower")
    for key in ("map50", "ap50_95", "recall", "precision"):
        put(f"e2e/{key}", _get(suites.get("e2e", {}), "metrics", key), "higher")
        put(f"accuracy/{key}", _get(suites.get("accuracy", {}), "metrics", key), "higher")
    for x in suites.get("robustness", {}).get("rows", []):
        put(f"robustness/{x['perturbation']}/map50", x.get("map50"), "higher")
    return out


def compare(base: Dict[str, Any], new: Dict[str, Any], threshold_pct: float = 15.0) -> Tuple[str, List[str]]:
    """Return a Markdown diff table and the list of regressions beyond ``threshold_pct``."""
    a, b = flatten_metrics(base), flatten_metrics(new)
    rows, regressions = [], []
    for key in sorted(set(a) & set(b)):
        (va, better), (vb, _) = a[key], b[key]
        if va == 0:
            continue
        delta = (vb - va) / abs(va) * 100.0
        worse = delta > threshold_pct if better == "lower" else delta < -threshold_pct
        better_flag = delta < -threshold_pct if better == "lower" else delta > threshold_pct
        status = "❌ regression" if worse else ("✅ improved" if better_flag else "·")
        if worse:
            regressions.append(f"{key}: {va:g} → {vb:g} ({delta:+.1f}%)")
        rows.append([key, f"{va:g}", f"{vb:g}", f"{delta:+.1f}%", better + " is better", status])
    only = sorted(set(a) ^ set(b))
    md = ["# Benchmark comparison", "", f"Regression threshold: {threshold_pct:g}%", ""]
    md.append(_table(["Metric", "Base", "New", "Δ", "Direction", "Status"], rows) if rows else "_No overlapping metrics._")
    if only:
        md += ["", f"_{len(only)} metric(s) present in only one run were ignored._"]
    return "\n".join(md) + "\n", regressions
