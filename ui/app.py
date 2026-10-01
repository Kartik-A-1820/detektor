"""Detektor console: a Gradio UI for in-process serving and standalone (remote backend) mode."""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import gradio as gr
import numpy as np
from PIL import Image

from api import __version__ as VERSION
from ui.render import (
    annotate_image,
    class_map_rows,
    detections_from_response,
    empty_state,
    fmt_metric,
    header_html,
    kpi_cards,
    latency_figure,
    overview_html,
    results_json,
    results_summary_html,
    summarize_detections,
    training_figure,
    validation_figure,
)
from ui.theme import blocks_kwargs, mount_kwargs  # noqa: F401  (mount_kwargs re-exported for serve.py)

DEFAULT_BACKEND_URL = os.getenv("DETEKTOR_UI_BACKEND", "http://localhost:8000")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: name -> (confidence threshold, NMS IoU threshold)
PRESETS: Dict[str, Tuple[float, float]] = {
    "Balanced": (0.25, 0.60),
    "High precision": (0.50, 0.50),
    "High recall": (0.10, 0.70),
}

_EMPTY_RESULTS = empty_state(
    "No results yet",
    "Add one or more images and press Run detection. Results, timings and a downloadable JSON appear here.",
)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _normalize_file_inputs(file_inputs: Optional[List[Any]]) -> List[Path]:
    paths: List[Path] = []
    for file_obj in file_inputs or []:
        path: Optional[str] = None
        if isinstance(file_obj, (str, Path)):
            path = str(file_obj)
        elif isinstance(file_obj, dict) and "name" in file_obj:
            path = file_obj["name"]
        elif hasattr(file_obj, "name"):
            path = file_obj.name
        if path:
            candidate = Path(path)
            if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(candidate)
    return paths


def _collect_image_paths(image_files: Optional[List[Any]], folder_path: str) -> List[Path]:
    seen: set[str] = set()
    results: List[Path] = []
    for path in _normalize_file_inputs(image_files):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            results.append(path)
    folder = (folder_path or "").strip()
    if folder:
        root = Path(folder).expanduser()
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    key = str(path.resolve())
                    if key not in seen:
                        seen.add(key)
                        results.append(path)
    return results


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _chunked(items: Sequence[Path], size: int) -> List[List[Path]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _write_download(payload: Dict[str, Any]) -> str:
    out_dir = Path(tempfile.gettempdir()) / "detektor_ui"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"detektor_results_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(results_json(payload), encoding="utf-8")
    return str(path)


def _find_benchmark_reports(limit: int = 20) -> List[Tuple[str, str]]:
    """(label, path) for result files in ``runs/benchmarks`` and ``benchmarks/results``, newest first."""
    found: List[Path] = []
    for pattern in ("runs/benchmarks/*/results.json", "benchmarks/results/*.json"):
        found.extend(Path.cwd().glob(pattern))
    found = sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return [(f"{p.parent.name}/{p.name}" if p.name == "results.json" else p.name, str(p)) for p in found]


# ---------------------------------------------------------------------------
# Dashboard state -> component updates
# ---------------------------------------------------------------------------

def _dashboard_outputs_from_state(state: Dict[str, Any]) -> Tuple[Any, ...]:
    available = list(state.get("available_checkpoints", {}).keys())
    active = state.get("active_checkpoint_key") or (available[0] if available else None)
    plots = state.get("plots", {})
    validation_rows = [
        [
            row.get("epoch"),
            fmt_metric(row.get("val_precision")),
            fmt_metric(row.get("val_recall")),
            fmt_metric(row.get("val_map50")),
            fmt_metric(row.get("val_mean_iou")),
        ]
        for row in state.get("validation_history", [])
    ]
    training_json = {
        "training": state.get("training_summary", {}),
        "validation": state.get("validation_summary", {}),
        "checkpoint": state.get("checkpoint_summary", {}),
        "runtime": state.get("runtime", {}),
    }
    return (
        header_html(state, VERSION),
        gr.update(choices=available, value=active),
        overview_html(state),
        class_map_rows(state.get("class_map", {})),
        state.get("dataset", {}),
        training_json,
        training_figure(state),
        validation_figure(state),
        validation_rows,
        plots.get("loss_total"),
        plots.get("loss_components"),
        plots.get("learning_rate"),
    )


class DetektorUIRuntime:
    """Bridge from Gradio callbacks to the active in-process inference runtime."""

    def __init__(
        self,
        *,
        get_runtime_state: Callable[[], Dict[str, Any]],
        get_service_snapshot: Callable[[], Tuple[Any, Any]],
        select_checkpoint: Callable[[str], Dict[str, Any]],
    ) -> None:
        self._get_runtime_state = get_runtime_state
        self._get_service_snapshot = get_service_snapshot
        self._select_checkpoint = select_checkpoint

    # -- dashboard ---------------------------------------------------------
    def refresh_dashboard(self) -> Tuple[Any, ...]:
        return _dashboard_outputs_from_state(self._get_runtime_state())

    def set_checkpoint(self, checkpoint_key: str) -> Tuple[Any, ...]:
        if checkpoint_key and checkpoint_key != self._get_runtime_state().get("active_checkpoint_key"):
            gr.Info(f"Switching to the '{checkpoint_key}' checkpoint…")
            self._select_checkpoint(checkpoint_key)
        return self.refresh_dashboard()

    # -- inference ---------------------------------------------------------
    def run_inference(
        self,
        image_files: Optional[List[Any]],
        folder_path: str,
        conf_thresh: float,
        iou_thresh: float,
        max_det: int,
        include_masks: bool,
        progress: gr.Progress = gr.Progress(),
    ) -> Tuple[str, Any, List[List[Any]], Dict[str, Any], Optional[str]]:
        image_paths = _collect_image_paths(image_files, folder_path)
        if not image_paths:
            gr.Warning("Add at least one image (or a folder path) first.")
            return _EMPTY_RESULTS, gr.update(value=[], visible=False), [], {}, None

        service, config = self._get_service_snapshot()
        state = self._get_runtime_state()
        class_map = state.get("class_map", {})
        batch_size = max(1, int(getattr(config, "max_batch_size", 16)))
        batches = math.ceil(len(image_paths) / batch_size)

        gallery: List[Tuple[Image.Image, str]] = []
        rows: List[List[Any]] = []
        raw: List[Dict[str, Any]] = []
        total_dets = 0
        model_ms = 0.0
        started = time.perf_counter()

        for index, batch_paths in enumerate(_chunked(image_paths, batch_size), start=1):
            progress((index - 1) / batches, desc=f"Detecting… batch {index}/{batches}")
            originals = [_load_rgb_image(p) for p in batch_paths]
            predictions, latency = service.predict_batch(
                images_bytes=[p.read_bytes() for p in batch_paths],
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                max_det=int(max_det),
                include_masks=include_masks,
            )
            model_ms += latency
            for path, original, prediction in zip(batch_paths, originals, predictions):
                dets = detections_from_response(prediction)
                total_dets += len(dets)
                gallery.append(
                    (
                        annotate_image(original, dets, include_masks, class_map),
                        f"{path.name} — {summarize_detections(dets, class_map)}",
                    )
                )
                rows.append([path.name, len(dets), summarize_detections(dets, class_map)])
                raw.append({"image": str(path), "prediction": prediction})

        wall_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "checkpoint": state.get("active_checkpoint_key"),
            "settings": {"conf_thresh": conf_thresh, "iou_thresh": iou_thresh, "max_det": int(max_det), "masks": include_masks},
            "images": len(image_paths),
            "total_detections": total_dets,
            "model_ms": round(model_ms, 2),
            "wall_ms": round(wall_ms, 2),
            "predictions": raw,
        }
        summary = results_summary_html(len(image_paths), total_dets, model_ms, wall_ms, state.get("active_checkpoint_key"))
        return summary, gr.update(value=gallery, visible=True), rows, payload, _write_download(payload)

    # -- benchmark ---------------------------------------------------------
    def run_benchmark(self, batch_size: int, runs: int, progress: gr.Progress = gr.Progress()) -> Tuple[str, Any]:
        import torch

        service, _ = self._get_service_snapshot()
        batch, runs = int(batch_size), int(runs)
        size = int(service.image_size)
        x = torch.rand(batch, 3, size, size, device=service.device)
        sizes = [(size, size)] * batch
        latencies: List[float] = []
        with torch.no_grad():
            for _ in range(3):  # warm-up, not measured
                service.model.predict(x, original_sizes=sizes)
            for i in range(runs):
                progress((i + 1) / runs, desc=f"Benchmark run {i + 1}/{runs}")
                start = time.perf_counter()
                service.model.predict(x, original_sizes=sizes)
                if str(service.device).startswith("cuda"):
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000.0)
        arr = np.asarray(latencies)
        mean = float(arr.mean())
        cards = kpi_cards(
            [
                ("Median (p50)", f"{np.percentile(arr, 50):.1f} ms", f"batch of {batch}"),
                ("p95", f"{np.percentile(arr, 95):.1f} ms", ""),
                ("p99", f"{np.percentile(arr, 99):.1f} ms", ""),
                ("Per image", f"{mean / batch:.1f} ms", f"{size}×{size} px"),
                ("Throughput", f"{batch * 1000.0 / mean:.1f} img/s", str(service.device)),
            ]
        )
        return cards, latency_figure(latencies)

    @staticmethod
    def load_report(path: Optional[str]) -> str:
        if not path:
            return "_No saved benchmark reports found. Generate one with `python -m benchmarks run`._"
        from benchmarks.report import render_markdown

        return render_markdown(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Remote-backend client (standalone mode)
# ---------------------------------------------------------------------------

def _encode_multipart_formdata(fields: Dict[str, str], files: Sequence[Tuple[str, str, bytes, str]]) -> Tuple[bytes, str]:
    boundary = "detektor-ui-boundary"
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += str(value).encode() + b"\r\n"
    for field_name, filename, content, content_type in files:
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode()
        body += f"Content-Type: {content_type}\r\n\r\n".encode() + content + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _post_json_multipart(
    url: str,
    files: Sequence[Tuple[str, str, bytes, str]],
    params: Dict[str, Any],
    timeout: int,
    api_key: str = "",
) -> Dict[str, Any]:
    query = urlencode({k: v for k, v in params.items() if v is not None})
    body, content_type = _encode_multipart_formdata({}, files)
    headers = {"Content-Type": content_type, "Accept": "application/json"}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    request = Request(f"{url}?{query}" if query else url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is operator supplied
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        hint = " (check the API key)" if exc.code == 401 else ""
        raise RuntimeError(f"HTTP {exc.code}{hint}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the backend: {exc.reason}") from exc


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _parse_class_map(text: str) -> Dict[str, str]:
    if not (text or "").strip():
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in payload.items()} if isinstance(payload, dict) else {}


def run_single_inference(image, backend_url, api_key, conf_thresh, iou_thresh, max_det, include_masks, class_map_text):
    if image is None:
        gr.Warning("Upload an image first.")
        return None, [], {}, _EMPTY_RESULTS
    class_map = _parse_class_map(class_map_text)
    try:
        started = time.perf_counter()
        resp = _post_json_multipart(
            f"{(backend_url or DEFAULT_BACKEND_URL).strip().rstrip('/')}/v1/predict",
            [("image", "upload.png", _png_bytes(image), "image/png")],
            {"conf_thresh": conf_thresh, "iou_thresh": iou_thresh, "max_det": int(max_det), "include_masks": include_masks},
            timeout=60,
            api_key=api_key or "",
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:  # noqa: BLE001
        gr.Warning(str(exc))
        return None, [], {"error": str(exc)}, empty_state("Request failed", str(exc), "⚠️")
    dets = detections_from_response(resp)
    rows = [[i, class_map.get(str(d.get("label")), d.get("label")), round(float(d.get("score") or 0), 3)] for i, d in enumerate(dets)]
    summary = results_summary_html(1, len(dets), float(resp.get("inference_time_ms") or wall_ms), wall_ms, "remote")
    return annotate_image(image, dets, include_masks, class_map), rows, resp, summary


def run_batch_inference(files, backend_url, api_key, conf_thresh, iou_thresh, max_det, include_masks, class_map_text):
    paths = _normalize_file_inputs(files)
    if not paths:
        gr.Warning("Upload at least one image first.")
        return [], {}, _EMPTY_RESULTS
    class_map = _parse_class_map(class_map_text)
    images = [_load_rgb_image(p) for p in paths]
    try:
        started = time.perf_counter()
        resp = _post_json_multipart(
            f"{(backend_url or DEFAULT_BACKEND_URL).strip().rstrip('/')}/v1/predict_batch",
            [("images", p.name, _png_bytes(img), "image/png") for p, img in zip(paths, images)],
            {"conf_thresh": conf_thresh, "iou_thresh": iou_thresh, "max_det": int(max_det), "include_masks": include_masks},
            timeout=120,
            api_key=api_key or "",
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:  # noqa: BLE001
        gr.Warning(str(exc))
        return [], {"error": str(exc)}, empty_state("Request failed", str(exc), "⚠️")
    gallery, total = [], 0
    for path, image, pred in zip(paths, images, resp.get("predictions", [])):
        dets = detections_from_response(pred)
        total += len(dets)
        gallery.append((annotate_image(image, dets, include_masks, class_map), f"{path.name} — {summarize_detections(dets, class_map)}"))
    model_ms = float(resp.get("total_inference_time_ms") or wall_ms)
    return gallery, resp, results_summary_html(len(paths), total, model_ms, wall_ms, "remote")


# ---------------------------------------------------------------------------
# Interface construction
# ---------------------------------------------------------------------------

def _settings_block() -> Tuple[Any, Any, Any, Any, Any]:
    """Shared detection-settings controls; returns (preset, conf, iou, max_det, masks)."""
    preset = gr.Radio(list(PRESETS), value="Balanced", label="Preset", info="Quick starting points; fine-tune below.")
    with gr.Row():
        conf = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Confidence", info="Higher = fewer, surer boxes")
        iou = gr.Slider(0.1, 0.9, value=0.6, step=0.01, label="NMS IoU", info="Lower = merge overlapping boxes")
    with gr.Row():
        max_det = gr.Slider(1, 300, value=100, step=1, label="Max detections")
        masks = gr.Checkbox(value=False, label="Draw segmentation masks")

    def apply_preset(name: str) -> Tuple[float, float]:
        return PRESETS.get(name, PRESETS["Balanced"])

    preset.change(apply_preset, inputs=preset, outputs=[conf, iou])
    return preset, conf, iou, max_det, masks


def _footer() -> gr.HTML:
    return gr.HTML(
        f'<div class="det-footer">Detektor v{VERSION} · '
        '<a href="/docs" target="_blank">API docs</a> · '
        '<a href="https://github.com/Kartik-A-1820/detektor" target="_blank">GitHub</a></div>'
    )


_ABOUT_MD = """
### REST API

The same model behind this console is available over HTTP. Interactive docs live at **`/docs`**.

| Endpoint | Purpose |
|---|---|
| `POST /v1/predict` | Detect on one image (`multipart/form-data`, field `image`) |
| `POST /v1/predict_batch` | Detect on several images (field `images`) |
| `GET /health` · `/ready` · `/version` | Liveness, readiness and version probes |
| `GET /metrics` · `/metrics/prometheus` | JSON / Prometheus metrics |
| `GET /runtime` · `POST /runtime/select_model` | Active run metadata and checkpoint switching |

```bash
curl -X POST "http://localhost:8000/v1/predict?conf_thresh=0.25" \\
     -H "X-API-Key: $DETEKTOR_API_KEY" \\
     -F "image=@photo.jpg"
```

### Tips
- **Presets** set sensible confidence/NMS pairs; *High recall* finds more objects, *High precision* trades recall for fewer false positives.
- Open the **Benchmark** tab to measure latency and throughput of the loaded model on this machine.
- Switch between `best` and `last` checkpoints from the **Model** tab without restarting.
"""


def build_interface(runtime: Optional[DetektorUIRuntime] = None) -> gr.Blocks:
    if runtime is None:
        return _build_remote_interface()

    with gr.Blocks(title="Detektor", **blocks_kwargs()) as demo:
        header = gr.HTML(header_html({}, VERSION))

        with gr.Tabs():
            # ------------------------------------------------------------- Detect
            with gr.Tab("🔍  Detect"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=340):
                        upload = gr.Files(label="Images", file_count="multiple", type="filepath",
                                          file_types=sorted(IMAGE_EXTENSIONS), height=190)
                        with gr.Accordion("Detection settings", open=True):
                            _, conf, iou, max_det, masks = _settings_block()
                        with gr.Accordion("Run on a server folder", open=False):
                            folder = gr.Textbox(label="Folder path", placeholder="/data/images",
                                                info="Every image under this folder (recursive) is processed.")
                        with gr.Row():
                            run_btn = gr.Button("Run detection", variant="primary", size="lg", elem_classes="det-run")
                            clear_btn = gr.Button("Clear", variant="secondary")
                    with gr.Column(scale=7):
                        results_summary = gr.HTML(_EMPTY_RESULTS)
                        gallery = gr.Gallery(label="Annotated results", columns=2, height=480, preview=False,
                                             object_fit="contain", show_label=False, visible=False)
                        with gr.Accordion("Per-image details & export", open=False):
                            table = gr.Dataframe(headers=["Image", "Detections", "Summary"], datatype=["str", "number", "str"],
                                                 interactive=False, wrap=True)
                            download = gr.File(label="Download results (JSON)", interactive=False)
                            payload = gr.JSON(label="Raw payload", open=False)

            # -------------------------------------------------------------- Model
            with gr.Tab("🧠  Model"):
                overview = gr.HTML()
                with gr.Row():
                    selector = gr.Dropdown(label="Active checkpoint", choices=[], value=None, scale=3,
                                           info="Switching reloads the model in place — no restart needed.")
                    refresh = gr.Button("Refresh", variant="secondary", scale=1)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        class_table = gr.Dataframe(headers=["ID", "Class"], datatype=["number", "str"], interactive=False,
                                                   label="Classes")
                    with gr.Column(scale=3):
                        dataset_json = gr.JSON(label="Dataset", open=False)
                        training_json = gr.JSON(label="Training, validation & checkpoint metadata", open=False)

            # ----------------------------------------------------------- Training
            with gr.Tab("📈  Training"):
                with gr.Row():
                    train_plot = gr.Plot(show_label=False)
                with gr.Row():
                    val_plot = gr.Plot(show_label=False)
                val_table = gr.Dataframe(headers=["Epoch", "Precision", "Recall", "mAP50", "Mean IoU"],
                                         datatype=["number", "str", "str", "str", "str"], interactive=False,
                                         label="Validation history")
                with gr.Accordion("Saved training plots", open=False):
                    with gr.Row():
                        loss_img = gr.Image(label="Loss", type="filepath", interactive=False)
                        comp_img = gr.Image(label="Loss components", type="filepath", interactive=False)
                        lr_img = gr.Image(label="Learning rate", type="filepath", interactive=False)

            # ----------------------------------------------------------- Benchmark
            with gr.Tab("⚡  Benchmark"):
                gr.HTML('<div class="det-note">Measure latency and throughput of the <b>loaded model</b> on this machine. '
                        'The benchmark shares the model with the API, so it may briefly delay concurrent requests.</div>')
                with gr.Row():
                    bench_batch = gr.Slider(1, 16, value=1, step=1, label="Batch size")
                    bench_runs = gr.Slider(5, 200, value=30, step=5, label="Timed runs")
                    bench_btn = gr.Button("Run live benchmark", variant="primary", elem_classes="det-run")
                bench_cards = gr.HTML(empty_state("Not run yet", "Press “Run live benchmark”.", "⚡"))
                bench_plot = gr.Plot(show_label=False)
                with gr.Accordion("Saved benchmark reports", open=False):
                    reports = _find_benchmark_reports()
                    report_pick = gr.Dropdown(label="Report", choices=reports, value=reports[0][1] if reports else None)
                    report_md = gr.Markdown(DetektorUIRuntime.load_report(reports[0][1] if reports else None))
                    report_pick.change(DetektorUIRuntime.load_report, inputs=report_pick, outputs=report_md)

            # --------------------------------------------------------------- About
            with gr.Tab("ℹ️  API & help"):
                gr.Markdown(_ABOUT_MD)

        _footer()

        dashboard = [header, selector, overview, class_table, dataset_json, training_json, train_plot, val_plot, val_table,
                     loss_img, comp_img, lr_img]
        demo.load(runtime.refresh_dashboard, outputs=dashboard)
        refresh.click(runtime.refresh_dashboard, outputs=dashboard)
        selector.change(runtime.set_checkpoint, inputs=[selector], outputs=dashboard, show_progress="minimal")
        run_btn.click(
            runtime.run_inference,
            inputs=[upload, folder, conf, iou, max_det, masks],
            outputs=[results_summary, gallery, table, payload, download],
        )
        clear_btn.click(
            lambda: (None, "", _EMPTY_RESULTS, gr.update(value=None, visible=False), [], {}, None),
            outputs=[upload, folder, results_summary, gallery, table, payload, download],
        )
        bench_btn.click(runtime.run_benchmark, inputs=[bench_batch, bench_runs], outputs=[bench_cards, bench_plot])

    demo.queue(default_concurrency_limit=1)
    return demo


def _build_remote_interface() -> gr.Blocks:
    with gr.Blocks(title="Detektor (remote)", **blocks_kwargs()) as demo:
        gr.HTML(header_html({"device": "remote"}, VERSION))
        with gr.Accordion("Backend connection", open=False):
            with gr.Row():
                backend = gr.Textbox(value=DEFAULT_BACKEND_URL, label="Backend URL", info="Where `serve.py` is running")
                api_key = gr.Textbox(label="API key", type="password", info="Only if the server sets DETEKTOR_API_KEY")
            class_map = gr.Textbox(label="Class map (JSON)", placeholder='{"0": "person", "1": "car"}',
                                   info="Optional: show names instead of class ids")
        with gr.Accordion("Detection settings", open=False):
            _, conf, iou, max_det, masks = _settings_block()

        with gr.Tabs():
            with gr.Tab("🖼️  Single image"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4):
                        image_in = gr.Image(type="pil", label="Image", height=360)
                        run_one = gr.Button("Run detection", variant="primary", size="lg", elem_classes="det-run")
                    with gr.Column(scale=6):
                        one_summary = gr.HTML(_EMPTY_RESULTS)
                        image_out = gr.Image(label="Annotated image", type="pil", height=420)
                        with gr.Accordion("Detections & raw JSON", open=False):
                            one_table = gr.Dataframe(headers=["#", "Class", "Score"], interactive=False)
                            one_json = gr.JSON(label="Response", open=False)
                run_one.click(run_single_inference,
                              inputs=[image_in, backend, api_key, conf, iou, max_det, masks, class_map],
                              outputs=[image_out, one_table, one_json, one_summary])
            with gr.Tab("🗂️  Batch"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4):
                        files_in = gr.Files(label="Images", file_count="multiple", type="filepath",
                                            file_types=sorted(IMAGE_EXTENSIONS), height=220)
                        run_many = gr.Button("Run batch", variant="primary", size="lg", elem_classes="det-run")
                    with gr.Column(scale=6):
                        many_summary = gr.HTML(_EMPTY_RESULTS)
                        many_gallery = gr.Gallery(label="Annotated results", columns=2, height=440, object_fit="contain",
                                                  show_label=False)
                        many_json = gr.JSON(label="Response", open=False)
                run_many.click(run_batch_inference,
                               inputs=[files_in, backend, api_key, conf, iou, max_det, masks, class_map],
                               outputs=[many_gallery, many_json, many_summary])
        _footer()
    demo.queue(default_concurrency_limit=1)
    return demo


def main() -> None:
    demo = build_interface()
    demo.launch(
        server_name=os.getenv("DETEKTOR_UI_HOST", "127.0.0.1"),
        server_port=int(os.getenv("DETEKTOR_UI_PORT", "7860")),
        **mount_kwargs(),
    )


if __name__ == "__main__":
    main()
