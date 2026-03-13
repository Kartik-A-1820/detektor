"""Detektor UI for direct in-process serving and standalone backend mode."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_BACKEND_URL = os.getenv("DETEKTOR_UI_BACKEND", "http://localhost:8000")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_FONT = None
_COLOR_PALETTE = np.array(
    [
        (217, 48, 37),
        (30, 136, 229),
        (67, 160, 71),
        (251, 140, 0),
        (106, 27, 154),
        (0, 137, 123),
        (84, 110, 122),
        (229, 57, 53),
    ],
    dtype=np.uint8,
)
_UI_CSS = """
:root {
  --det-bg: linear-gradient(135deg, #f4efe5 0%, #fbf8f2 48%, #e8f0ea 100%);
  --det-panel: rgba(255, 252, 247, 0.88);
  --det-line: rgba(67, 80, 71, 0.16);
  --det-ink: #1f2a24;
  --det-muted: #627066;
  --det-accent: #c25b2d;
  --det-accent-2: #2b7a78;
}
.gradio-container {
  background: var(--det-bg);
}
.det-shell {
  border: 1px solid var(--det-line);
  border-radius: 24px;
  background: var(--det-panel);
  backdrop-filter: blur(10px);
  padding: 18px;
  box-shadow: 0 20px 60px rgba(43, 59, 50, 0.08);
}
.det-hero h1 {
  margin: 0;
  font-size: 2rem;
  color: var(--det-ink);
}
.det-hero p {
  margin: 6px 0 0;
  color: var(--det-muted);
}
.det-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
}
.det-card {
  border: 1px solid var(--det-line);
  border-radius: 18px;
  padding: 14px;
  background: rgba(255, 255, 255, 0.66);
}
.det-card-label {
  color: var(--det-muted);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.det-card-value {
  color: var(--det-ink);
  font-size: 1.1rem;
  font-weight: 700;
  margin-top: 6px;
}
"""


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

    def refresh_dashboard(self) -> Tuple[Any, ...]:
        return _dashboard_outputs_from_state(self._get_runtime_state())

    def set_checkpoint(self, checkpoint_key: str) -> Tuple[Any, ...]:
        if checkpoint_key:
            self._select_checkpoint(checkpoint_key)
        return self.refresh_dashboard()

    def run_gallery_inference(
        self,
        image_files: Optional[List[Any]],
        folder_path: str,
        conf_thresh: float,
        iou_thresh: float,
        max_det: int,
        include_masks: bool,
        progress: gr.Progress = gr.Progress(),
    ) -> Tuple[List[Tuple[Image.Image, str]], List[List[Any]], Dict[str, Any], str]:
        service, config = self._get_service_snapshot()
        state = self._get_runtime_state()
        class_map = state.get("class_map", {})
        image_paths = _collect_image_paths(image_files, folder_path)
        if not image_paths:
            return [], [], {}, "Provide image files or a folder path containing images."

        max_batch_size = max(1, int(getattr(config, "max_batch_size", 16)))
        gallery: List[Tuple[Image.Image, str]] = []
        rows: List[List[Any]] = []
        raw_predictions: List[Dict[str, Any]] = []
        total_model_latency = 0.0
        total_started = time.perf_counter()

        for batch_index, batch_paths in enumerate(_chunked(image_paths, max_batch_size), start=1):
            progress(
                batch_index / math.ceil(len(image_paths) / max_batch_size),
                desc=f"Running batch {batch_index}",
            )
            images_bytes = [path.read_bytes() for path in batch_paths]
            originals = [_load_rgb_image(path) for path in batch_paths]
            predictions, batch_latency = service.predict_batch(
                images_bytes=images_bytes,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                max_det=max_det,
                include_masks=include_masks,
            )
            total_model_latency += batch_latency
            for image_path, original_image, prediction in zip(batch_paths, originals, predictions):
                detections = _detections_from_response(prediction)
                annotated = _annotate_image(original_image, detections, include_masks, class_map)
                raw_predictions.append(
                    {
                        "image": str(image_path),
                        "prediction": prediction,
                    }
                )
                rows.append(
                    [
                        image_path.name,
                        prediction.get("num_detections", 0),
                        _summarize_detections(detections, class_map),
                    ]
                )
                gallery.append((annotated, _gallery_caption(image_path.name, detections, class_map)))

        wall_time_ms = (time.perf_counter() - total_started) * 1000.0
        summary = {
            "checkpoint": state.get("active_checkpoint_key"),
            "images": len(image_paths),
            "total_model_latency_ms": round(total_model_latency, 2),
            "wall_time_ms": round(wall_time_ms, 2),
            "predictions": raw_predictions,
        }
        latency_text = (
            f"Processed {len(image_paths)} image(s) with '{state.get('active_checkpoint_key')}' "
            f"| model {total_model_latency:.2f} ms | wall {wall_time_ms:.2f} ms"
        )
        return gallery, rows, summary, latency_text


def _encode_multipart_formdata(
    fields: Dict[str, str],
    files: Sequence[Tuple[str, str, bytes, str]],
) -> Tuple[bytes, str]:
    boundary = "detektor-ui-boundary"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    for field_name, filename, content, content_type in files:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _post_json_multipart(
    url: str,
    files: Sequence[Tuple[str, str, bytes, str]],
    params: Dict[str, Any],
    timeout: int,
) -> Dict[str, Any]:
    query = urlencode({k: v for k, v in params.items() if v is not None})
    request_url = f"{url}?{query}" if query else url
    body, content_type = _encode_multipart_formdata({}, files)
    request = Request(
        request_url,
        data=body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc

    return json.loads(payload)


def _get_font() -> ImageFont.ImageFont:
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            _FONT = ImageFont.load_default()
    return _FONT


def _mask_to_array(mask_b64: str, target_size: Tuple[int, int]) -> np.ndarray:
    mask_bytes = base64.b64decode(mask_b64)
    mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")
    if mask_image.size != (target_size[1], target_size[0]):
        mask_image = mask_image.resize((target_size[1], target_size[0]), Image.NEAREST)
    return np.array(mask_image, dtype=np.uint8)


def _annotate_image(
    image: Image.Image,
    detections: Sequence[Dict[str, Any]],
    include_masks: bool,
    class_map: Dict[str, str],
) -> Image.Image:
    annotated = image.convert("RGBA")
    draw = ImageDraw.Draw(annotated)
    font = _get_font()

    for idx, det in enumerate(detections):
        box = det.get("box") or det.get("boxes")
        if not box:
            continue
        if isinstance(box[0], (list, tuple)):
            box = box[0]
        x1, y1, x2, y2 = map(float, box)
        label_id = str(det.get("label", det.get("labels", 0)))
        label_name = class_map.get(label_id, f"class_{label_id}")
        score = float(det.get("score", det.get("scores", 0)))
        color = tuple(_COLOR_PALETTE[idx % len(_COLOR_PALETTE)].tolist())

        if include_masks and det.get("mask"):
            overlay_arr = np.zeros((annotated.size[1], annotated.size[0], 4), dtype=np.uint8)
            mask_arr = _mask_to_array(det["mask"], (annotated.size[1], annotated.size[0]))
            overlay_arr[..., :3] = color
            overlay_arr[..., 3] = (mask_arr > 0).astype(np.uint8) * 85
            annotated = Image.alpha_composite(annotated, Image.fromarray(overlay_arr, mode="RGBA"))
            draw = ImageDraw.Draw(annotated)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{label_name} {score:.2f}"
        text_box = draw.textbbox((x1, y1), label_text, font=font)
        text_height = text_box[3] - text_box[1]
        text_width = text_box[2] - text_box[0]
        y_text = max(y1 - text_height - 5, 0)
        draw.rectangle([x1, y_text, x1 + text_width + 6, y1], fill=color)
        draw.text((x1 + 3, y_text + 1), label_text, fill="white", font=font)

    return annotated.convert("RGB")


def _detections_from_response(resp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "detections" in resp_json:
        return resp_json["detections"]
    boxes = resp_json.get("boxes", [])
    scores = resp_json.get("scores", [])
    labels = resp_json.get("labels", [])
    masks = resp_json.get("masks", [])
    detections = []
    for idx, box in enumerate(boxes):
        detection = {
            "box": box,
            "score": scores[idx] if idx < len(scores) else None,
            "label": labels[idx] if idx < len(labels) else None,
        }
        if idx < len(masks):
            detection["mask"] = masks[idx]
        detections.append(detection)
    return detections


def _parse_class_map(class_map_text: str) -> Dict[str, str]:
    if not class_map_text.strip():
        return {}
    try:
        payload = json.loads(class_map_text)
    except json.JSONDecodeError:
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _send_single_request(
    image: Image.Image,
    backend_url: str,
    conf_thresh: float,
    iou_thresh: float,
    max_det: int,
    include_masks: bool,
) -> Tuple[Dict[str, Any], float]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = _post_json_multipart(
        url=f"{backend_url.rstrip('/')}/v1/predict",
        files=[("image", "upload.png", buffer.getvalue(), "image/png")],
        params={
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
            "max_det": max_det,
            "include_masks": include_masks,
        },
        timeout=60,
    )
    return response, float(response.get("inference_time_ms") or 0.0)


def _send_batch_request(
    images: Sequence[Image.Image],
    backend_url: str,
    conf_thresh: float,
    iou_thresh: float,
    max_det: int,
    include_masks: bool,
) -> Tuple[Dict[str, Any], float]:
    files = []
    for idx, image in enumerate(images):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        files.append(("images", f"image_{idx}.png", buffer.getvalue(), "image/png"))
    started = time.perf_counter()
    response = _post_json_multipart(
        url=f"{backend_url.rstrip('/')}/v1/predict_batch",
        files=files,
        params={
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
            "max_det": max_det,
            "include_masks": include_masks,
        },
        timeout=120,
    )
    return response, (time.perf_counter() - started) * 1000.0


def _load_images_from_files(file_inputs: Optional[List[Any]]) -> List[Tuple[str, Image.Image]]:
    images: List[Tuple[str, Image.Image]] = []
    for path in _normalize_file_inputs(file_inputs):
        images.append((path.name, _load_rgb_image(path)))
    return images


def run_single_inference(
    image: Optional[Image.Image],
    backend_url: str,
    conf_thresh: float,
    iou_thresh: float,
    max_det: int,
    include_masks: bool,
    class_map_text: str,
) -> Tuple[Optional[Image.Image], List[List[Any]], Dict[str, Any], str]:
    if image is None:
        return None, [], {}, "Upload an image to run inference."

    class_map = _parse_class_map(class_map_text)
    try:
        resp_json, backend_latency = _send_single_request(
            image,
            backend_url.strip() or DEFAULT_BACKEND_URL,
            conf_thresh,
            iou_thresh,
            max_det,
            include_masks,
        )
    except Exception as exc:  # noqa: BLE001
        return None, [], {"error": str(exc)}, f"Request failed: {exc}"

    detections = _detections_from_response(resp_json)
    annotated = _annotate_image(image, detections, include_masks, class_map)
    rows = [[index, class_map.get(str(det.get("label")), det.get("label")), round(float(det.get("score", 0.0)), 3)] for index, det in enumerate(detections)]
    return annotated, rows, resp_json, f"Model latency: {backend_latency:.2f} ms"


def run_batch_inference(
    image_files: Optional[List[Any]],
    backend_url: str,
    conf_thresh: float,
    iou_thresh: float,
    max_det: int,
    include_masks: bool,
    class_map_text: str,
) -> Tuple[List[Tuple[Image.Image, str]], Dict[str, Any], str]:
    images = _load_images_from_files(image_files)
    if not images:
        return [], {}, "Upload at least one image."

    class_map = _parse_class_map(class_map_text)
    names = [name for name, _ in images]
    pil_images = [image for _, image in images]
    try:
        resp_json, latency_ms = _send_batch_request(
            pil_images,
            backend_url.strip() or DEFAULT_BACKEND_URL,
            conf_thresh,
            iou_thresh,
            max_det,
            include_masks,
        )
    except Exception as exc:  # noqa: BLE001
        return [], {"error": str(exc)}, f"Batch request failed: {exc}"

    gallery_items: List[Tuple[Image.Image, str]] = []
    for name, image, prediction in zip(names, pil_images, resp_json.get("predictions", [])):
        detections = _detections_from_response(prediction)
        gallery_items.append((_annotate_image(image, detections, include_masks, class_map), _gallery_caption(name, detections, class_map)))

    return gallery_items, resp_json, f"Batch latency: {latency_ms:.2f} ms"


def _dashboard_outputs_from_state(state: Dict[str, Any]) -> Tuple[Any, ...]:
    available = list(state.get("available_checkpoints", {}).keys())
    active_key = state.get("active_checkpoint_key") or (available[0] if available else None)
    dataset = state.get("dataset", {})
    training_summary = state.get("training_summary", {})
    validation_history = state.get("validation_history", [])
    class_map = state.get("class_map", {})
    plots = state.get("plots", {})
    overview_html = _build_overview_html(state)
    dataset_json = dataset
    training_json = {
        "training": training_summary,
        "validation": state.get("validation_summary", {}),
        "checkpoint": state.get("checkpoint_summary", {}),
        "runtime": state.get("runtime", {}),
    }
    validation_rows = [
        [
            row.get("epoch"),
            _fmt_metric(row.get("val_precision")),
            _fmt_metric(row.get("val_recall")),
            _fmt_metric(row.get("val_map50")),
            _fmt_metric(row.get("val_mean_iou")),
        ]
        for row in validation_history
    ]
    status = (
        f"Active checkpoint: {active_key or 'unavailable'} | "
        f"Run: {state.get('run_dir', 'n/a')} | Device: {state.get('device', 'n/a')}"
    )
    return (
        gr.update(choices=available, value=active_key),
        overview_html,
        status,
        dataset_json,
        training_json,
        json.dumps(class_map, indent=2),
        validation_rows,
        _plot_training_curves(state),
        _plot_validation_curves(state),
        plots.get("loss_total"),
        plots.get("loss_components"),
        plots.get("learning_rate"),
    )


def _build_overview_html(state: Dict[str, Any]) -> str:
    runtime = state.get("runtime", {})
    checkpoint = state.get("checkpoint_summary", {})
    dataset = state.get("dataset", {})
    cards = [
        ("Checkpoint", str(state.get("active_checkpoint_key", "n/a")).upper()),
        ("Model", runtime.get("model_display_name") or checkpoint.get("model_config", {}).get("display_name") or "n/a"),
        ("Dataset Size", dataset.get("dataset_size") or "n/a"),
        ("Classes", dataset.get("num_classes") or "n/a"),
        ("Batch", runtime.get("batch_size") or "n/a"),
        ("Image Size", runtime.get("img_size") or "n/a"),
        ("Best Metric", _fmt_metric(checkpoint.get("best_metric"))),
        ("Epoch", checkpoint.get("epoch") or "n/a"),
    ]
    card_html = "".join(
        f"<div class='det-card'><div class='det-card-label'>{label}</div><div class='det-card-value'>{value}</div></div>"
        for label, value in cards
    )
    return (
        "<div class='det-shell det-hero'>"
        "<h1>Detektor Serving Console</h1>"
        "<p>Checkpoint-aware inference UI backed by the active training run.</p>"
        f"<div class='det-card-grid'>{card_html}</div>"
        "</div>"
    )


def _plot_training_curves(state: Dict[str, Any]):
    rows = state.get("train_curve", [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    if rows:
        steps = [row["step"] for row in rows if row.get("step") is not None]
        losses = [row["loss_total"] for row in rows if row.get("loss_total") is not None]
        lrs = [row["lr"] for row in rows if row.get("lr") is not None]
        if steps and losses:
            axes[0].plot(steps[: len(losses)], losses, color="#c25b2d", linewidth=2)
        if steps and lrs:
            axes[1].plot(steps[: len(lrs)], lrs, color="#2b7a78", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[1].set_title("Learning Rate")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("LR")
    fig.tight_layout()
    return fig


def _plot_validation_curves(state: Dict[str, Any]):
    rows = state.get("validation_history", [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    if rows:
        epochs = [row["epoch"] for row in rows if row.get("epoch") is not None]
        map50 = [row.get("val_map50") for row in rows]
        recall = [row.get("val_recall") for row in rows]
        mean_iou = [row.get("val_mean_iou") for row in rows]
        if epochs:
            axes[0].plot(epochs, map50, color="#c25b2d", linewidth=2, label="mAP50")
            axes[0].plot(epochs, recall, color="#2b7a78", linewidth=2, label="Recall")
            axes[1].plot(epochs, mean_iou, color="#1e88e5", linewidth=2, label="Mean IoU")
    axes[0].set_title("Validation Metrics")
    axes[1].set_title("Validation IoU")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlabel("Epoch")
        axis.legend(loc="best")
    fig.tight_layout()
    return fig


def _collect_image_paths(image_files: Optional[List[Any]], folder_path: str) -> List[Path]:
    seen: set[str] = set()
    results: List[Path] = []
    for path in _normalize_file_inputs(image_files):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            results.append(path)
    folder = folder_path.strip()
    if folder:
        folder_root = Path(folder).expanduser()
        if folder_root.is_dir():
            for path in sorted(folder_root.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    key = str(path.resolve())
                    if key not in seen:
                        seen.add(key)
                        results.append(path)
    return results


def _normalize_file_inputs(file_inputs: Optional[List[Any]]) -> List[Path]:
    paths: List[Path] = []
    for file_obj in file_inputs or []:
        path: Optional[str] = None
        if isinstance(file_obj, str):
            path = file_obj
        elif isinstance(file_obj, dict) and "name" in file_obj:
            path = file_obj["name"]
        elif hasattr(file_obj, "name"):
            path = getattr(file_obj, "name")
        if path:
            candidate = Path(path)
            if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(candidate)
    return paths


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _gallery_caption(filename: str, detections: Sequence[Dict[str, Any]], class_map: Dict[str, str]) -> str:
    summary = _summarize_detections(detections, class_map)
    return f"{filename} | {summary}"


def _summarize_detections(detections: Sequence[Dict[str, Any]], class_map: Dict[str, str]) -> str:
    if not detections:
        return "No detections"
    parts = []
    for det in detections[:6]:
        label_id = str(det.get("label"))
        label_name = class_map.get(label_id, f"class_{label_id}")
        parts.append(f"{label_name} {float(det.get('score', 0.0)):.2f}")
    if len(detections) > 6:
        parts.append(f"+{len(detections) - 6} more")
    return ", ".join(parts)


def _chunked(items: Sequence[Path], chunk_size: int) -> List[List[Path]]:
    return [list(items[index : index + chunk_size]) for index in range(0, len(items), chunk_size)]


def _fmt_metric(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.4f}"


def build_interface(runtime: Optional[DetektorUIRuntime] = None) -> gr.Blocks:
    if runtime is None:
        return _build_remote_interface()

    with gr.Blocks(title="Detektor UI", css=_UI_CSS) as demo:
        checkpoint_selector = gr.Dropdown(label="Model checkpoint", choices=[], value=None)
        refresh_button = gr.Button("Refresh", variant="secondary")
        overview_html = gr.HTML()
        status_box = gr.Textbox(label="Runtime status", interactive=False)

        with gr.Row():
            with gr.Column(scale=5):
                upload_files = gr.Files(
                    label="Upload or drag image files",
                    type="filepath",
                    file_types=sorted(IMAGE_EXTENSIONS),
                )
                folder_input = gr.Textbox(
                    label="Folder path",
                    placeholder=r"F:\data\images",
                )
                with gr.Row():
                    conf_slider = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="Confidence")
                    iou_slider = gr.Slider(0.1, 0.9, value=0.6, step=0.01, label="IoU")
                with gr.Row():
                    max_det_slider = gr.Slider(1, 300, value=100, step=1, label="Max detections")
                    include_masks = gr.Checkbox(value=False, label="Render masks")
                run_button = gr.Button("Run Inference", variant="primary")
                latency_box = gr.Textbox(label="Inference summary", interactive=False)
            with gr.Column(scale=4):
                class_map_box = gr.Code(label="Class map", language="json", interactive=False)
                dataset_json = gr.JSON(label="Dataset details")
                training_json = gr.JSON(label="Training + checkpoint details")

        gallery_output = gr.Gallery(label="Annotated predictions", height=520, preview=True, object_fit="contain")
        batch_table = gr.Dataframe(
            headers=["Image", "Detections", "Summary"],
            datatype=["str", "number", "str"],
            interactive=False,
            label="Per-image summary",
        )
        prediction_json = gr.JSON(label="Prediction payload")

        with gr.Row():
            train_plot = gr.Plot(label="Training curves")
            val_plot = gr.Plot(label="Validation curves")

        validation_table = gr.Dataframe(
            headers=["Epoch", "Precision", "Recall", "mAP50", "Mean IoU"],
            datatype=["number", "str", "str", "str", "str"],
            interactive=False,
            label="Validation history",
        )

        with gr.Row():
            loss_total_img = gr.Image(label="Saved loss plot", type="filepath")
            loss_components_img = gr.Image(label="Saved loss components", type="filepath")
            learning_rate_img = gr.Image(label="Saved LR plot", type="filepath")

        dashboard_outputs = [
            checkpoint_selector,
            overview_html,
            status_box,
            dataset_json,
            training_json,
            class_map_box,
            validation_table,
            train_plot,
            val_plot,
            loss_total_img,
            loss_components_img,
            learning_rate_img,
        ]

        demo.load(runtime.refresh_dashboard, outputs=dashboard_outputs)
        refresh_button.click(runtime.refresh_dashboard, outputs=dashboard_outputs)
        checkpoint_selector.change(runtime.set_checkpoint, inputs=[checkpoint_selector], outputs=dashboard_outputs)
        run_button.click(
            runtime.run_gallery_inference,
            inputs=[upload_files, folder_input, conf_slider, iou_slider, max_det_slider, include_masks],
            outputs=[gallery_output, batch_table, prediction_json, latency_box],
        )

    demo.queue(default_concurrency_limit=1)
    return demo


def _build_remote_interface() -> gr.Blocks:
    with gr.Blocks(title="Detektor Internal UI", css=_UI_CSS) as demo:
        gr.Markdown(
            """
            # Detektor Remote UI
            Use this mode when the FastAPI inference server is already running elsewhere.
            """
        )
        backend_input = gr.Textbox(value=DEFAULT_BACKEND_URL, label="Backend URL")
        class_map_input = gr.Textbox(label="Class map JSON", placeholder='{"0":"player"}')

        with gr.Tab("Single Image"):
            single_image = gr.Image(type="pil", label="Upload image")
            conf_slider = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="Confidence")
            iou_slider = gr.Slider(0.1, 0.9, value=0.6, step=0.01, label="IoU")
            max_det_slider = gr.Slider(1, 300, value=100, step=1, label="Max detections")
            include_masks_chk = gr.Checkbox(value=False, label="Render masks")
            run_button = gr.Button("Run Inference", variant="primary")
            annotated_output = gr.Image(label="Annotated image", type="pil")
            table_output = gr.Dataframe(headers=["#", "Class", "Score"], interactive=False, label="Detections")
            json_output = gr.JSON(label="Raw JSON response")
            latency_output = gr.Textbox(label="Latency", interactive=False)
            run_button.click(
                run_single_inference,
                inputs=[single_image, backend_input, conf_slider, iou_slider, max_det_slider, include_masks_chk, class_map_input],
                outputs=[annotated_output, table_output, json_output, latency_output],
            )

        with gr.Tab("Batch"):
            batch_images = gr.Files(label="Upload multiple images", type="filepath")
            batch_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="Confidence")
            batch_iou = gr.Slider(0.1, 0.9, value=0.6, step=0.01, label="IoU")
            batch_max_det = gr.Slider(1, 300, value=100, step=1, label="Max detections per image")
            batch_masks = gr.Checkbox(value=False, label="Render masks")
            batch_button = gr.Button("Run Batch Inference")
            gallery_output = gr.Gallery(label="Annotated results", height=520)
            batch_json_output = gr.JSON(label="Raw batch JSON response")
            batch_latency_output = gr.Textbox(label="Batch latency", interactive=False)
            batch_button.click(
                run_batch_inference,
                inputs=[batch_images, backend_input, batch_conf, batch_iou, batch_max_det, batch_masks, class_map_input],
                outputs=[gallery_output, batch_json_output, batch_latency_output],
            )

    demo.queue(default_concurrency_limit=1)
    return demo


def main() -> None:
    demo = build_interface()
    demo.launch(
        server_name=os.getenv("DETEKTOR_UI_HOST", "0.0.0.0"),
        server_port=int(os.getenv("DETEKTOR_UI_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
