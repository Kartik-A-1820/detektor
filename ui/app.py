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
.det-gate-open {
  color: #2e7d32;
  font-weight: 700;
  font-size: 1.1rem;
}
.det-gate-blocked {
  color: #c62828;
  font-weight: 700;
  font-size: 1.1rem;
}
"""

_MODEL_PROFILES = ["firefly", "comet", "nova", "pulsar", "quasar", "supernova"]


class DetektorUIRuntime:
    """Bridge from Gradio callbacks to the active in-process inference runtime."""

    def __init__(
        self,
        *,
        get_runtime_state: Callable[[], Dict[str, Any]],
        get_service_snapshot: Callable[[], Tuple[Any, Any]],
        select_checkpoint: Callable[[str], Dict[str, Any]],
        backend_port: int = 8000,
    ) -> None:
        self._get_runtime_state = get_runtime_state
        self._get_service_snapshot = get_service_snapshot
        self._select_checkpoint = select_checkpoint
        self._backend_port = backend_port

    @property
    def _local_base_url(self) -> str:
        return f"http://127.0.0.1:{self._backend_port}"

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

    # ------------------------------------------------------------------ #
    #  Training / Validation / Dataset-check bridge methods               #
    # ------------------------------------------------------------------ #

    def start_training(
        self,
        data_yaml: str,
        config_path: str,
        epochs: int,
        batch_size: int,
        lr: float,
        model_profile: str,
        focal_loss_gamma: float,
        out_dir: str,
        run_val: bool,
    ) -> Dict[str, Any]:
        payload = {
            "data_yaml": data_yaml,
            "config_path": config_path or None,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "model_profile": model_profile or None,
            "focal_loss_gamma": float(focal_loss_gamma),
            "out_dir": out_dir or None,
            "run_val": bool(run_val),
        }
        return _post_json(f"{self._local_base_url}/v1/train/start", payload)

    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        return _get_json(f"{self._local_base_url}/v1/train/status/{job_id}")

    def stop_training(self, job_id: str) -> Dict[str, Any]:
        return _post_json(f"{self._local_base_url}/v1/train/stop/{job_id}", {})

    def run_validation(
        self,
        weights: str,
        data_yaml: str,
        conf_thresh: float,
        iou_thresh: float,
        output_dir: str,
    ) -> Dict[str, Any]:
        payload = {
            "weights": weights or None,
            "data_yaml": data_yaml or None,
            "conf_thresh": float(conf_thresh),
            "iou_thresh": float(iou_thresh),
            "output_dir": output_dir or None,
        }
        return _post_json(f"{self._local_base_url}/v1/validate/run", payload)

    def check_dataset(self, data_yaml: str, output_dir: str) -> Dict[str, Any]:
        payload = {
            "data_yaml": data_yaml,
            "output_dir": output_dir or "reports",
        }
        return _post_json(f"{self._local_base_url}/v1/dataset/check", payload)


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


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    """POST a JSON body and return the parsed JSON response."""
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc


def _get_json(url: str, timeout: int = 30) -> Dict[str, Any]:
    """GET a URL and return the parsed JSON response."""
    req = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc


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


# ------------------------------------------------------------------ #
#  Training tab callbacks                                              #
# ------------------------------------------------------------------ #

def _start_training_remote(
    backend_url: str,
    data_yaml: str,
    config_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    model_profile: str,
    focal_loss_gamma: float,
    out_dir: str,
    run_val: bool,
) -> Tuple[str, str]:
    """Start training via remote backend. Returns (status_text, job_id)."""
    if not data_yaml.strip():
        return "Error: data_yaml is required.", ""
    payload = {
        "data_yaml": data_yaml.strip(),
        "config_path": config_path.strip() or None,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "model_profile": model_profile or None,
        "focal_loss_gamma": float(focal_loss_gamma),
        "out_dir": out_dir.strip() or None,
        "run_val": bool(run_val),
    }
    try:
        resp = _post_json(f"{backend_url.rstrip('/')}/v1/train/start", payload)
        job_id = resp.get("job_id", "")
        return f"Training started. Job ID: {job_id}", job_id
    except Exception as exc:  # noqa: BLE001
        return f"Error starting training: {exc}", ""


def _refresh_training_status_remote(
    backend_url: str,
    job_id: str,
) -> Tuple[str, str, Dict[str, Any], Any]:
    """Refresh training status. Returns (status_text, log_text, metrics_json, loss_plot)."""
    if not job_id:
        return "No active job.", "", {}, None
    try:
        resp = _get_json(f"{backend_url.rstrip('/')}/v1/train/status/{job_id}")
        status = resp.get("status", "unknown")
        log_tail = resp.get("log_tail", [])
        metrics = resp.get("metrics", {})
        log_text = "\n".join(log_tail[-30:])
        status_text = f"Job {job_id[:8]}… | Status: {status}"
        if resp.get("error"):
            status_text += f" | Error: {resp['error']}"
        loss_plot = _plot_live_loss(metrics)
        return status_text, log_text, metrics, loss_plot
    except Exception as exc:  # noqa: BLE001
        return f"Error fetching status: {exc}", "", {}, None


def _stop_training_remote(backend_url: str, job_id: str) -> str:
    if not job_id:
        return "No active job to stop."
    try:
        resp = _post_json(f"{backend_url.rstrip('/')}/v1/train/stop/{job_id}", {})
        return f"Stop requested: {resp.get('message', resp.get('status', 'ok'))}"
    except Exception as exc:  # noqa: BLE001
        return f"Error stopping training: {exc}"


def _plot_live_loss(metrics: Dict[str, Any]) -> Optional[Any]:
    """Build a simple loss plot from training metrics dict."""
    train_curve = metrics.get("train_curve") or []
    if not train_curve:
        return None
    try:
        steps = [row.get("step") for row in train_curve if row.get("step") is not None]
        losses = [row.get("loss_total") for row in train_curve if row.get("loss_total") is not None]
        if not steps or not losses:
            return None
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(steps[: len(losses)], losses, color="#c25b2d", linewidth=2)
        ax.set_title("Live Training Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ #
#  Validation tab callbacks                                            #
# ------------------------------------------------------------------ #

def _run_validation_remote(
    backend_url: str,
    weights: str,
    data_yaml: str,
    conf_thresh: float,
    iou_thresh: float,
    output_dir: str,
) -> Tuple[Dict[str, Any], List[List[Any]], str, str]:
    """Run validation via remote backend. Returns (metrics_json, per_class_rows, status_text, gate_html)."""
    payload = {
        "weights": weights.strip() or None,
        "data_yaml": data_yaml.strip() or None,
        "conf_thresh": float(conf_thresh),
        "iou_thresh": float(iou_thresh),
        "output_dir": output_dir.strip() or None,
    }
    try:
        resp = _post_json(f"{backend_url.rstrip('/')}/v1/validate/run", payload, timeout=300)
        metrics = resp.get("metrics", {})
        per_class = _extract_per_class_rows(metrics)
        gate_html = _build_gate_html(metrics)
        return metrics, per_class, "Validation complete.", gate_html
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}, [], f"Validation failed: {exc}", "<span class='det-gate-blocked'>GATE: BLOCKED (error)</span>"


def _extract_per_class_rows(metrics: Dict[str, Any]) -> List[List[Any]]:
    """Extract per-class metrics rows from a validation metrics dict."""
    per_class = metrics.get("per_class") or metrics.get("per_class_metrics") or []
    rows = []
    if isinstance(per_class, list):
        for entry in per_class:
            if isinstance(entry, dict):
                rows.append([
                    entry.get("class_name", entry.get("class", "")),
                    _fmt_metric(entry.get("precision")),
                    _fmt_metric(entry.get("recall")),
                    _fmt_metric(entry.get("f1")),
                    _fmt_metric(entry.get("ap50")),
                ])
    elif isinstance(per_class, dict):
        for class_name, class_metrics in per_class.items():
            if isinstance(class_metrics, dict):
                rows.append([
                    class_name,
                    _fmt_metric(class_metrics.get("precision")),
                    _fmt_metric(class_metrics.get("recall")),
                    _fmt_metric(class_metrics.get("f1")),
                    _fmt_metric(class_metrics.get("ap50")),
                ])
    return rows


def _build_gate_html(metrics: Dict[str, Any]) -> str:
    """Build a promotion gate HTML snippet based on validation metrics."""
    if not metrics or "error" in metrics:
        return "<span class='det-gate-blocked'>GATE: BLOCKED (no metrics)</span>"

    map50 = metrics.get("map50") or metrics.get("val_map50") or metrics.get("mAP50")
    recall = metrics.get("recall") or metrics.get("val_recall")

    try:
        map50_val = float(map50) if map50 is not None else None
        recall_val = float(recall) if recall is not None else None
    except (TypeError, ValueError):
        map50_val = None
        recall_val = None

    # Gate criteria: mAP50 >= 0.5 and recall >= 0.5
    gate_open = (
        map50_val is not None and map50_val >= 0.5
        and (recall_val is None or recall_val >= 0.5)
    )

    map50_str = f"{map50_val:.4f}" if map50_val is not None else "n/a"
    recall_str = f"{recall_val:.4f}" if recall_val is not None else "n/a"

    if gate_open:
        return (
            f"<span class='det-gate-open'>✅ GATE: OPEN</span> "
            f"<span style='color:var(--det-muted);font-size:0.9rem'>"
            f"mAP50={map50_str} recall={recall_str}</span>"
        )
    return (
        f"<span class='det-gate-blocked'>🚫 GATE: BLOCKED</span> "
        f"<span style='color:var(--det-muted);font-size:0.9rem'>"
        f"mAP50={map50_str} recall={recall_str} (need ≥0.5)</span>"
    )


# ------------------------------------------------------------------ #
#  Dataset check tab callbacks                                         #
# ------------------------------------------------------------------ #

def _run_dataset_check_remote(
    backend_url: str,
    data_yaml: str,
    output_dir: str,
) -> Tuple[Dict[str, Any], List[List[Any]], str]:
    """Run dataset check via remote backend. Returns (summary_json, issues_rows, status_text)."""
    if not data_yaml.strip():
        return {}, [], "Error: data_yaml is required."
    payload = {
        "data_yaml": data_yaml.strip(),
        "output_dir": output_dir.strip() or "reports",
    }
    try:
        resp = _post_json(f"{backend_url.rstrip('/')}/v1/dataset/check", payload, timeout=300)
        summary = resp.get("summary", {})
        issues = resp.get("issues", [])
        rows = [
            [
                issue.get("severity", ""),
                issue.get("category", ""),
                issue.get("message", ""),
                issue.get("file", ""),
            ]
            for issue in issues
        ]
        has_errors = summary.get("has_errors", False)
        has_warnings = summary.get("has_warnings", False)
        if has_errors:
            status = f"Dataset check FAILED — {summary.get('num_issues', 0)} issue(s) found."
        elif has_warnings:
            status = f"Dataset check PASSED with warnings — {summary.get('num_issues', 0)} issue(s)."
        else:
            status = "Dataset check PASSED — no issues found."
        return summary, rows, status
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}, [], f"Dataset check failed: {exc}"


def build_interface(runtime: Optional[DetektorUIRuntime] = None) -> gr.Blocks:
    if runtime is None:
        return _build_remote_interface()

    with gr.Blocks(title="Detektor UI", css=_UI_CSS) as demo:
        with gr.Tabs():
            # ---------------------------------------------------------- #
            #  Tab 1: Inference                                            #
            # ---------------------------------------------------------- #
            with gr.Tab("Inference"):
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

            # ---------------------------------------------------------- #
            #  Tab 2: Training                                             #
            # ---------------------------------------------------------- #
            with gr.Tab("Training"):
                _build_training_tab_inprocess(runtime)

            # ---------------------------------------------------------- #
            #  Tab 3: Validation                                           #
            # ---------------------------------------------------------- #
            with gr.Tab("Validation"):
                _build_validation_tab_inprocess(runtime)

            # ---------------------------------------------------------- #
            #  Tab 4: Dataset Check                                        #
            # ---------------------------------------------------------- #
            with gr.Tab("Dataset Check"):
                _build_dataset_check_tab_inprocess(runtime)

    demo.queue(default_concurrency_limit=1)
    return demo


def _build_training_tab_inprocess(runtime: DetektorUIRuntime) -> None:
    """Build the Training tab wired to the in-process runtime."""
    job_id_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=3):
            tr_data_yaml = gr.Textbox(label="data_yaml path", placeholder="path/to/data.yaml")
            tr_config_path = gr.Textbox(label="config_yaml path (optional)", placeholder="path/to/config.yaml")
            with gr.Row():
                tr_epochs = gr.Slider(1, 100, value=10, step=1, label="Epochs")
                tr_batch_size = gr.Slider(1, 32, value=4, step=1, label="Batch size")
            with gr.Row():
                tr_lr = gr.Number(value=0.002, label="Learning rate")
                tr_model_profile = gr.Dropdown(choices=_MODEL_PROFILES, value="nova", label="Model profile")
            tr_focal_gamma = gr.Slider(0.0, 3.0, value=0.0, step=0.1, label="Focal loss gamma")
            tr_out_dir = gr.Textbox(label="Output directory", placeholder="runs/train")
            tr_run_val = gr.Checkbox(value=False, label="Run validation after training")
            with gr.Row():
                tr_start_btn = gr.Button("Start Training", variant="primary")
                tr_stop_btn = gr.Button("Stop Training", variant="stop")
                tr_refresh_btn = gr.Button("Refresh Status", variant="secondary")
        with gr.Column(scale=4):
            tr_status_box = gr.Textbox(label="Status", interactive=False)
            tr_log_box = gr.Textbox(label="Training log (last 30 lines)", lines=15, interactive=False)
            tr_metrics_json = gr.JSON(label="Metrics")
            tr_loss_plot = gr.Plot(label="Live loss plot")

    def _start(data_yaml, config_path, epochs, batch_size, lr, model_profile, focal_gamma, out_dir, run_val):
        try:
            resp = runtime.start_training(
                data_yaml=data_yaml,
                config_path=config_path,
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                model_profile=model_profile,
                focal_loss_gamma=float(focal_gamma),
                out_dir=out_dir,
                run_val=bool(run_val),
            )
            job_id = resp.get("job_id", "")
            return f"Training started. Job ID: {job_id}", job_id
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}", ""

    def _refresh(job_id):
        if not job_id:
            return "No active job.", "", {}, None
        try:
            resp = runtime.get_training_status(job_id)
            status = resp.get("status", "unknown")
            log_tail = resp.get("log_tail", [])
            metrics = resp.get("metrics", {})
            log_text = "\n".join(log_tail[-30:])
            status_text = f"Job {job_id[:8]}… | Status: {status}"
            if resp.get("error"):
                status_text += f" | Error: {resp['error']}"
            return status_text, log_text, metrics, _plot_live_loss(metrics)
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}", "", {}, None

    def _stop(job_id):
        if not job_id:
            return "No active job to stop."
        try:
            resp = runtime.stop_training(job_id)
            return f"Stop requested: {resp.get('message', resp.get('status', 'ok'))}"
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}"

    tr_start_btn.click(
        _start,
        inputs=[tr_data_yaml, tr_config_path, tr_epochs, tr_batch_size, tr_lr, tr_model_profile, tr_focal_gamma, tr_out_dir, tr_run_val],
        outputs=[tr_status_box, job_id_state],
    )
    tr_refresh_btn.click(
        _refresh,
        inputs=[job_id_state],
        outputs=[tr_status_box, tr_log_box, tr_metrics_json, tr_loss_plot],
    )
    tr_stop_btn.click(
        _stop,
        inputs=[job_id_state],
        outputs=[tr_status_box],
    )


def _build_validation_tab_inprocess(runtime: DetektorUIRuntime) -> None:
    """Build the Validation tab wired to the in-process runtime."""
    with gr.Row():
        with gr.Column(scale=3):
            val_weights = gr.Textbox(label="Weights path (leave blank for active checkpoint)", placeholder="path/to/model.pt")
            val_data_yaml = gr.Textbox(label="data_yaml path", placeholder="path/to/data.yaml")
            with gr.Row():
                val_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="Confidence threshold")
                val_iou = gr.Slider(0.1, 0.9, value=0.5, step=0.01, label="IoU threshold")
            val_out_dir = gr.Textbox(label="Output directory", placeholder="runs/val")
            val_run_btn = gr.Button("Run Validation", variant="primary")
            val_status_box = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=4):
            val_gate_html = gr.HTML(label="Promotion gate")
            val_metrics_json = gr.JSON(label="Metrics")
            val_per_class_table = gr.Dataframe(
                headers=["Class", "Precision", "Recall", "F1", "AP50"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
                label="Per-class metrics",
            )

    def _run_val(weights, data_yaml, conf, iou, out_dir):
        try:
            resp = runtime.run_validation(
                weights=weights,
                data_yaml=data_yaml,
                conf_thresh=float(conf),
                iou_thresh=float(iou),
                output_dir=out_dir,
            )
            metrics = resp.get("metrics", resp)
            per_class = _extract_per_class_rows(metrics)
            gate_html = _build_gate_html(metrics)
            return metrics, per_class, "Validation complete.", gate_html
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}, [], f"Validation failed: {exc}", "<span class='det-gate-blocked'>GATE: BLOCKED (error)</span>"

    val_run_btn.click(
        _run_val,
        inputs=[val_weights, val_data_yaml, val_conf, val_iou, val_out_dir],
        outputs=[val_metrics_json, val_per_class_table, val_status_box, val_gate_html],
    )


def _build_dataset_check_tab_inprocess(runtime: DetektorUIRuntime) -> None:
    """Build the Dataset Check tab wired to the in-process runtime."""
    with gr.Row():
        with gr.Column(scale=3):
            dc_data_yaml = gr.Textbox(label="data_yaml path", placeholder="path/to/data.yaml")
            dc_out_dir = gr.Textbox(label="Output directory", placeholder="reports", value="reports")
            dc_run_btn = gr.Button("Run Check", variant="primary")
            dc_status_box = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=4):
            dc_summary_json = gr.JSON(label="Summary")
            dc_issues_table = gr.Dataframe(
                headers=["Severity", "Category", "Message", "File"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                label="Issues",
            )

    def _run_check(data_yaml, out_dir):
        try:
            resp = runtime.check_dataset(data_yaml=data_yaml, output_dir=out_dir)
            summary = resp.get("summary", {})
            issues = resp.get("issues", [])
            rows = [
                [i.get("severity", ""), i.get("category", ""), i.get("message", ""), i.get("file", "")]
                for i in issues
            ]
            has_errors = summary.get("has_errors", False)
            has_warnings = summary.get("has_warnings", False)
            if has_errors:
                status = f"Dataset check FAILED — {summary.get('num_issues', 0)} issue(s) found."
            elif has_warnings:
                status = f"Dataset check PASSED with warnings — {summary.get('num_issues', 0)} issue(s)."
            else:
                status = "Dataset check PASSED — no issues found."
            return summary, rows, status
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}, [], f"Dataset check failed: {exc}"

    dc_run_btn.click(
        _run_check,
        inputs=[dc_data_yaml, dc_out_dir],
        outputs=[dc_summary_json, dc_issues_table, dc_status_box],
    )


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

        with gr.Tabs():
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

            # ---------------------------------------------------------- #
            #  Training tab (remote)                                       #
            # ---------------------------------------------------------- #
            with gr.Tab("Training"):
                job_id_state = gr.State("")

                with gr.Row():
                    with gr.Column(scale=3):
                        tr_data_yaml = gr.Textbox(label="data_yaml path", placeholder="path/to/data.yaml")
                        tr_config_path = gr.Textbox(label="config_yaml path (optional)", placeholder="path/to/config.yaml")
                        with gr.Row():
                            tr_epochs = gr.Slider(1, 100, value=10, step=1, label="Epochs")
                            tr_batch_size = gr.Slider(1, 32, value=4, step=1, label="Batch size")
                        with gr.Row():
                            tr_lr = gr.Number(value=0.002, label="Learning rate")
                            tr_model_profile = gr.Dropdown(choices=_MODEL_PROFILES, value="nova", label="Model profile")
                        tr_focal_gamma = gr.Slider(0.0, 3.0, value=0.0, step=0.1, label="Focal loss gamma")
                        tr_out_dir = gr.Textbox(label="Output directory", placeholder="runs/train")
                        tr_run_val = gr.Checkbox(value=False, label="Run validation after training")
                        with gr.Row():
                            tr_start_btn = gr.Button("Start Training", variant="primary")
                            tr_stop_btn = gr.Button("Stop Training", variant="stop")
                            tr_refresh_btn = gr.Button("Refresh Status", variant="secondary")
                    with gr.Column(scale=4):
                        tr_status_box = gr.Textbox(label="Status", interactive=False)
                        tr_log_box = gr.Textbox(label="Training log (last 30 lines)", lines=15, interactive=False)
                        tr_metrics_json = gr.JSON(label="Metrics")
                        tr_loss_plot = gr.Plot(label="Live loss plot")

                tr_start_btn.click(
                    _start_training_remote,
                    inputs=[backend_input, tr_data_yaml, tr_config_path, tr_epochs, tr_batch_size, tr_lr, tr_model_profile, tr_focal_gamma, tr_out_dir, tr_run_val],
                    outputs=[tr_status_box, job_id_state],
                )
                tr_refresh_btn.click(
                    _refresh_training_status_remote,
                    inputs=[backend_input, job_id_state],
                    outputs=[tr_status_box, tr_log_box, tr_metrics_json, tr_loss_plot],
                )
                tr_stop_btn.click(
                    _stop_training_remote,
                    inputs=[backend_input, job_id_state],
                    outputs=[tr_status_box],
                )

            # ---------------------------------------------------------- #
            #  Validation tab (remote)                                     #
            # ---------------------------------------------------------- #
            with gr.Tab("Validation"):
                with gr.Row():
                    with gr.Column(scale=3):
                        val_weights = gr.Textbox(label="Weights path (leave blank for active checkpoint)", placeholder="path/to/model.pt")
                        val_data_yaml = gr.Textbox(label="data_yaml path", placeholder="path/to/data.yaml")
                        with gr.Row():
                            val_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="Confidence threshold")
                            val_iou = gr.Slider(0.1, 0.9, value=0.5, step=0.01, label="IoU threshold")
                        val_out_dir = gr.Textbox(label="Output directory", placeholder="runs/val")
                        val_run_btn = gr.Button("Run Validation", variant="primary")
                        val_status_box = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=4):
                        val_gate_html = gr.HTML(label="Promotion gate")
                        val_metrics_json = gr.JSON(label="Metrics")
                        val_per_class_table = gr.Dataframe(
                            headers=["Class", "Precision", "Recall", "F1", "AP50"],
                            datatype=["str", "str", "str", "str", "str"],
                            interactive=False,
                            label="Per-class metrics",
                        )

                val_run_btn.click(
                    _run_validation_remote,
                    inputs=[backend_input, val_weights, val_data_yaml, val_conf, val_iou, val_out_dir],
                    outputs=[val_metrics_json, val_per_class_table, val_status_box, val_gate_html],
                )

            # ---------------------------------------------------------- #
            #  Dataset Check tab (remote)                                  #
            # ---------------------------------------------------------- #
            with gr.Tab("Dataset Check"):
                with gr.Row():
                    with gr.Column(scale=3):
                        dc_data_yaml = gr.Textbox(label="data_yaml path", placeholder="path/to/data.yaml")
                        dc_out_dir = gr.Textbox(label="Output directory", placeholder="reports", value="reports")
                        dc_run_btn = gr.Button("Run Check", variant="primary")
                        dc_status_box = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=4):
                        dc_summary_json = gr.JSON(label="Summary")
                        dc_issues_table = gr.Dataframe(
                            headers=["Severity", "Category", "Message", "File"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            label="Issues",
                        )

                dc_run_btn.click(
                    _run_dataset_check_remote,
                    inputs=[backend_input, dc_data_yaml, dc_out_dir],
                    outputs=[dc_summary_json, dc_issues_table, dc_status_box],
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
