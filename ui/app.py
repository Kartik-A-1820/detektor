"""Lightweight Gradio UI for Detektor FastAPI backend."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import gradio as gr

DEFAULT_BACKEND_URL = os.getenv("DETEKTOR_UI_BACKEND", "http://localhost:8000")
_COLOR_PALETTE = np.array(
    [
        (231, 76, 60),
        (46, 204, 113),
        (52, 152, 219),
        (155, 89, 182),
        (241, 196, 15),
        (230, 126, 34),
        (26, 188, 156),
        (149, 165, 166),
    ],
    dtype=np.uint8,
)
_FONT = None


def _get_font() -> ImageFont.ImageFont:
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            _FONT = ImageFont.load_default()
    return _FONT


def _parse_class_map(class_map_text: str) -> Dict[str, str]:
    if not class_map_text.strip():
        return {}
    try:
        data = json.loads(class_map_text)
        return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        return {}


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
            # Legacy list of boxes
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
            overlay_arr[..., 3] = (mask_arr > 0).astype(np.uint8) * 90
            annotated = Image.alpha_composite(annotated, Image.fromarray(overlay_arr, mode="RGBA"))
            draw = ImageDraw.Draw(annotated)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{label_name} {score:.2f}"
        text_size = draw.textbbox((x1, y1), label_text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        draw.rectangle([x1, max(y1 - text_height - 4, 0), x1 + text_width + 4, y1], fill=color)
        draw.text((x1 + 2, max(y1 - text_height - 2, 0)), label_text, fill="white", font=font)

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
        det = {
            "box": box,
            "score": scores[idx] if idx < len(scores) else None,
            "label": labels[idx] if idx < len(labels) else None,
        }
        if idx < len(masks):
            det["mask"] = masks[idx]
        detections.append(det)
    return detections


def _build_detection_table(
    detections: Sequence[Dict[str, Any]],
    class_map: Dict[str, str],
) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for idx, det in enumerate(detections):
        label_id = str(det.get("label", ""))
        label_name = class_map.get(label_id, f"class_{label_id}")
        score = det.get("score")
        box = det.get("box", [None, None, None, None])
        rows.append(
            [
                idx,
                label_name,
                round(float(score), 3) if score is not None else None,
                ", ".join(f"{coord:.1f}" for coord in box) if box else "",
            ]
        )
    return rows


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
    buffer.seek(0)

    files = {"image": ("upload.png", buffer.getvalue(), "image/png")}
    params = {
        "conf_thresh": conf_thresh,
        "iou_thresh": iou_thresh,
        "max_det": max_det,
        "include_masks": include_masks,
    }
    start = time.perf_counter()
    response = requests.post(
        f"{backend_url.rstrip('/')}/v1/predict",
        files=files,
        params={k: v for k, v in params.items() if v is not None},
        timeout=60,
    )
    duration_ms = (time.perf_counter() - start) * 1000.0
    response.raise_for_status()
    return response.json(), duration_ms


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
        buffer.seek(0)
        files.append(("images", (f"image_{idx}.png", buffer.getvalue(), "image/png")))

    params = {
        "conf_thresh": conf_thresh,
        "iou_thresh": iou_thresh,
        "max_det": max_det,
        "include_masks": include_masks,
    }
    start = time.perf_counter()
    response = requests.post(
        f"{backend_url.rstrip('/')}/v1/predict_batch",
        files=files,
        params={k: v for k, v in params.items() if v is not None},
        timeout=120,
    )
    duration_ms = (time.perf_counter() - start) * 1000.0
    response.raise_for_status()
    return response.json(), duration_ms


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

    backend = backend_url.strip() or DEFAULT_BACKEND_URL
    class_map = _parse_class_map(class_map_text)

    try:
        resp_json, http_latency = _send_single_request(
            image=image,
            backend_url=backend,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            max_det=max_det,
            include_masks=include_masks,
        )
    except Exception as exc:  # noqa: BLE001
        return None, [], {"error": str(exc)}, f"Request failed: {exc}"

    detections = _detections_from_response(resp_json)
    annotated = _annotate_image(image, detections, include_masks, class_map)
    table = _build_detection_table(detections, class_map)

    backend_latency = resp_json.get("inference_time_ms")
    latency_parts = []
    if backend_latency is not None:
        latency_parts.append(f"Model latency: {backend_latency:.2f} ms")
    latency_parts.append(f"HTTP round-trip: {http_latency:.2f} ms")
    latency_text = " | ".join(latency_parts)

    return annotated, table, resp_json, latency_text


def _load_images_from_files(file_inputs: Optional[List[Any]]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for file_obj in file_inputs or []:
        path: Optional[str] = None
        if isinstance(file_obj, dict) and "name" in file_obj:
            path = file_obj["name"]
        elif hasattr(file_obj, "name"):
            path = getattr(file_obj, "name")
        if not path:
            continue
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


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

    backend = backend_url.strip() or DEFAULT_BACKEND_URL
    class_map = _parse_class_map(class_map_text)

    try:
        resp_json, http_latency = _send_batch_request(
            images=images,
            backend_url=backend,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            max_det=max_det,
            include_masks=include_masks,
        )
    except Exception as exc:  # noqa: BLE001
        return [], {"error": str(exc)}, f"Batch request failed: {exc}"

    predictions = resp_json.get("predictions", [])
    gallery_items: List[Tuple[Image.Image, str]] = []
    for image, prediction in zip(images, predictions):
        detections = _detections_from_response(prediction)
        annotated = _annotate_image(image, detections, include_masks, class_map)
        summary_parts = []
        for det in detections:
            label_id = str(det.get("label"))
            label_name = class_map.get(label_id, f"class_{label_id}")
            score = float(det.get("score", 0.0))
            summary_parts.append(f"{label_name} ({score:.2f})")
        caption = ", ".join(summary_parts) if summary_parts else "No detections"
        gallery_items.append((annotated, caption))

    backend_latency = resp_json.get("total_inference_time_ms")
    latency_text = f"HTTP round-trip: {http_latency:.2f} ms"
    if backend_latency is not None:
        latency_text = f"Batch model latency: {backend_latency:.2f} ms | {latency_text}"

    return gallery_items, resp_json, latency_text


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Detektor Internal UI") as demo:
        gr.Markdown(
            """
            # Detektor Internal UI
            Upload images and run inference against the FastAPI backend.
            Configure the backend URL if the API is running on a different host/port.
            """
        )

        with gr.Accordion("Connection Settings", open=False):
            backend_input = gr.Textbox(value=DEFAULT_BACKEND_URL, label="Backend URL", placeholder="http://localhost:8000")
            class_map_input = gr.Textbox(
                label="Class map JSON (optional)",
                placeholder='{"0": "person", "1": "car"}',
            )

        with gr.Tab("Single Image"):
            single_image = gr.Image(type="pil", label="Upload image")
            conf_slider = gr.Slider(0.05, 0.95, value=0.35, step=0.01, label="Confidence threshold")
            iou_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.01, label="IoU threshold")
            max_det_slider = gr.Slider(1, 300, value=100, step=1, label="Max detections")
            include_masks_chk = gr.Checkbox(value=True, label="Include masks")
            run_button = gr.Button("Run Inference", variant="primary")

            annotated_output = gr.Image(label="Annotated image", type="pil")
            table_output = gr.Dataframe(
                headers=["#", "Class", "Score", "Box [x1, y1, x2, y2]"],
                datatype=["number", "str", "number", "str"],
                interactive=False,
                label="Detections",
            )
            json_output = gr.JSON(label="Raw JSON response")
            latency_output = gr.Textbox(label="Latency", interactive=False)

            run_button.click(
                fn=run_single_inference,
                inputs=[
                    single_image,
                    backend_input,
                    conf_slider,
                    iou_slider,
                    max_det_slider,
                    include_masks_chk,
                    class_map_input,
                ],
                outputs=[annotated_output, table_output, json_output, latency_output],
            )

        with gr.Tab("Batch (Experimental)"):
            batch_images = gr.Files(label="Upload multiple images", type="file")
            batch_conf = gr.Slider(0.05, 0.95, value=0.35, step=0.01, label="Confidence threshold")
            batch_iou = gr.Slider(0.1, 0.9, value=0.5, step=0.01, label="IoU threshold")
            batch_max_det = gr.Slider(1, 300, value=100, step=1, label="Max detections per image")
            batch_masks = gr.Checkbox(value=True, label="Include masks")
            batch_button = gr.Button("Run Batch Inference")

            gallery_output = gr.Gallery(label="Annotated results", show_label=True)
            batch_json_output = gr.JSON(label="Raw batch JSON response")
            batch_latency_output = gr.Textbox(label="Batch latency", interactive=False)

            batch_button.click(
                fn=run_batch_inference,
                inputs=[
                    batch_images,
                    backend_input,
                    batch_conf,
                    batch_iou,
                    batch_max_det,
                    batch_masks,
                    class_map_input,
                ],
                outputs=[gallery_output, batch_json_output, batch_latency_output],
            )

    return demo


def main() -> None:
    demo = build_interface()
    demo.queue()
    demo.launch(server_name=os.getenv("DETEKTOR_UI_HOST", "0.0.0.0"), server_port=int(os.getenv("DETEKTOR_UI_PORT", "7860")))


if __name__ == "__main__":
    main()
