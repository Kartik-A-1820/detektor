"""Pure rendering helpers for the Detektor console (no Gradio imports, fully unit-testable)."""

from __future__ import annotations

import base64
import html
import io
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

# Okabe–Ito colour-blind-safe palette (RGB). Colours are assigned per *class*, so the same
# class always has the same colour across images, galleries and tables.
CLASS_COLORS: List[Tuple[int, int, int]] = [
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
    (153, 153, 153),
]

_FONT_CACHE: Dict[int, ImageFont.ImageFont] = {}


def class_color(label: Any) -> Tuple[int, int, int]:
    try:
        index = int(label)
    except (TypeError, ValueError):
        index = 0
    return CLASS_COLORS[index % len(CLASS_COLORS)]


def _font(size: int) -> ImageFont.ImageFont:
    size = max(10, int(size))
    if size not in _FONT_CACHE:
        font: Optional[ImageFont.ImageFont] = None
        for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
            try:
                font = ImageFont.truetype(name, size)
                break
            except Exception:  # noqa: BLE001
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=size)  # Pillow >= 10.1
            except Exception:  # noqa: BLE001
                font = ImageFont.load_default()
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

def detections_from_response(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalise both the new (``detections``) and legacy (parallel arrays) response shapes."""
    if resp.get("detections") is not None:
        return list(resp["detections"])
    boxes, scores, labels, masks = (resp.get(k) or [] for k in ("boxes", "scores", "labels", "masks"))
    out = []
    for i, box in enumerate(boxes):
        det: Dict[str, Any] = {
            "box": box,
            "score": scores[i] if i < len(scores) else 0.0,
            "label": labels[i] if i < len(labels) else 0,
        }
        if i < len(masks) and masks[i]:
            det["mask"] = masks[i]
        out.append(det)
    return out


def class_name(label: Any, class_map: Dict[str, str]) -> str:
    return class_map.get(str(label), f"class {label}")


def summarize_detections(detections: Sequence[Dict[str, Any]], class_map: Dict[str, str], limit: int = 6) -> str:
    if not detections:
        return "No detections"
    counts: Dict[str, int] = {}
    for det in detections:
        name = class_name(det.get("label"), class_map)
        counts[name] = counts.get(name, 0) + 1
    parts = [f"{n}× {name}" if n > 1 else name for name, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    if len(parts) > limit:
        parts = parts[:limit] + [f"+{len(parts) - limit} more"]
    return ", ".join(parts)


def mask_to_array(mask_b64: str, size_hw: Tuple[int, int]) -> np.ndarray:
    image = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
    if image.size != (size_hw[1], size_hw[0]):
        image = image.resize((size_hw[1], size_hw[0]), Image.NEAREST)
    return np.array(image, dtype=np.uint8)


def annotate_image(
    image: Image.Image,
    detections: Sequence[Dict[str, Any]],
    include_masks: bool,
    class_map: Dict[str, str],
) -> Image.Image:
    """Draw boxes, labels and (optionally) translucent masks; scales stroke/text to image size."""
    base = image.convert("RGBA")
    width, height = base.size
    stroke = max(2, round(min(width, height) / 280))
    font = _font(max(12, round(min(width, height) / 38)))

    if include_masks:
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        for det in detections:
            if det.get("mask"):
                mask = mask_to_array(det["mask"], (height, width)) > 0
                overlay[mask, :3] = class_color(det.get("label"))
                overlay[mask, 3] = 96
        base = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))

    draw = ImageDraw.Draw(base)
    for det in detections:
        box = det.get("box") or det.get("boxes")
        if not box:
            continue
        if isinstance(box[0], (list, tuple)):
            box = box[0]
        x1, y1, x2, y2 = (float(v) for v in box)
        color = class_color(det.get("label"))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)
        text = f"{class_name(det.get('label'), class_map)} {float(det.get('score') or 0):.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        pad = max(3, stroke + 1)
        ty = y1 - th - 2 * pad
        if ty < 0:  # keep the label inside the frame when the box touches the top edge
            ty = y1
        tx = min(max(x1, 0.0), max(width - tw - 2 * pad, 0.0))  # keep the label inside the frame
        draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=color)
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        draw.text((tx + pad, ty + pad - top), text, fill=(20, 20, 20) if luminance > 150 else (255, 255, 255), font=font)
    return base.convert("RGB")


# ---------------------------------------------------------------------------
# HTML fragments (use Gradio CSS variables so they adapt to light/dark automatically)
# ---------------------------------------------------------------------------

LOGO_SVG = (
    '<svg viewBox="0 0 48 48" width="40" height="40" aria-hidden="true">'
    '<rect width="48" height="48" rx="12" fill="url(#dg)"/>'
    '<defs><linearGradient id="dg" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#4f6bff"/><stop offset="1" stop-color="#14b8a6"/></linearGradient></defs>'
    '<path d="M13 19v-6h6M29 13h6v6M35 29v6h-6M19 35h-6v-6" fill="none" stroke="#fff" stroke-width="3" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="24" cy="24" r="4.5" fill="#fff"/></svg>'
)


def esc(value: Any) -> str:
    return html.escape("n/a" if value in (None, "") else str(value))


def fmt_metric(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pill(text: str, tone: str = "neutral") -> str:
    return f'<span class="det-pill det-pill-{tone}">{esc(text)}</span>'


def header_html(state: Dict[str, Any], version: str) -> str:
    runtime = state.get("runtime", {})
    checkpoint = state.get("checkpoint_summary", {})
    model_name = runtime.get("model_display_name") or checkpoint.get("model_config", {}).get("display_name")
    pills = [pill("● Ready" if state else "● Starting", "ok" if state else "warn")]
    if state.get("device"):
        pills.append(pill(str(state["device"]).upper(), "neutral"))
    if model_name:
        pills.append(pill(f"Model · {model_name}", "neutral"))
    if state.get("active_checkpoint_key"):
        pills.append(pill(f"Checkpoint · {state['active_checkpoint_key']}", "neutral"))
    pills.append(pill(f"v{version}", "muted"))
    return (
        '<div class="det-header">'
        f'<div class="det-brand">{LOGO_SVG}<div><div class="det-title">Detektor</div>'
        '<div class="det-subtitle">Object detection &amp; instance segmentation console</div></div></div>'
        f'<div class="det-pills">{"".join(pills)}</div></div>'
    )


def kpi_cards(items: Sequence[Tuple[str, Any, str]]) -> str:
    """Cards from ``(label, value, hint)`` tuples."""
    cards = "".join(
        f'<div class="det-kpi"><div class="det-kpi-label">{esc(label)}</div>'
        f'<div class="det-kpi-value">{esc(value)}</div>'
        + (f'<div class="det-kpi-hint">{esc(hint)}</div>' if hint else "")
        + "</div>"
        for label, value, hint in items
    )
    return f'<div class="det-kpis">{cards}</div>'


def empty_state(title: str, hint: str, icon: str = "🖼️") -> str:
    return (
        f'<div class="det-empty"><div class="det-empty-icon">{icon}</div>'
        f'<div class="det-empty-title">{esc(title)}</div><div class="det-empty-hint">{esc(hint)}</div></div>'
    )


def overview_html(state: Dict[str, Any]) -> str:
    runtime = state.get("runtime", {})
    checkpoint = state.get("checkpoint_summary", {})
    dataset = state.get("dataset", {})
    best = checkpoint.get("best_metric")
    return kpi_cards(
        [
            ("Checkpoint", str(state.get("active_checkpoint_key", "n/a")).title(),
             f"{checkpoint['file_size_mb']} MB" if checkpoint.get("file_size_mb") else ""),
            ("Architecture", runtime.get("model_display_name") or checkpoint.get("model_config", {}).get("display_name"), ""),
            ("Best metric", fmt_metric(best, 4), "validation mAP50" if best is not None else ""),
            ("Epoch", checkpoint.get("epoch"), ""),
            ("Classes", dataset.get("num_classes"), ""),
            ("Dataset size", dataset.get("dataset_size"), "training images"),
            ("Input size", f"{runtime['img_size']} px" if runtime.get("img_size") else None, ""),
            ("Device", state.get("device"), ""),
        ]
    )


def results_summary_html(num_images: int, detections: int, model_ms: float, wall_ms: float, checkpoint: Any) -> str:
    per_image = model_ms / num_images if num_images else 0.0
    fps = 1000.0 / per_image if per_image > 0 else 0.0
    return kpi_cards(
        [
            ("Images", f"{num_images:,}", f"checkpoint · {checkpoint}"),
            ("Detections", f"{detections:,}", f"{detections / max(num_images, 1):.1f} per image"),
            ("Model time", f"{per_image:.1f} ms", "per image"),
            ("Throughput", f"{fps:.1f} img/s", "model only"),
            ("Wall time", f"{wall_ms / 1000.0:.2f} s", "incl. decode & drawing"),
        ]
    )


def class_map_rows(class_map: Dict[str, str]) -> List[List[Any]]:
    def key(item: Tuple[str, str]) -> Tuple[int, str]:
        return (int(item[0]), item[1]) if str(item[0]).isdigit() else (10**9, item[0])

    return [[int(k) if str(k).isdigit() else k, v] for k, v in sorted(class_map.items(), key=key)]


# ---------------------------------------------------------------------------
# Charts (matplotlib Figure objects are *not* registered with pyplot, so they never leak)
# ---------------------------------------------------------------------------

_INK = "#8b95a7"
_GRID = "#8b95a7"
ACCENT = "#4f6bff"
ACCENT_2 = "#14b8a6"
ACCENT_3 = "#f59e0b"


def _style_axes(ax: Any, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, fontweight="semibold", color=_INK, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=_INK)
    ax.set_ylabel(ylabel, fontsize=9, color=_INK)
    ax.tick_params(colors=_INK, labelsize=8)
    ax.grid(alpha=0.18, color=_GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK)
        ax.spines[side].set_alpha(0.4)


def _new_figure(ncols: int, width: float = 11.0, height: float = 3.4) -> Tuple[Figure, Any]:
    fig = Figure(figsize=(width, height), dpi=110)
    fig.patch.set_alpha(0.0)
    axes = fig.subplots(1, ncols)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor("none")
    return fig, axes


def _placeholder(ax: Any, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=_INK, fontsize=10, alpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])


def training_figure(state: Dict[str, Any]) -> Figure:
    rows = state.get("train_curve", [])
    fig, (ax_loss, ax_lr) = _new_figure(2)
    steps = [r["step"] for r in rows if r.get("step") is not None and r.get("loss_total") is not None]
    losses = [r["loss_total"] for r in rows if r.get("step") is not None and r.get("loss_total") is not None]
    if steps:
        ax_loss.plot(steps, losses, color=ACCENT, linewidth=1.8)
        ax_loss.fill_between(steps, losses, min(losses), color=ACCENT, alpha=0.08)
    else:
        _placeholder(ax_loss, "No training log found for this run")
    lr_steps = [r["step"] for r in rows if r.get("step") is not None and r.get("lr") is not None]
    lrs = [r["lr"] for r in rows if r.get("step") is not None and r.get("lr") is not None]
    if lr_steps:
        ax_lr.plot(lr_steps, lrs, color=ACCENT_2, linewidth=1.8)
    else:
        _placeholder(ax_lr, "No learning-rate log")
    _style_axes(ax_loss, "Training loss", "step", "loss")
    _style_axes(ax_lr, "Learning rate", "step", "lr")
    fig.tight_layout()
    return fig


def validation_figure(state: Dict[str, Any]) -> Figure:
    rows = [r for r in state.get("validation_history", []) if r.get("epoch") is not None]
    fig, (ax_a, ax_b) = _new_figure(2)
    if rows:
        epochs = [r["epoch"] for r in rows]
        for key, label, color in (("val_map50", "mAP50", ACCENT), ("val_recall", "Recall", ACCENT_2), ("val_precision", "Precision", ACCENT_3)):
            vals = [r.get(key) for r in rows]
            if any(v is not None for v in vals):
                ax_a.plot(epochs, [np.nan if v is None else v for v in vals], marker="o", markersize=3.5, linewidth=1.8, color=color, label=label)
        ious = [r.get("val_mean_iou") for r in rows]
        if any(v is not None for v in ious):
            ax_b.plot(epochs, [np.nan if v is None else v for v in ious], marker="o", markersize=3.5, linewidth=1.8, color="#a855f7", label="Mean IoU")
        ax_a.legend(frameon=False, fontsize=8, labelcolor=_INK)
        ax_b.legend(frameon=False, fontsize=8, labelcolor=_INK)
        ax_a.set_ylim(0, 1.02)
        ax_b.set_ylim(0, 1.02)
    else:
        _placeholder(ax_a, "No validation history — train with --run-val")
        _placeholder(ax_b, "No validation history")
    _style_axes(ax_a, "Validation metrics", "epoch")
    _style_axes(ax_b, "Validation IoU", "epoch")
    fig.tight_layout()
    return fig


def latency_figure(latencies_ms: Sequence[float]) -> Figure:
    fig, (ax_hist, ax_line) = _new_figure(2, height=3.2)
    if len(latencies_ms):
        arr = np.asarray(latencies_ms, dtype=float)
        ax_hist.hist(arr, bins=min(24, max(5, len(arr) // 2)), color=ACCENT, alpha=0.85, edgecolor="none")
        for pct, color in ((50, ACCENT_2), (95, ACCENT_3)):
            value = float(np.percentile(arr, pct))
            ax_hist.axvline(value, color=color, linewidth=1.6, linestyle="--", label=f"p{pct} {value:.1f} ms")
        ax_hist.legend(frameon=False, fontsize=8, labelcolor=_INK)
        ax_line.plot(range(1, len(arr) + 1), arr, color=ACCENT, linewidth=1.5)
        ax_line.set_ylim(bottom=0)
    else:
        _placeholder(ax_hist, "Run the benchmark to see results")
        _placeholder(ax_line, "")
    _style_axes(ax_hist, "Latency distribution", "ms per batch", "runs")
    _style_axes(ax_line, "Latency per run", "run", "ms")
    fig.tight_layout()
    return fig


def results_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)
