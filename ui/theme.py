"""Theme and CSS for the Detektor console. Works in light and dark mode."""

from __future__ import annotations

import gradio as gr

CSS = """
:root {
  --det-accent: #4f6bff;
  --det-accent-2: #14b8a6;
  --det-radius: 14px;
  --det-ok: #16a34a;
  --det-warn: #d97706;
}
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* Header ---------------------------------------------------------------- */
.det-header {
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;
  padding: 18px 22px; border: 1px solid var(--border-color-primary); border-radius: var(--det-radius);
  background:
    radial-gradient(1200px 220px at 0% 0%, color-mix(in srgb, var(--det-accent) 14%, transparent), transparent 70%),
    radial-gradient(800px 200px at 100% 100%, color-mix(in srgb, var(--det-accent-2) 14%, transparent), transparent 70%),
    var(--block-background-fill);
}
.det-brand { display: flex; align-items: center; gap: 14px; }
.det-title { font-size: 1.45rem; font-weight: 750; letter-spacing: -0.02em; color: var(--body-text-color); line-height: 1.15; }
.det-subtitle { font-size: 0.86rem; color: var(--body-text-color-subdued); margin-top: 2px; }
.det-pills { display: flex; flex-wrap: wrap; gap: 8px; }
.det-pill {
  display: inline-flex; align-items: center; padding: 4px 11px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
  border: 1px solid var(--border-color-primary); color: var(--body-text-color);
  background: color-mix(in srgb, var(--body-text-color) 5%, transparent);
}
.det-pill-ok { color: var(--det-ok); border-color: color-mix(in srgb, var(--det-ok) 40%, transparent);
               background: color-mix(in srgb, var(--det-ok) 10%, transparent); }
.det-pill-warn { color: var(--det-warn); border-color: color-mix(in srgb, var(--det-warn) 40%, transparent);
                 background: color-mix(in srgb, var(--det-warn) 10%, transparent); }
.det-pill-muted { color: var(--body-text-color-subdued); }

/* KPI cards -------------------------------------------------------------- */
.det-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.det-kpi {
  padding: 14px 16px; border: 1px solid var(--border-color-primary); border-radius: 12px;
  background: var(--block-background-fill);
  transition: transform .15s ease, box-shadow .15s ease;
}
.det-kpi:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(30, 41, 59, 0.08); }
.det-kpi-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--body-text-color-subdued); font-weight: 600; }
.det-kpi-value { font-size: 1.35rem; font-weight: 750; letter-spacing: -0.01em; margin-top: 4px; color: var(--body-text-color); }
.det-kpi-hint { font-size: 0.76rem; color: var(--body-text-color-subdued); margin-top: 2px; }

/* Empty state ------------------------------------------------------------ */
.det-empty {
  text-align: center; padding: 44px 20px; border: 1.5px dashed var(--border-color-primary); border-radius: var(--det-radius);
  background: color-mix(in srgb, var(--body-text-color) 2.5%, transparent);
}
.det-empty-icon { font-size: 2rem; }
.det-empty-title { font-weight: 700; margin-top: 8px; color: var(--body-text-color); }
.det-empty-hint { color: var(--body-text-color-subdued); font-size: 0.9rem; margin-top: 4px; }

/* Controls --------------------------------------------------------------- */
.det-run button, button.det-run { font-weight: 700 !important; letter-spacing: 0.01em; }
.det-section-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 700;
                     color: var(--body-text-color-subdued); margin: 6px 2px 2px; }
.det-footer { text-align: center; color: var(--body-text-color-subdued); font-size: 0.8rem; padding: 18px 0 6px; }
.det-footer a { color: var(--det-accent); text-decoration: none; }
.det-note { color: var(--body-text-color-subdued); font-size: 0.85rem; }
.tab-nav button, .tabs > div > button { font-weight: 600 !important; }

@media (max-width: 720px) {
  .det-header { padding: 14px; }
  .det-kpi-value { font-size: 1.15rem; }
}
"""


def build_theme() -> gr.themes.Base:
    """A neutral, modern theme with an indigo primary and teal secondary accent."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        text_size=gr.themes.sizes.text_md,
        font=["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        font_mono=["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
    ).set(
        button_primary_background_fill="linear-gradient(135deg, #4f6bff 0%, #6d5bff 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #3f5bf0 0%, #5d4bf0 100%)",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_label_text_weight="600",
    )


def gradio_major() -> int:
    try:
        return int(gr.__version__.split(".")[0])
    except Exception:  # noqa: BLE001
        return 4


def style_kwargs() -> dict:
    """``theme``/``css`` kwargs for Blocks (Gradio < 6) or launch()/mount_gradio_app() (Gradio >= 6)."""
    return {"theme": build_theme(), "css": CSS}


def blocks_kwargs() -> dict:
    return style_kwargs() if gradio_major() < 6 else {}


def mount_kwargs() -> dict:
    """Extra kwargs to pass to ``gr.mount_gradio_app`` / ``Blocks.launch``."""
    return style_kwargs() if gradio_major() >= 6 else {}
