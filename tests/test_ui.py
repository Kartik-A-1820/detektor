"""Tests for the console's rendering helpers and Gradio wiring (no browser required)."""

from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ui import render
from ui.app import (
    PRESETS,
    DetektorUIRuntime,
    _collect_image_paths,
    _dashboard_outputs_from_state,
    _parse_class_map,
    build_interface,
)

CLASS_MAP = {"0": "person", "1": "car"}


def _png_b64(mask: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class RenderTests(unittest.TestCase):
    def test_class_colors_are_stable_and_wrap(self) -> None:
        self.assertEqual(render.class_color(0), render.class_color(len(render.CLASS_COLORS)))
        self.assertNotEqual(render.class_color(0), render.class_color(1))
        self.assertEqual(render.class_color("garbage"), render.class_color(0))

    def test_detections_from_response_supports_both_shapes(self) -> None:
        new = {"detections": [{"box": [1, 2, 3, 4], "score": 0.9, "label": 0}]}
        self.assertEqual(len(render.detections_from_response(new)), 1)
        legacy = {"boxes": [[1, 2, 3, 4], [5, 6, 7, 8]], "scores": [0.9, 0.8], "labels": [0, 1], "masks": ["m", ""]}
        dets = render.detections_from_response(legacy)
        self.assertEqual([d["label"] for d in dets], [0, 1])
        self.assertIn("mask", dets[0])
        self.assertNotIn("mask", dets[1])
        self.assertEqual(render.detections_from_response({}), [])

    def test_summaries(self) -> None:
        self.assertEqual(render.summarize_detections([], CLASS_MAP), "No detections")
        dets = [{"label": 0}, {"label": 0}, {"label": 1}]
        self.assertEqual(render.summarize_detections(dets, CLASS_MAP), "2× person, car")
        self.assertEqual(render.class_name(7, CLASS_MAP), "class 7")

    def test_annotate_image_draws_boxes_masks_and_clamps_labels(self) -> None:
        image = Image.new("RGB", (200, 120), (30, 30, 30))
        mask = np.zeros((120, 200), dtype=np.uint8)
        mask[20:60, 20:80] = 1
        dets = [
            {"box": [20, 20, 80, 60], "score": 0.91, "label": 0, "mask": _png_b64(mask)},
            {"box": [150, 0, 199, 40], "score": 0.5, "label": 1},  # touches top & right edge
        ]
        plain = render.annotate_image(image, dets, False, CLASS_MAP)
        masked = render.annotate_image(image, dets, True, CLASS_MAP)
        self.assertEqual(plain.size, image.size)
        self.assertNotEqual(np.asarray(plain).tobytes(), np.asarray(image).tobytes())
        # mask overlay changes interior pixels relative to the box-only render
        self.assertNotEqual(np.asarray(plain)[40, 50].tolist(), np.asarray(masked)[40, 50].tolist())
        # original untouched
        self.assertEqual(image.getpixel((50, 40)), (30, 30, 30))

    def test_html_is_escaped(self) -> None:
        html_out = render.kpi_cards([("<b>x</b>", "<script>alert(1)</script>", "")])
        self.assertNotIn("<script>", html_out)
        self.assertIn("&lt;script&gt;", html_out)
        self.assertNotIn("<script>", render.empty_state("<img onerror=x>", "<i>"))

    def test_header_reflects_state(self) -> None:
        state = {"device": "cuda", "active_checkpoint_key": "best", "runtime": {"model_display_name": "Nova"}}
        out = render.header_html(state, "9.9.9")
        for needle in ("CUDA", "Nova", "best", "v9.9.9", "Ready"):
            self.assertIn(needle, out)
        self.assertIn("Starting", render.header_html({}, "1"))

    def test_results_summary_math(self) -> None:
        out = render.results_summary_html(4, 10, model_ms=200.0, wall_ms=1500.0, checkpoint="best")
        self.assertIn("50.0 ms", out)       # 200 / 4
        self.assertIn("20.0 img/s", out)    # 1000 / 50
        self.assertIn("2.5 per image", out)
        self.assertIn("1.50 s", out)

    def test_class_map_rows_sorted_numerically(self) -> None:
        rows = render.class_map_rows({"10": "j", "2": "c", "1": "b"})
        self.assertEqual([r[0] for r in rows], [1, 2, 10])

    def test_figures_handle_empty_and_populated_state(self) -> None:
        self.assertIsNotNone(render.training_figure({}))
        self.assertIsNotNone(render.validation_figure({}))
        self.assertIsNotNone(render.latency_figure([]))
        state = {
            "train_curve": [{"step": i, "loss_total": 5 - i * 0.1, "lr": 0.001} for i in range(10)],
            "validation_history": [{"epoch": e, "val_map50": e / 10, "val_recall": 0.5, "val_precision": None, "val_mean_iou": 0.6} for e in range(1, 5)],
        }
        self.assertIsNotNone(render.training_figure(state))
        self.assertIsNotNone(render.validation_figure(state))
        self.assertIsNotNone(render.latency_figure([10.0, 11.0, 12.5, 9.8, 30.0]))


class FakeService:
    """Minimal InferenceService stand-in returning one detection per image."""

    image_size = 64
    device = "cpu"

    def predict_batch(self, images_bytes, **kwargs):
        preds = [
            {"num_detections": 1, "detections": [{"box": [4, 4, 30, 30], "score": 0.8, "label": 0}], "image_width": 64, "image_height": 64}
            for _ in images_bytes
        ]
        return preds, 12.0 * len(images_bytes)


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = []
        for i in range(3):
            path = Path(self.tmp.name) / f"img_{i}.png"
            Image.new("RGB", (64, 64), (i * 40, 80, 120)).save(path)
            self.paths.append(path)
        (Path(self.tmp.name) / "notes.txt").write_text("not an image")
        state = {"class_map": CLASS_MAP, "active_checkpoint_key": "best", "available_checkpoints": {"best": "b.pt", "last": "l.pt"}}
        self.state = state
        self.runtime = DetektorUIRuntime(
            get_runtime_state=lambda: state,
            get_service_snapshot=lambda: (FakeService(), SimpleNamespace(max_batch_size=2)),
            select_checkpoint=lambda key: state.update(active_checkpoint_key=key) or state,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_collect_image_paths_filters_dedups_and_recurses(self) -> None:
        found = _collect_image_paths([str(self.paths[0])], self.tmp.name)
        self.assertEqual(len(found), 3)  # file upload + folder scan de-duplicated, txt ignored
        self.assertEqual(_collect_image_paths(None, ""), [])
        self.assertEqual(_collect_image_paths(["/nonexistent.png"], "/nonexistent-dir"), [])

    def test_run_inference_batches_and_summarises(self) -> None:
        summary, gallery, rows, payload, download = self.runtime.run_inference(
            [str(p) for p in self.paths], "", 0.25, 0.6, 50, False
        )
        self.assertIn("Detections", summary)
        self.assertEqual(len(gallery["value"]), 3)
        self.assertTrue(gallery["visible"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(payload["total_detections"], 3)
        self.assertEqual(payload["settings"]["max_det"], 50)
        self.assertTrue(Path(download).exists())
        self.assertIn('"total_detections": 3', Path(download).read_text())

    def test_run_inference_without_images_is_a_noop(self) -> None:
        summary, gallery, rows, payload, download = self.runtime.run_inference([], "", 0.25, 0.6, 10, False)
        self.assertIn("No results yet", summary)
        self.assertFalse(gallery["visible"])
        self.assertIsNone(download)

    def test_set_checkpoint_switches_and_refreshes(self) -> None:
        outputs = self.runtime.set_checkpoint("last")
        self.assertEqual(self.state["active_checkpoint_key"], "last")
        self.assertEqual(len(outputs), 12)

    def test_live_benchmark_runs_on_real_tiny_model(self) -> None:
        from benchmarks.common import build_profile_model

        model = build_profile_model("firefly", 2, "cpu")
        service = SimpleNamespace(model=model, device="cpu", image_size=64)
        runtime = DetektorUIRuntime(
            get_runtime_state=lambda: self.state,
            get_service_snapshot=lambda: (service, None),
            select_checkpoint=lambda k: self.state,
        )
        cards, fig = runtime.run_benchmark(1, 5)
        self.assertIn("Median", cards)
        self.assertIn("img/s", cards)
        self.assertIsNotNone(fig)

    def test_dashboard_outputs_shape_with_empty_state(self) -> None:
        self.assertEqual(len(_dashboard_outputs_from_state({})), 12)


class MiscTests(unittest.TestCase):
    def test_presets_are_sane(self) -> None:
        for name, (conf, iou) in PRESETS.items():
            self.assertTrue(0 < conf < 1 and 0 < iou < 1, name)
        self.assertGreater(PRESETS["High precision"][0], PRESETS["High recall"][0])

    def test_parse_class_map(self) -> None:
        self.assertEqual(_parse_class_map('{"0": "a", "1": "b"}'), {"0": "a", "1": "b"})
        self.assertEqual(_parse_class_map("not json"), {})
        self.assertEqual(_parse_class_map("[1,2]"), {})
        self.assertEqual(_parse_class_map(""), {})

    def test_interfaces_build(self) -> None:
        self.assertIsNotNone(build_interface())  # remote mode
        runtime = DetektorUIRuntime(
            get_runtime_state=lambda: {}, get_service_snapshot=lambda: (None, None), select_checkpoint=lambda k: {}
        )
        self.assertIsNotNone(build_interface(runtime=runtime))  # serving mode


if __name__ == "__main__":
    unittest.main()
