from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import yaml

from api.run_artifacts import discover_checkpoint_options, load_run_artifacts


def _make_tmp_dir() -> Path:
    root = Path("F:/detektor/.tmp_testdata/run_artifacts")
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


class RunArtifactsTests(unittest.TestCase):
    def test_discover_checkpoint_options_prefers_best(self) -> None:
        run_dir = _make_tmp_dir()
        try:
            (run_dir / "chimera_best.pt").write_bytes(b"best")
            (run_dir / "chimera_last.pt").write_bytes(b"last")

            resolved_dir, options, default_key = discover_checkpoint_options(run_dir / "chimera_last.pt")

            self.assertEqual(resolved_dir, run_dir.resolve())
            self.assertEqual(default_key, "best")
            self.assertIn("best", options)
            self.assertIn("last", options)
        finally:
            for path in sorted(run_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                else:
                    path.rmdir()
            run_dir.rmdir()

    def test_load_run_artifacts_reads_training_outputs(self) -> None:
        run_dir = _make_tmp_dir()
        try:
            checkpoint_path = run_dir / "chimera_best.pt"
            checkpoint_path.write_bytes(b"placeholder")
            (run_dir / "plots").mkdir()
            (run_dir / "plots" / "loss_total.png").write_bytes(b"png")

            (run_dir / "resolved_train_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "num_classes": 2,
                            "names": ["player", "referee"],
                            "train": "F:/data/train",
                            "val": "F:/data/valid",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "resolved_runtime": {
                            "dataset_yaml": "F:/data/data.yaml",
                            "dataset_size": 12,
                            "model_display_name": "Comet",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "training_summary.json").write_text(
                json.dumps({"training": {"final_loss": 1.23}, "validation": {"val_map50": 0.42}}),
                encoding="utf-8",
            )
            (run_dir / "val_metrics.jsonl").write_text(
                '{"epoch": 1, "val_map50": 0.2, "val_recall": 0.3, "val_precision": 0.4, "val_mean_iou": 0.5}\n',
                encoding="utf-8",
            )
            (run_dir / "train_metrics.csv").write_text(
                "epoch,step,loss_total,lr\n1,1,3.0,0.01\n1,2,2.5,0.009\n",
                encoding="utf-8",
            )

            with patch(
                "api.run_artifacts.torch.load",
                return_value={"epoch": 4, "best_metric": 0.42, "model_config": {"display_name": "Comet"}},
            ):
                state = load_run_artifacts(run_dir, checkpoint_path, "best")

            self.assertEqual(state["active_checkpoint_key"], "best")
            self.assertEqual(state["dataset"]["num_classes"], 2)
            self.assertEqual(state["class_map"]["0"], "player")
            self.assertEqual(state["training_summary"]["final_loss"], 1.23)
            self.assertEqual(len(state["validation_history"]), 1)
            self.assertEqual(len(state["train_curve"]), 2)
            self.assertIn("loss_total", state["plots"])
        finally:
            for path in sorted(run_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                else:
                    path.rmdir()
            run_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
