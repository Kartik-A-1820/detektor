"""Tests for reporting module."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils.reporting import (
    generate_metrics_summary,
    load_epoch_summaries,
    load_train_metrics,
    plot_confusion_matrix,
    plot_epoch_metrics,
    plot_learning_rate,
    plot_loss_curves,
    plot_validation_metrics,
    save_per_class_metrics,
)


class TestReporting(unittest.TestCase):
    """Test reporting utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create sample training metrics
        self.sample_train_df = pd.DataFrame({
            "epoch": [1, 1, 1, 2, 2, 2],
            "step": [1, 2, 3, 4, 5, 6],
            "loss_total": [10.5, 9.8, 9.2, 8.5, 8.0, 7.5],
            "loss_cls": [3.5, 3.2, 3.0, 2.8, 2.6, 2.4],
            "loss_box": [4.0, 3.8, 3.6, 3.4, 3.2, 3.0],
            "loss_obj": [2.0, 1.9, 1.8, 1.7, 1.6, 1.5],
            "loss_mask": [1.0, 0.9, 0.8, 0.6, 0.6, 0.6],
            "lr": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0005],
        })
        
        # Create sample epoch summaries
        self.sample_epoch_df = pd.DataFrame({
            "epoch": [1, 2],
            "epoch_loss": [9.5, 8.0],
            "best_metric": [9.5, 8.0],
            "global_step": [3, 6],
        })
        
        # Create sample validation metrics
        self.sample_val_metrics = {
            "per_class_ap": [0.85, 0.92, 0.78, 0.88],
            "class_names": ["ball", "goalkeeper", "player", "referee"],
            "map50": 0.8575,
            "precision": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "recall": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "confusion_matrix": [
                [50, 2, 3, 1],
                [1, 45, 2, 0],
                [2, 1, 48, 3],
                [0, 1, 2, 47],
            ],
        }
    
    def test_load_train_metrics_csv(self):
        """Test loading training metrics from CSV."""
        csv_path = self.temp_path / "train_metrics.csv"
        self.sample_train_df.to_csv(csv_path, index=False)
        
        df = load_train_metrics(csv_path)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 6)
        self.assertIn("loss_total", df.columns)
    
    def test_load_train_metrics_jsonl(self):
        """Test loading training metrics from JSONL."""
        jsonl_path = self.temp_path / "train_metrics.jsonl"
        with open(jsonl_path, "w") as f:
            for _, row in self.sample_train_df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")
        
        df = load_train_metrics(jsonl_path)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 6)
        self.assertIn("loss_total", df.columns)
    
    def test_load_train_metrics_missing(self):
        """Test loading training metrics from non-existent file."""
        df = load_train_metrics(self.temp_path / "nonexistent.csv")
        self.assertIsNone(df)
    
    def test_load_epoch_summaries(self):
        """Test loading epoch summaries."""
        jsonl_path = self.temp_path / "epoch_summaries.jsonl"
        with open(jsonl_path, "w") as f:
            for _, row in self.sample_epoch_df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")
        
        df = load_epoch_summaries(jsonl_path)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertIn("epoch_loss", df.columns)
    
    def test_plot_loss_curves(self):
        """Test generating loss curve plots."""
        plots_dir = self.temp_path / "plots"
        saved_plots = plot_loss_curves(self.sample_train_df, plots_dir)
        
        self.assertGreater(len(saved_plots), 0)
        self.assertTrue((plots_dir / "loss_total.png").exists())
        self.assertTrue((plots_dir / "loss_components.png").exists())
    
    def test_plot_learning_rate(self):
        """Test generating learning rate plot."""
        plots_dir = self.temp_path / "plots"
        plot_path = plot_learning_rate(self.sample_train_df, plots_dir)
        
        self.assertIsNotNone(plot_path)
        self.assertTrue(plot_path.exists())
    
    def test_plot_epoch_metrics(self):
        """Test generating epoch metrics plot."""
        plots_dir = self.temp_path / "plots"
        plot_path = plot_epoch_metrics(self.sample_epoch_df, plots_dir)
        
        self.assertIsNotNone(plot_path)
        self.assertTrue(plot_path.exists())
    
    def test_plot_validation_metrics(self):
        """Test generating validation metrics plots."""
        plots_dir = self.temp_path / "plots"
        saved_plots = plot_validation_metrics(self.sample_val_metrics, plots_dir)
        
        self.assertGreater(len(saved_plots), 0)
        self.assertTrue((plots_dir / "per_class_ap.png").exists())
        self.assertTrue((plots_dir / "precision_recall_curve.png").exists())
    
    def test_plot_confusion_matrix(self):
        """Test generating confusion matrix plot."""
        plots_dir = self.temp_path / "plots"
        cm = np.array(self.sample_val_metrics["confusion_matrix"])
        class_names = self.sample_val_metrics["class_names"]
        
        plot_path = plot_confusion_matrix(cm, class_names, plots_dir, normalize=True)
        
        self.assertTrue(plot_path.exists())
    
    def test_generate_metrics_summary(self):
        """Test generating metrics summary."""
        summary_path = self.temp_path / "metrics_summary.json"
        summary = generate_metrics_summary(
            self.sample_train_df,
            self.sample_epoch_df,
            self.sample_val_metrics,
            summary_path,
        )
        
        self.assertTrue(summary_path.exists())
        self.assertIn("training", summary)
        self.assertIn("validation", summary)
        self.assertEqual(summary["training"]["total_steps"], 6)
        self.assertEqual(summary["training"]["total_epochs"], 2)
    
    def test_save_per_class_metrics(self):
        """Test saving per-class metrics to CSV."""
        csv_path = self.temp_path / "per_class_metrics.csv"
        saved_path = save_per_class_metrics(self.sample_val_metrics, csv_path)
        
        self.assertIsNotNone(saved_path)
        self.assertTrue(saved_path.exists())
        
        df = pd.read_csv(saved_path)
        self.assertEqual(len(df), 4)
        self.assertIn("class", df.columns)
        self.assertIn("ap50", df.columns)
    
    def test_graceful_degradation(self):
        """Test graceful degradation with missing data."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        plots_dir = self.temp_path / "plots"
        
        # Should not crash
        saved_plots = plot_loss_curves(empty_df, plots_dir)
        self.assertEqual(len(saved_plots), 0)
        
        # Missing LR column
        df_no_lr = self.sample_train_df.drop(columns=["lr"])
        lr_plot = plot_learning_rate(df_no_lr, plots_dir)
        self.assertIsNone(lr_plot)
        
        # Empty validation metrics
        val_plots = plot_validation_metrics({}, plots_dir)
        self.assertEqual(len(val_plots), 0)


if __name__ == "__main__":
    unittest.main()
