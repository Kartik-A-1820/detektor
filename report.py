"""Generate comprehensive training and validation reports with plots and summaries.

This module reads training metrics, epoch summaries, and validation results to generate
Ultralytics-style reporting artifacts including loss curves, learning rate plots,
validation metrics, and machine-readable summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

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


LOGGER = logging.getLogger("detektor.report")


def generate_report(
    run_dir: Path,
    output_plots_dir: Optional[Path] = None,
    output_reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate comprehensive training and validation report.
    
    Args:
        run_dir: Directory containing training artifacts (e.g., runs/chimera)
        output_plots_dir: Directory to save plots (default: run_dir/plots)
        output_reports_dir: Directory to save reports (default: run_dir/reports)
        
    Returns:
        Dictionary with report generation status and paths
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Set default output directories
    if output_plots_dir is None:
        output_plots_dir = run_dir / "plots"
    if output_reports_dir is None:
        output_reports_dir = run_dir / "reports"
    
    output_plots_dir.mkdir(parents=True, exist_ok=True)
    output_reports_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Generating report for run: {run_dir}")
    
    report_status = {
        "run_dir": str(run_dir),
        "plots_generated": [],
        "reports_generated": [],
        "warnings": [],
    }
    
    # Load training metrics
    train_metrics_csv = run_dir / "train_metrics.csv"
    train_metrics_jsonl = run_dir / "train_metrics.jsonl"
    
    train_df = None
    if train_metrics_csv.exists():
        LOGGER.info(f"Loading training metrics from {train_metrics_csv}")
        train_df = load_train_metrics(train_metrics_csv)
    elif train_metrics_jsonl.exists():
        LOGGER.info(f"Loading training metrics from {train_metrics_jsonl}")
        train_df = load_train_metrics(train_metrics_jsonl)
    else:
        report_status["warnings"].append("No training metrics found (train_metrics.csv or train_metrics.jsonl)")
        LOGGER.warning("No training metrics found")
    
    # Load epoch summaries
    epoch_summaries_path = run_dir / "epoch_summaries.jsonl"
    epoch_df = None
    if epoch_summaries_path.exists():
        LOGGER.info(f"Loading epoch summaries from {epoch_summaries_path}")
        epoch_df = load_epoch_summaries(epoch_summaries_path)
    else:
        report_status["warnings"].append("No epoch summaries found (epoch_summaries.jsonl)")
        LOGGER.warning("No epoch summaries found")
    
    # Load validation metrics if available
    val_metrics_path = run_dir / "val_metrics.json"
    val_metrics = None
    if val_metrics_path.exists():
        LOGGER.info(f"Loading validation metrics from {val_metrics_path}")
        with open(val_metrics_path, "r") as f:
            val_metrics = json.load(f)
    else:
        report_status["warnings"].append("No validation metrics found (val_metrics.json)")
        LOGGER.info("No validation metrics found (optional)")
    
    # Generate training plots
    if train_df is not None and not train_df.empty:
        LOGGER.info("Generating loss curves...")
        try:
            loss_plots = plot_loss_curves(train_df, output_plots_dir)
            report_status["plots_generated"].extend([str(p) for p in loss_plots])
            LOGGER.info(f"Generated {len(loss_plots)} loss curve plots")
        except Exception as e:
            LOGGER.error(f"Failed to generate loss curves: {e}")
            report_status["warnings"].append(f"Loss curves generation failed: {e}")
        
        LOGGER.info("Generating learning rate plot...")
        try:
            lr_plot = plot_learning_rate(train_df, output_plots_dir)
            if lr_plot:
                report_status["plots_generated"].append(str(lr_plot))
                LOGGER.info("Generated learning rate plot")
            else:
                report_status["warnings"].append("Learning rate column not found in metrics")
        except Exception as e:
            LOGGER.error(f"Failed to generate learning rate plot: {e}")
            report_status["warnings"].append(f"Learning rate plot generation failed: {e}")
    
    # Generate epoch plots
    if epoch_df is not None and not epoch_df.empty:
        LOGGER.info("Generating epoch metrics plot...")
        try:
            epoch_plot = plot_epoch_metrics(epoch_df, output_plots_dir)
            if epoch_plot:
                report_status["plots_generated"].append(str(epoch_plot))
                LOGGER.info("Generated epoch metrics plot")
        except Exception as e:
            LOGGER.error(f"Failed to generate epoch metrics plot: {e}")
            report_status["warnings"].append(f"Epoch metrics plot generation failed: {e}")
    
    # Generate validation plots
    if val_metrics is not None:
        LOGGER.info("Generating validation plots...")
        try:
            val_plots = plot_validation_metrics(val_metrics, output_plots_dir)
            report_status["plots_generated"].extend([str(p) for p in val_plots])
            LOGGER.info(f"Generated {len(val_plots)} validation plots")
        except Exception as e:
            LOGGER.error(f"Failed to generate validation plots: {e}")
            report_status["warnings"].append(f"Validation plots generation failed: {e}")
        
        # Generate confusion matrix if available
        if "confusion_matrix" in val_metrics and "class_names" in val_metrics:
            LOGGER.info("Generating confusion matrix...")
            try:
                import numpy as np
                cm = np.array(val_metrics["confusion_matrix"])
                cm_plot = plot_confusion_matrix(
                    cm,
                    val_metrics["class_names"],
                    output_plots_dir,
                    normalize=True,
                )
                report_status["plots_generated"].append(str(cm_plot))
                LOGGER.info("Generated confusion matrix plot")
            except Exception as e:
                LOGGER.error(f"Failed to generate confusion matrix: {e}")
                report_status["warnings"].append(f"Confusion matrix generation failed: {e}")
    
    # Generate machine-readable summary
    LOGGER.info("Generating metrics summary...")
    try:
        summary_path = output_reports_dir / "metrics_summary.json"
        summary = generate_metrics_summary(train_df, epoch_df, val_metrics, summary_path)
        report_status["reports_generated"].append(str(summary_path))
        LOGGER.info(f"Generated metrics summary: {summary_path}")
    except Exception as e:
        LOGGER.error(f"Failed to generate metrics summary: {e}")
        report_status["warnings"].append(f"Metrics summary generation failed: {e}")
    
    # Generate per-class metrics CSV
    if val_metrics is not None and "per_class_ap" in val_metrics:
        LOGGER.info("Generating per-class metrics CSV...")
        try:
            per_class_path = output_reports_dir / "per_class_metrics.csv"
            csv_path = save_per_class_metrics(val_metrics, per_class_path)
            if csv_path:
                report_status["reports_generated"].append(str(csv_path))
                LOGGER.info(f"Generated per-class metrics: {csv_path}")
        except Exception as e:
            LOGGER.error(f"Failed to generate per-class metrics: {e}")
            report_status["warnings"].append(f"Per-class metrics generation failed: {e}")
    
    # Save report status
    status_path = output_reports_dir / "report_status.json"
    with open(status_path, "w") as f:
        json.dump(report_status, f, indent=2)
    
    LOGGER.info(f"Report generation complete. Status saved to: {status_path}")
    LOGGER.info(f"Generated {len(report_status['plots_generated'])} plots")
    LOGGER.info(f"Generated {len(report_status['reports_generated'])} reports")
    
    if report_status["warnings"]:
        LOGGER.warning(f"Encountered {len(report_status['warnings'])} warnings during report generation")
    
    return report_status


def main() -> None:
    """CLI entrypoint for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive training and validation reports"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory containing training artifacts (e.g., runs/chimera)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: <run-dir>/plots)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Directory to save reports (default: <run-dir>/reports)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(message)s",
    )
    
    # Generate report
    try:
        run_dir = Path(args.run_dir)
        plots_dir = Path(args.plots_dir) if args.plots_dir else None
        reports_dir = Path(args.reports_dir) if args.reports_dir else None
        
        status = generate_report(run_dir, plots_dir, reports_dir)
        
        print("\n" + "=" * 60)
        print("REPORT GENERATION SUMMARY")
        print("=" * 60)
        print(f"Run directory: {status['run_dir']}")
        print(f"Plots generated: {len(status['plots_generated'])}")
        print(f"Reports generated: {len(status['reports_generated'])}")
        
        if status["warnings"]:
            print(f"\nWarnings ({len(status['warnings'])}):")
            for warning in status["warnings"]:
                print(f"  - {warning}")
        
        print("\nPlots saved to:")
        for plot_path in status["plots_generated"]:
            print(f"  - {plot_path}")
        
        print("\nReports saved to:")
        for report_path in status["reports_generated"]:
            print(f"  - {report_path}")
        
        print("=" * 60)
        
    except Exception as e:
        LOGGER.error(f"Report generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
