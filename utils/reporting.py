"""Reporting utilities for generating training and validation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def load_train_metrics(metrics_path: Path) -> Optional[pd.DataFrame]:
    """Load training metrics from CSV or JSONL file.
    
    Args:
        metrics_path: Path to metrics file (CSV or JSONL)
        
    Returns:
        DataFrame with metrics or None if file doesn't exist
    """
    if not metrics_path.exists():
        return None
    
    if metrics_path.suffix == ".csv":
        return pd.read_csv(metrics_path)
    elif metrics_path.suffix == ".jsonl":
        records = []
        with open(metrics_path, "r") as f:
            for line in f:
                records.append(json.loads(line))
        return pd.DataFrame(records)
    return None


def load_epoch_summaries(summaries_path: Path) -> Optional[pd.DataFrame]:
    """Load epoch summaries from JSONL file.
    
    Args:
        summaries_path: Path to epoch summaries JSONL
        
    Returns:
        DataFrame with epoch summaries or None if file doesn't exist
    """
    if not summaries_path.exists():
        return None
    
    records = []
    with open(summaries_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def plot_loss_curves(
    df: pd.DataFrame,
    output_dir: Path,
    loss_columns: Optional[List[str]] = None,
) -> List[Path]:
    """Generate loss curve plots from training metrics.
    
    Args:
        df: DataFrame with training metrics
        output_dir: Directory to save plots
        loss_columns: List of loss column names to plot
        
    Returns:
        List of saved plot paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []
    
    if loss_columns is None:
        loss_columns = [col for col in df.columns if col.startswith("loss_")]
    
    # Plot total loss
    if "loss_total" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot with better styling
        ax.plot(df["step"], df["loss_total"], linewidth=2, alpha=0.9, color="#2E86AB", label="Total Loss")
        
        # Add smoothed trend line
        if len(df) > 10:
            window = min(max(len(df) // 20, 5), 50)
            smoothed = df["loss_total"].rolling(window=window, center=True).mean()
            ax.plot(df["step"], smoothed, linewidth=3, alpha=0.7, color="#A23B72", linestyle="--", label=f"Smoothed (window={window})")
        
        ax.set_xlabel("Training Step", fontsize=13, fontweight="bold")
        ax.set_ylabel("Loss Value", fontsize=13, fontweight="bold")
        ax.set_title("Training Loss Curve", fontsize=15, fontweight="bold", pad=20)
        ax.legend(loc="best", fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        plot_path = output_dir / "loss_total.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved_plots.append(plot_path)
    
    # Plot loss components
    component_losses = [col for col in loss_columns if col != "loss_total"]
    if component_losses:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Define distinct colors for each component
        colors = ["#E63946", "#F77F00", "#06A77D", "#118AB2", "#073B4C", "#8338EC"]
        
        for idx, loss_col in enumerate(component_losses):
            if loss_col in df.columns:
                color = colors[idx % len(colors)]
                ax.plot(df["step"], df[loss_col], label=loss_col.replace("loss_", ""), 
                       linewidth=2.5, alpha=0.85, color=color)
        
        ax.set_xlabel("Training Step", fontsize=13, fontweight="bold")
        ax.set_ylabel("Loss Value", fontsize=13, fontweight="bold")
        ax.set_title("Loss Components Breakdown", fontsize=15, fontweight="bold", pad=20)
        ax.legend(loc="best", fontsize=11, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        plot_path = output_dir / "loss_components.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved_plots.append(plot_path)
    
    return saved_plots


def plot_learning_rate(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Generate learning rate curve plot.
    
    Args:
        df: DataFrame with training metrics
        output_dir: Directory to save plot
        
    Returns:
        Path to saved plot or None if LR column doesn't exist
    """
    if "lr" not in df.columns:
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot with gradient fill
    ax.plot(df["step"], df["lr"], linewidth=2.5, alpha=0.9, color="#FF6B35", label="Learning Rate")
    ax.fill_between(df["step"], 0, df["lr"], alpha=0.15, color="#FF6B35")
    
    # Add annotations for key points
    max_lr = df["lr"].max()
    min_lr = df["lr"].min()
    max_idx = df["lr"].idxmax()
    
    ax.annotate(f'Max LR: {max_lr:.6f}', 
                xy=(df.loc[max_idx, "step"], max_lr),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'),
                fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Training Step", fontsize=13, fontweight="bold")
    ax.set_ylabel("Learning Rate", fontsize=13, fontweight="bold")
    ax.set_title("Learning Rate Schedule", fontsize=15, fontweight="bold", pad=20)
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    
    plot_path = output_dir / "learning_rate.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return plot_path


def plot_epoch_metrics(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Generate epoch-level metrics plot.
    
    Args:
        df: DataFrame with epoch summaries
        output_dir: Directory to save plot
        
    Returns:
        Path to saved plot or None if no epoch data
    """
    if df is None or df.empty or "epoch_loss" not in df.columns:
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot with markers and connecting lines
    ax.plot(df["epoch"], df["epoch_loss"], marker="o", linewidth=3, markersize=10, 
           alpha=0.9, color="#06A77D", markerfacecolor="#06A77D", 
           markeredgecolor="white", markeredgewidth=2, label="Epoch Loss")
    
    # Add value labels on points
    for idx, row in df.iterrows():
        ax.annotate(f'{row["epoch_loss"]:.2f}', 
                   xy=(row["epoch"], row["epoch_loss"]),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
    
    # Highlight best epoch
    best_idx = df["epoch_loss"].idxmin()
    best_epoch = df.loc[best_idx, "epoch"]
    best_loss = df.loc[best_idx, "epoch_loss"]
    ax.scatter([best_epoch], [best_loss], s=300, c='gold', marker='*', 
              edgecolors='red', linewidths=2, zorder=5, label=f'Best Epoch ({int(best_epoch)})')
    
    ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Loss", fontsize=13, fontweight="bold")
    ax.set_title("Per-Epoch Training Loss", fontsize=15, fontweight="bold", pad=20)
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Set integer ticks for epochs
    ax.set_xticks(df["epoch"].unique())
    
    plot_path = output_dir / "epoch_loss.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return plot_path


def plot_validation_metrics(
    val_metrics: Dict[str, Any],
    output_dir: Path,
) -> List[Path]:
    """Generate validation metrics plots.
    
    Args:
        val_metrics: Dictionary with validation metrics
        output_dir: Directory to save plots
        
    Returns:
        List of saved plot paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []
    
    # Plot per-class AP if available
    if "per_class_ap" in val_metrics and val_metrics["per_class_ap"]:
        class_names = val_metrics.get("class_names", [f"class_{i}" for i in range(len(val_metrics["per_class_ap"]))])
        ap_values = val_metrics["per_class_ap"]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.4)))
        y_pos = np.arange(len(class_names))
        ax.barh(y_pos, ap_values, alpha=0.8, color="tab:blue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Average Precision", fontsize=12)
        ax.set_title("Per-Class AP@0.5", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        
        plot_path = output_dir / "per_class_ap.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(plot_path)
    
    # Plot precision-recall curve if available
    if "precision" in val_metrics and "recall" in val_metrics:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(val_metrics["recall"], val_metrics["precision"], linewidth=2, alpha=0.8)
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plot_path = output_dir / "precision_recall_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(plot_path)
    
    return saved_plots


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_dir: Path,
    normalize: bool = True,
) -> Path:
    """Generate confusion matrix heatmap.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        output_dir: Directory to save plot
        normalize: Whether to normalize the matrix
        
    Returns:
        Path to saved plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if normalize:
        cm = confusion_matrix.astype("float") / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-6)
    else:
        cm = confusion_matrix
    
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(8, len(class_names))))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix (Normalized)" if normalize else "Confusion Matrix",
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    
    plot_path = output_dir / "confusion_matrix.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return plot_path


def generate_metrics_summary(
    train_df: Optional[pd.DataFrame],
    epoch_df: Optional[pd.DataFrame],
    val_metrics: Optional[Dict[str, Any]],
    output_path: Path,
) -> Dict[str, Any]:
    """Generate machine-readable metrics summary.
    
    Args:
        train_df: Training metrics DataFrame
        epoch_df: Epoch summaries DataFrame
        val_metrics: Validation metrics dictionary
        output_path: Path to save summary JSON
        
    Returns:
        Summary dictionary
    """
    summary = {
        "training": {},
        "validation": {},
        "metadata": {},
    }
    
    # Training summary
    if train_df is not None and not train_df.empty:
        loss_cols = [col for col in train_df.columns if col.startswith("loss_")]
        summary["training"]["total_steps"] = int(train_df["step"].max())
        summary["training"]["final_loss"] = float(train_df["loss_total"].iloc[-1]) if "loss_total" in train_df.columns else None
        summary["training"]["min_loss"] = float(train_df["loss_total"].min()) if "loss_total" in train_df.columns else None
        summary["training"]["final_lr"] = float(train_df["lr"].iloc[-1]) if "lr" in train_df.columns else None
        
        # Average loss components
        for col in loss_cols:
            summary["training"][f"avg_{col}"] = float(train_df[col].mean())
    
    # Epoch summary
    if epoch_df is not None and not epoch_df.empty:
        summary["training"]["total_epochs"] = int(epoch_df["epoch"].max())
        summary["training"]["best_epoch"] = int(epoch_df.loc[epoch_df["epoch_loss"].idxmin(), "epoch"])
        summary["training"]["best_epoch_loss"] = float(epoch_df["epoch_loss"].min())
    
    # Validation summary
    if val_metrics is not None:
        summary["validation"] = val_metrics
    
    # Save summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def save_per_class_metrics(
    val_metrics: Dict[str, Any],
    output_path: Path,
) -> Optional[Path]:
    """Save per-class metrics to CSV.
    
    Args:
        val_metrics: Validation metrics dictionary
        output_path: Path to save CSV
        
    Returns:
        Path to saved CSV or None if no per-class metrics
    """
    if "per_class_ap" not in val_metrics or not val_metrics["per_class_ap"]:
        return None
    
    class_names = val_metrics.get("class_names", [f"class_{i}" for i in range(len(val_metrics["per_class_ap"]))])
    ap_values = val_metrics["per_class_ap"]
    
    df = pd.DataFrame({
        "class": class_names,
        "ap50": ap_values,
    })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return output_path


def create_prediction_gallery(
    image_paths: List[Path],
    predictions: List[Dict[str, Any]],
    output_path: Path,
    max_images: int = 16,
    grid_cols: int = 4,
    title: str = "Prediction Gallery",
) -> Path:
    """Create a grid of prediction visualizations.
    
    Args:
        image_paths: List of image file paths
        predictions: List of prediction dictionaries with boxes, labels, scores
        output_path: Path to save gallery image
        max_images: Maximum number of images to include
        grid_cols: Number of columns in grid
        title: Gallery title
        
    Returns:
        Path to saved gallery image
    """
    import cv2
    
    n_images = min(len(image_paths), max_images)
    grid_rows = (n_images + grid_cols - 1) // grid_cols
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 4, grid_rows * 4))
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1 or grid_cols == 1:
        axes = axes.reshape(grid_rows, grid_cols)
    
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    for idx in range(grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        ax = axes[row, col]
        
        if idx < n_images:
            # Load and display image
            img_path = image_paths[idx]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                
                # Add prediction info
                pred = predictions[idx]
                n_det = len(pred.get("boxes", []))
                avg_conf = np.mean(pred.get("scores", [0])) if pred.get("scores") else 0
                ax.set_title(f"{img_path.name}\n{n_det} dets, conf={avg_conf:.2f}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "Failed to load", ha="center", va="center")
                ax.set_title(img_path.name, fontsize=8)
        else:
            ax.axis("off")
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path
