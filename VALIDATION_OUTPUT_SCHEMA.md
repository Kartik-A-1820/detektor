# Validation Output Schema Documentation

## Overview

The production-grade validation pipeline generates comprehensive metrics and artifacts for evaluating model performance on detection and instance segmentation tasks.

## Output Directory Structure

```
runs/validate/<run_name>/
├── metrics.json              # Comprehensive metrics summary
├── per_class_metrics.csv     # Per-class performance metrics
├── confusion_matrix.csv      # Confusion matrix
├── threshold_sweep.csv       # Confidence threshold analysis
└── images/                   # Annotated validation images (optional)
    ├── val_0_0.jpg
    ├── val_0_1.jpg
    └── ...
```

## File Formats

### metrics.json

Complete validation results in JSON format.

**Structure:**
```json
{
  "overall": {
    "precision": 0.8542,
    "recall": 0.7891,
    "f1": 0.8203,
    "ap50": 0.8234,
    "map50": 0.8123,
    "ap50_95": 0.6234,
    "mean_box_iou": 0.7456,
    "mean_mask_iou": 0.6789,
    "mean_dice": 0.7123,
    "total_tp": 1234,
    "total_fp": 211,
    "total_fn": 329,
    "num_images": 100
  },
  "per_class": [
    {
      "class_id": 0,
      "class_name": "ball",
      "precision": 0.92,
      "recall": 0.85,
      "f1": 0.88,
      "ap50": 0.89,
      "tp": 85,
      "fp": 7,
      "fn": 15
    },
    ...
  ],
  "threshold_sweep": {
    "sweep": [
      {
        "threshold": 0.1,
        "precision": 0.65,
        "recall": 0.92,
        "f1": 0.76,
        "tp": 920,
        "fp": 495,
        "fn": 80
      },
      ...
    ],
    "best_threshold": 0.4,
    "best_f1": 0.8203,
    "best_precision": 0.8542,
    "best_recall": 0.7891
  },
  "class_names": ["ball", "goalkeeper", "player", "referee"],
  "confusion_matrix": [
    [50, 2, 3, 1, 0],
    [1, 45, 2, 0, 0],
    [2, 1, 48, 3, 1],
    [0, 1, 2, 47, 0],
    [5, 3, 8, 2, 0]
  ],
  "precision_recall_curve": {
    "precision": [1.0, 0.98, 0.95, ...],
    "recall": [0.05, 0.12, 0.23, ...]
  }
}
```

**Field Descriptions:**

#### overall
- `precision`: True positives / (True positives + False positives)
- `recall`: True positives / (True positives + False negatives)
- `f1`: Harmonic mean of precision and recall
- `ap50`: Average Precision at IoU threshold 0.5
- `map50`: Mean Average Precision across all classes at IoU 0.5
- `ap50_95`: Average Precision averaged over IoU thresholds 0.5:0.05:0.95 (COCO-style)
- `mean_box_iou`: Average IoU of matched bounding boxes
- `mean_mask_iou`: Average IoU of matched instance masks
- `mean_dice`: Average Dice coefficient of matched masks
- `total_tp`: Total true positive detections
- `total_fp`: Total false positive detections
- `total_fn`: Total false negative (missed) detections
- `num_images`: Number of validation images processed

#### per_class
Array of per-class metrics, one entry per class:
- `class_id`: Integer class identifier
- `class_name`: Human-readable class name
- `precision`: Class-specific precision
- `recall`: Class-specific recall
- `f1`: Class-specific F1 score
- `ap50`: Class-specific Average Precision at IoU 0.5
- `tp`: Class-specific true positives
- `fp`: Class-specific false positives
- `fn`: Class-specific false negatives

#### threshold_sweep
Analysis of performance across confidence thresholds:
- `sweep`: Array of metrics at different thresholds
  - `threshold`: Confidence threshold value
  - `precision`, `recall`, `f1`: Metrics at this threshold
  - `tp`, `fp`, `fn`: Counts at this threshold
- `best_threshold`: Threshold that maximizes F1 score
- `best_f1`: F1 score at best threshold
- `best_precision`: Precision at best threshold
- `best_recall`: Recall at best threshold

#### confusion_matrix
2D array of shape `[num_classes+1, num_classes+1]`:
- Rows represent ground truth classes
- Columns represent predicted classes
- Last row/column represents background (false positives/negatives)
- Entry `[i, j]` is the count of GT class `i` predicted as class `j`

#### precision_recall_curve
- `precision`: List of precision values at different recall levels
- `recall`: List of recall values (sorted ascending)

### per_class_metrics.csv

Tabular format for easy analysis in spreadsheet tools.

**Columns:**
- `class_id`: Integer class identifier
- `class_name`: Class name
- `precision`: Precision score
- `recall`: Recall score
- `f1`: F1 score
- `ap50`: Average Precision at IoU 0.5
- `tp`: True positives
- `fp`: False positives
- `fn`: False negatives

**Example:**
```csv
class_id,class_name,precision,recall,f1,ap50,tp,fp,fn
0,ball,0.92,0.85,0.88,0.89,85,7,15
1,goalkeeper,0.88,0.91,0.89,0.90,91,12,9
2,player,0.84,0.76,0.80,0.78,152,29,48
3,referee,0.90,0.94,0.92,0.93,47,5,3
```

### confusion_matrix.csv

Confusion matrix in CSV format with row/column headers.

**Format:**
- Rows: Ground truth classes + background
- Columns: Predicted classes + background

**Example:**
```csv
,ball,goalkeeper,player,referee,background
ball,50,2,3,1,0
goalkeeper,1,45,2,0,0
player,2,1,48,3,1
referee,0,1,2,47,0
background,5,3,8,2,0
```

### threshold_sweep.csv

Confidence threshold analysis in tabular format.

**Columns:**
- `threshold`: Confidence threshold
- `precision`: Precision at this threshold
- `recall`: Recall at this threshold
- `f1`: F1 score at this threshold
- `tp`: True positives at this threshold
- `fp`: False positives at this threshold
- `fn`: False negatives at this threshold

**Example:**
```csv
threshold,precision,recall,f1,tp,fp,fn
0.1,0.65,0.92,0.76,920,495,80
0.2,0.72,0.88,0.79,880,342,120
0.3,0.79,0.84,0.81,840,223,160
0.4,0.85,0.79,0.82,790,139,210
0.5,0.89,0.72,0.80,720,89,280
```

## Usage Examples

### Loading Results in Python

```python
import json
import pandas as pd

# Load comprehensive metrics
with open("runs/validate/chimera_best/metrics.json") as f:
    metrics = json.load(f)

print(f"Overall mAP50: {metrics['overall']['map50']:.4f}")
print(f"Best threshold: {metrics['threshold_sweep']['best_threshold']}")

# Load per-class metrics
df_per_class = pd.read_csv("runs/validate/chimera_best/per_class_metrics.csv")
print(df_per_class)

# Load confusion matrix
df_cm = pd.read_csv("runs/validate/chimera_best/confusion_matrix.csv", index_col=0)
print(df_cm)
```

### Analyzing Results

```python
# Find worst performing class
worst_class = min(metrics['per_class'], key=lambda x: x['f1'])
print(f"Worst class: {worst_class['class_name']} (F1={worst_class['f1']:.4f})")

# Find optimal threshold
best_thresh = metrics['threshold_sweep']['best_threshold']
print(f"Use confidence threshold: {best_thresh}")

# Analyze confusion
import numpy as np
cm = np.array(metrics['confusion_matrix'])
# Most confused pairs
for i in range(cm.shape[0]-1):
    for j in range(cm.shape[1]-1):
        if i != j and cm[i, j] > 5:
            print(f"{metrics['class_names'][i]} confused with {metrics['class_names'][j]}: {cm[i, j]} times")
```

## Edge Cases Handled

The validation pipeline gracefully handles:

1. **Empty Predictions**: When model produces no detections
2. **Empty Ground Truth**: When validation images have no annotations
3. **Missing Classes**: Classes present in config but absent in validation split
4. **Segmentation Disabled**: When masks are not available or disabled
5. **NaN/Inf Values**: Invalid predictions are sanitized
6. **Memory Constraints**: Batch processing for GTX 1650 Ti 4GB compatibility

## Performance Considerations

- **AP50-95 Computation**: Optional (use `--compute-ap50-95` flag) as it's slower
- **Image Saving**: Limited to `--max-images` to save disk space
- **Batch Size**: Adjust `--batch-size` based on available VRAM
- **Memory Usage**: Results are computed incrementally to minimize memory footprint

## Integration with Reporting

Validation outputs are compatible with the reporting module:

```bash
# Run validation
python validate_v2.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml

# Generate report (will include validation metrics if val_metrics.json exists)
python report.py --run-dir runs/validate/chimera_best
```

The reporting module will automatically detect and visualize validation metrics including confusion matrices, per-class AP charts, and precision-recall curves.
