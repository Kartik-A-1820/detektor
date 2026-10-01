# Validation

## Production-Grade Validation

Run comprehensive validation with detailed metrics and artifacts:

**Basic validation:**
```bash
python validate.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml
```

**With image saving:**
```bash
python validate.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml --save-images --max-images 50
```

**With AP50-95 (COCO-style):**
```bash
python validate.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml --compute-ap50-95
```

**Custom output directory:**
```bash
python validate.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml --output-dir my_validation
```

## Validation Metrics

The production-grade validation computes:

**Detection Metrics:**
- Precision, Recall, F1 score
- AP50 (Average Precision at IoU 0.5)
- mAP50 (Mean AP across all classes)
- AP50-95 (COCO-style, optional)
- Mean box IoU
- Per-class precision, recall, F1, AP50

**Segmentation Metrics:**
- Mean mask IoU
- Mean Dice coefficient
- Per-instance mask quality

**Analysis Tools:**
- Confusion matrix
- Confidence threshold sweep
- Precision-Recall curves
- Per-class breakdown

## Validation Outputs

```
runs/validate/<run_name>/
├── metrics.json              # Comprehensive metrics
├── per_class_metrics.csv     # Per-class performance
├── confusion_matrix.csv      # Confusion matrix
├── threshold_sweep.csv       # Threshold analysis
└── images/                   # Annotated images (optional)
```

## Metrics Summary Example

```json
{
  "overall": {
    "precision": 0.8542,
    "recall": 0.7891,
    "f1": 0.8203,
    "ap50": 0.8234,
    "map50": 0.8123,
    "mean_box_iou": 0.7456,
    "mean_mask_iou": 0.6789
  },
  "per_class": [
    {
      "class_name": "ball",
      "precision": 0.92,
      "recall": 0.85,
      "f1": 0.88,
      "ap50": 0.89
    }
  ],
  "threshold_sweep": {
    "best_threshold": 0.4,
    "best_f1": 0.8203
  }
}
```

## Validation Options

- `--config`: Path to config YAML (required)
- `--weights`: Path to model weights (required)
- `--data-yaml`: Dataset YAML for class names
- `--batch-size`: Validation batch size (default: 4)
- `--conf-thresh`: Confidence threshold (default: 0.25)
- `--iou-thresh`: IoU threshold for matching (default: 0.5)
- `--output-dir`: Custom output directory
- `--save-images`: Save annotated validation images
- `--max-images`: Max images to save (default: 20)
- `--compute-ap50-95`: Compute COCO-style AP (slower)

## Edge Cases Handled

✅ **Empty predictions** - Gracefully handles no detections  
✅ **Empty ground truth** - Handles images without annotations  
✅ **Missing classes** - Classes absent in validation split  
✅ **Segmentation disabled** - Works without masks  
✅ **NaN/Inf values** - Sanitizes invalid predictions  
✅ **Memory efficient** - Optimized for GTX 1650 Ti 4GB  

## Integration with Reporting

Validation outputs integrate seamlessly with the reporting module:

```bash
# Run validation
python validate.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --data-yaml F:/data/data.yaml

# Generate visual report
python -m scripts.report --run-dir runs/validate/chimera_best
```

See `docs/reference/VALIDATION_OUTPUT_SCHEMA.md` for detailed output format documentation.



## Reporting

Detektor includes a comprehensive reporting module that **automatically generates** training visualizations and summaries during training.

### Automatic Integration

**Reports are generated automatically:**
- ✅ **After each epoch** - Loss curves, LR plot, epoch metrics updated
- ✅ **At end of training** - Comprehensive final report with all plots and summary
- ✅ **No separate command needed** - Everything happens during training

**Output location:** `runs/chimera/plots/`

### Features

- **Loss Curves**: Track all loss components over training
- **Learning Rate Schedule**: Visualize LR changes across epochs
- **Epoch Metrics**: Monitor training progress
- **Training Summary**: Detailed statistics and configuration info
- **Graceful Degradation**: Handles missing data elegantly

### Manual Report Generation (Optional)

If you need to regenerate reports manually:

```bash
python -m scripts.report --run-dir runs/chimera
```

**Specify custom output directory:**
```bash
python -m scripts.report --run-dir runs/chimera --output-dir my_reports
```

**Verbose output:**
```bash
python -m scripts.report --run-dir runs/chimera --verbose
```

### Generated Artifacts

**Plots (`runs/chimera/plots/`):**
- `loss_total.png` - Total training loss curve
- `loss_components.png` - Individual loss components (cls, box, obj, mask)
- `learning_rate.png` - Learning rate schedule
- `epoch_loss.png` - Per-epoch average loss
- `per_class_ap.png` - Per-class Average Precision bar chart (if validation data exists)
- `precision_recall_curve.png` - Precision-Recall curve (if validation data exists)
- `confusion_matrix.png` - Normalized confusion matrix heatmap (if validation data exists)

**Reports (`runs/chimera/reports/`):**
- `metrics_summary.json` - Machine-readable training and validation summary
- `per_class_metrics.csv` - Per-class AP metrics in CSV format
- `report_status.json` - Report generation status and warnings

### Metrics Summary Example

```json
{
  "training": {
    "total_steps": 190,
    "total_epochs": 5,
    "final_loss": 9.395267,
    "min_loss": 8.872341,
    "best_epoch": 5,
    "best_epoch_loss": 9.395267,
    "final_lr": 0.001,
    "avg_loss_cls": 2.456,
    "avg_loss_box": 3.123,
    "avg_loss_obj": 1.987,
    "avg_loss_mask": 1.829
  },
  "validation": {
    "map50": 0.8575,
    "per_class_ap": [0.85, 0.92, 0.78, 0.88],
    "class_names": ["ball", "goalkeeper", "player", "referee"]
  }
}
```

### Required Input Files

The report generator reads from your training run directory:

**Required:**
- `train_metrics.csv` or `train_metrics.jsonl` - Per-step training metrics
- `epoch_summaries.jsonl` - Per-epoch summaries

**Optional:**
- `val_metrics.json` - Validation results (for validation plots)

### Graceful Degradation

The report generator handles missing data gracefully:
- If validation metrics are missing, only training plots are generated
- If learning rate is not logged, LR plot is skipped
- Warnings are logged for missing files
- Report generation continues even if some plots fail

### Integration with Training

Reports are automatically compatible with the training output format. After training:

```bash
## Train your model
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml

## Generate comprehensive report
python -m scripts.report --run-dir runs/chimera
```

All plots and reports will be saved in the run directory for easy access and version control.
