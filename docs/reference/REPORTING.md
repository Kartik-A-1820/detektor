# Reporting Module Documentation

## Overview

The Detektor reporting module provides Ultralytics-style training and validation artifacts generation. It reads existing training logs and generates comprehensive plots, summaries, and machine-readable metrics.

## Features

### Training Plots
- **Loss Curves**: Total loss and individual components (cls, box, obj, mask)
- **Learning Rate Schedule**: Visualization of LR changes during training
- **Epoch Metrics**: Per-epoch average loss tracking

### Validation Plots
- **Per-Class AP**: Bar chart showing Average Precision for each class
- **Precision-Recall Curve**: Model performance across confidence thresholds
- **Confusion Matrix**: Normalized heatmap showing classification performance

### Machine-Readable Reports
- **Metrics Summary JSON**: Comprehensive training and validation statistics
- **Per-Class Metrics CSV**: Tabular format for easy analysis
- **Report Status JSON**: Generation status with warnings and file paths

### Prediction Gallery
- **Visual Grid**: Sample predictions for qualitative analysis
- **Configurable Layout**: Customizable grid size and image count

## Usage

### Basic Command
```bash
python -m scripts.report --run-dir runs/chimera
```

### Advanced Options
```bash
python -m scripts.report \
  --run-dir runs/chimera \
  --plots-dir custom_plots \
  --reports-dir custom_reports \
  --verbose
```

## Input Files

### Required
- `train_metrics.csv` or `train_metrics.jsonl` - Per-step training metrics
- `epoch_summaries.jsonl` - Per-epoch summaries

### Optional
- `val_metrics.json` - Validation results for validation plots

## Output Structure

```
runs/chimera/
├── plots/
│   ├── loss_total.png
│   ├── loss_components.png
│   ├── learning_rate.png
│   ├── epoch_loss.png
│   ├── per_class_ap.png           # if validation data exists
│   ├── precision_recall_curve.png # if validation data exists
│   └── confusion_matrix.png       # if validation data exists
└── reports/
    ├── metrics_summary.json
    ├── per_class_metrics.csv      # if validation data exists
    └── report_status.json
```

## Graceful Degradation

The reporting module handles missing data gracefully:

- ✅ Generates available plots even if some data is missing
- ✅ Logs warnings for missing files
- ✅ Continues execution on individual plot failures
- ✅ Provides detailed status in `report_status.json`

## Integration with Training

The module is designed to work seamlessly with existing training workflows:

```bash
# 1. Train your model
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml

# 2. Generate report
python -m scripts.report --run-dir runs/chimera
```

## API Usage

You can also use the reporting utilities programmatically:

```python
from pathlib import Path
from utils.reporting import (
    load_train_metrics,
    plot_loss_curves,
    generate_metrics_summary,
)

# Load metrics
train_df = load_train_metrics(Path("runs/chimera/train_metrics.csv"))

# Generate plots
plot_loss_curves(train_df, Path("runs/chimera/plots"))

# Generate summary
summary = generate_metrics_summary(
    train_df,
    epoch_df,
    val_metrics,
    Path("runs/chimera/reports/metrics_summary.json"),
)
```

## Testing

Run the reporting module tests:

```bash
python -m unittest tests.test_reporting
```

## Dependencies

The reporting module uses only lightweight dependencies:
- `matplotlib` - Plot generation
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `opencv-python` (cv2) - Image loading for galleries

All dependencies are already included in `requirements.txt`.

## Backward Compatibility

The reporting module:
- ✅ Does NOT modify existing training/validation/inference APIs
- ✅ Works with existing log formats
- ✅ Is completely optional (training works without it)
- ✅ Can be run at any time after training completes

## Examples

### Example 1: Training Report
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml
python -m scripts.report --run-dir runs/chimera
```

### Example 2: Custom Output Directories
```bash
python -m scripts.report \
  --run-dir runs/chimera \
  --plots-dir analysis/plots \
  --reports-dir analysis/reports
```

### Example 3: Verbose Debugging
```bash
python -m scripts.report --run-dir runs/chimera --verbose
```

## Troubleshooting

### No plots generated
- Check that `train_metrics.csv` or `train_metrics.jsonl` exists
- Verify the run directory path is correct
- Use `--verbose` flag to see detailed error messages

### Missing validation plots
- Validation plots require `val_metrics.json` in the run directory
- This is optional - training plots will still be generated

### Plot generation errors
- Check `reports/report_status.json` for detailed warnings
- Individual plot failures don't stop the entire report generation
- Verify matplotlib is installed: `pip install matplotlib`

## Future Enhancements

Potential additions (not yet implemented):
- Real-time prediction galleries from validation set
- Interactive HTML reports
- Tensorboard integration
- Comparison reports across multiple runs
- Automated failure case analysis

## License

Same as the main Detektor project (MIT License).


