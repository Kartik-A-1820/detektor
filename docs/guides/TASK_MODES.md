# Task Modes

Detektor intelligently supports two task modes:

## 🎯 Detection Mode (`detect`)
- **Bounding box detection only**
- Faster training (no mask loss)
- Lower memory usage
- Returns: boxes, scores, labels

## 🎭 Segmentation Mode (`segment`)
- **Instance segmentation with masks**
- Full mask + box training
- Returns: boxes, scores, labels, masks
- Auto-generates boxes from masks if needed

## Auto-Detection

The system automatically detects your dataset type:

```
==========================================================
TASK DETECTION SUMMARY
==========================================================
Detected task mode: segment
Total label files: 150
Sampled files: 50
  - Bbox format: 0
  - Segment format: 50

Mode: INSTANCE SEGMENTATION
  - Training: Box + mask losses
  - Inference: Returns bounding boxes + masks
  - Boxes auto-generated from masks if needed
==========================================================
```

**Dataset Format Detection:**
- **Segmentation format**: `class_id x1 y1 x2 y2 x3 y3 ...` (polygon points)
- **Detection format**: `class_id x_center y_center width height` (bbox only)

The system samples your dataset and automatically chooses the appropriate mode.
- prepare for later runtime optimization such as TensorRT

The codebase is intentionally lightweight and practical, with a bias toward single-machine workflows and modest GPUs such as the GTX 1650 Ti 4GB rather than distributed training or cloud-scale deployment.
