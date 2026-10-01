# Dataset Validation

Before training, validate your dataset to catch common issues:

```bash
python check_dataset.py --data-yaml F:/data/data.yaml
```

## What It Checks

**File Integrity:**
- Missing image files
- Missing label files
- Corrupt or unreadable images
- Empty label files

**Label Format:**
- Malformed YOLO label rows (must have at least 5 values)
- Invalid class IDs (out of range)
- Invalid normalized coordinates (must be in [0, 1])
- Zero or negative box dimensions

**Dataset Quality:**
- Class distribution across dataset
- Image size distribution
- Duplicate filenames (potential data issues)

## Validation Output

The tool generates two report files in `reports/`:

**JSON Summary (`dataset_check.json`):**
```json
{
  "has_errors": false,
  "has_warnings": true,
  "total_issues": 3,
  "error_count": 0,
  "warning_count": 3,
  "stats": {
    "total_images": 150,
    "total_labels": 148,
    "total_annotations": 892,
    "empty_labels": 2,
    "corrupt_images": 0,
    "class_distribution": {
      "0": 234,
      "1": 312,
      "2": 198,
      "3": 148
    },
    "image_size_distribution": {
      "640x480": 120,
      "1280x720": 30
    },
    "duplicate_filenames": []
  }
}
```

**CSV Issues (`dataset_check.csv`):**
```csv
severity,category,message,file_path,line_number
warning,empty_label,Label file is empty,F:/data/train/labels/img_042.txt,
error,invalid_class_id,Class ID 5 out of range [0, 3],F:/data/train/labels/img_089.txt,3
```

## Exit Codes

- **Exit 0**: Validation passed (warnings allowed)
- **Exit 1**: Validation failed (errors found)

Use in CI/CD or pre-training hooks:

```bash
python check_dataset.py --data-yaml F:/data/data.yaml || exit 1
python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml
```

## Validation Options

- `--data-yaml`: Dataset YAML file (required)
- `--output-dir`: Report output directory (default: `reports`)
- `--splits`: Dataset splits to validate (default: `train val`)

## Inference Options

- `--weights`: Model checkpoint path (required)
- `--source`: Image file or folder (required)
- `--data-yaml`: Dataset YAML for class names (optional)
- `--num-classes`: Override auto-detection (optional)
- `--img-size`: Input size (default: the size the checkpoint was trained at, else 512)
- `--conf-thresh`: Confidence threshold (default: 0.25)
- `--iou-thresh`: NMS IoU threshold (default: 0.6)
- `--max-det`: Max detections per image (default: 100)
- `--save-path`: Output file or folder (optional)

## Auto-Detection Features

**num_classes:** Automatically detected from checkpoint
```
auto-detected num_classes=4 from checkpoint
```

**Class names:** Loaded from `--data-yaml` if provided
```
loaded class names: ['ball', 'goalkeeper', 'player', 'referee']
detections: 24, labels: ['player', 'player', 'goalkeeper', ...]
```



## Dataset Format

### YOLO/Roboflow Format

Detektor supports standard YOLO-style datasets exported from Roboflow or similar tools.

**Dataset YAML (`data.yaml`):**
```yaml
train: F:/data/train/images
val: F:/data/valid/images
test: F:/data/test/images
nc: 4
names: ['ball', 'goalkeeper', 'player', 'referee']
```

**Directory Structure:**
```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Label Format

YOLO format: `class x_center y_center width height` (normalized 0-1)

**Example `labels/image1.txt`:**
```
0 0.5 0.5 0.3 0.4
2 0.7 0.3 0.2 0.3
```

### Auto-Configuration

When `--data-yaml` is provided, Detektor automatically:

✅ Resolves train/val paths  
✅ Converts `images/` paths to split roots  
✅ Sets `data.format = "yolo"`  
✅ Updates `data.num_classes` from `nc`  
✅ Loads class names from `names`  
✅ Validates directory structure  

### Supported Image Formats

- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.webp`

### Creating Your Dataset

1. **Export from Roboflow:**
   - Choose "YOLOv8" format
   - Download and extract

2. **Verify structure:**
   ```bash
   ls F:/data/train/images  # Should show images
   ls F:/data/train/labels  # Should show .txt files
   ```

3. **Check data.yaml:**
   ```bash
   cat F:/data/data.yaml
   ```

4. **Train:**
   ```bash
   python train.py --config configs/chimera_s_512.yaml --data-yaml F:/data/data.yaml
   ```
