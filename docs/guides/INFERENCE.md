# Inference

## Single Image

**Basic inference:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source image.jpg
```

**With class names:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source image.jpg --data-yaml F:/data/data.yaml
```

**Custom output:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source image.jpg --save-path my_output.jpg
```

## Folder Inference (Batch Processing)

**Process entire folder:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source F:/data/test/images --data-yaml F:/data/data.yaml
```

**Custom output directory:**
```bash
python infer.py --weights runs/chimera/chimera_best.pt --source F:/data/test/images --save-path my_results
```

**Output:**
- Processes all `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` images
- Saves to `runs/inference/` by default
- Files named: `{original_name}_pred.jpg`
- Shows progress: `[1/25] processing: image.jpg`
- Displays detections with class names
