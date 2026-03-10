# Chimera-ODIS (GTX 1650 Ti Optimized)

Efficient **Object Detection + Instance Segmentation** training/inference scaffold designed to run on a **single consumer GPU (GTX 1650 Ti / 4GB VRAM)** with best-effort VRAM capping (~80%).

> **Status**
>
> This repo is a **fast, clean baseline scaffold** (dataset loaders, training loop, VRAM guardrails, utilities, visualization).
> The model/loss in this scaffold is intentionally minimal so it can run everywhere.
> If you want full production-grade detection+seg losses/decoding/mAP, extend `models/chimera.py` and add metric code in `validate.py`.

---

## 1) System prerequisites

### Linux (recommended)
- NVIDIA GPU + driver installed
- CUDA toolkit is **not** strictly required if you install the correct PyTorch wheel (it bundles CUDA libs).
- Python 3.10+ recommended.

### Windows
Works as well (PowerShell). You must have a working NVIDIA driver.

---

## 2) Setup with `uv` (fast Python packaging)

### Install uv
**Linux/macOS**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Verify:
```bash
uv --version
```

---

## 3) Create a venv and install CUDA-enabled PyTorch

### Option A (Recommended): Install PyTorch CUDA wheels first, then install project deps
This avoids CPU-only PyTorch installs on some systems.

1) Create venv:
```bash
uv venv --python 3.10
```

2) Activate:
```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

3) Install CUDA PyTorch (pick one)

#### CUDA 12.1 wheels (common)
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

#### CUDA 11.8 wheels (older drivers)
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
```

4) Install remaining deps:
```bash
uv pip install -r requirements.txt
```

5) Quick CUDA check:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

> If `CUDA: False`, your NVIDIA driver / wheel selection likely mismatched. Try cu118 vs cu121.

---

### Option B: Install all deps in one go
If your environment reliably resolves CUDA wheels:
```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```
Then verify CUDA with the same command above.

---

## 4) Dataset formats supported

### A) YOLOv8-seg format (recommended for this scaffold)
```
data/train/
  images/*.jpg
  labels/*.txt   # polygon labels (YOLOv8 seg)
data/val/
  images/*.jpg
  labels/*.txt
```

Label line format:
```
cls x1 y1 x2 y2 ... xN yN   # normalized polygon coordinates
```

Update config:
- `data.format: "yolo"`
- `data.train: "data/train"`
- `data.val: "data/val"`

---

### B) COCO JSON segmentation
If you have COCO `instances_*.json`:
- Set:
  - `data.format: "coco"`
  - `data.coco_train_json`, `data.coco_val_json`
  - `data.images_dir_train`, `data.images_dir_val`

---

## 5) Run training / validation / inference

### Train
```bash
python train.py --config configs/chimera_s_512.yaml
```

### Validate
```bash
python validate.py --config configs/chimera_s_512.yaml
```

### Inference
```bash
python infer.py --weights chimera_last.pt --source path/to/image.jpg
```

---

## 6) VRAM usage (80% cap)

This project attempts to limit VRAM growth:
- Config: `train.vram_cap: 0.80`
- Code: `utils/vram.py -> set_vram_cap()`

Notes:
- This is a **best-effort** cap (allocator behavior can vary).
- If you hit OOM:
  - reduce `train.batch_size`
  - reduce `train.img_size` (try 512)
  - reduce `train.max_instances_per_img`

---

## 7) Useful utilities

### Plot a dataset sample with boxes/masks
Use:
- `utils/visualize.py`
- `plot_annotated_sample(img_tensor, target, save_path=...)`

---

## 8) Troubleshooting

### OpenCV import errors
Try reinstall:
```bash
uv pip uninstall opencv-python -y
uv pip install opencv-python
```

### pycocotools build issues (Windows)
If `pycocotools` fails on Windows, try:
```bash
uv pip install pycocotools-windows
```
and update `requirements.txt` accordingly.

---

## License
Choose your preferred open-source license (MIT/Apache-2.0) and add `LICENSE`.
