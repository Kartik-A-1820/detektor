"""Public datasets for real-data benchmarks, downloaded on demand.

Nothing here is redistributed or committed: ``fetch`` downloads a pinned archive to a local cache
(``~/.cache/detektor/datasets`` or ``$DETEKTOR_DATA_DIR``), verifies its SHA-256 and extracts it safely.
Each entry records the dataset's license so you can decide whether it suits your use. Any YOLO-format
dataset of your own works too: pass its ``data.yaml`` to the ``accuracy`` / ``real`` suites instead.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

_BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    sha256: str
    size_mb: float
    description: str
    license: str
    yaml_name: str


DATASETS: Dict[str, DatasetSpec] = {
    "african-wildlife": DatasetSpec(
        "african-wildlife", f"{_BASE}/african-wildlife.zip",
        "6b97a768bd8c89a4ac6f1fd72549a5af805f68568d8846fd0fe68e30c0040061", 105.1,
        "4 classes (buffalo, elephant, rhino, zebra); 1,052 train / 225 val / 227 test photos",
        "AGPL-3.0 (as distributed by Ultralytics)", "african-wildlife.yaml"),
    "brain-tumor": DatasetSpec(
        "brain-tumor", f"{_BASE}/brain-tumor.zip",
        "0710c520add47c5de7d0da78bc206662a0a0b4b26777b582edcb4e6f7660cb0d", 4.4,
        "2 classes (negative, positive); 893 train / 223 val MRI slices",
        "AGPL-3.0 (as distributed by Ultralytics)", "brain-tumor.yaml"),
    "signature": DatasetSpec(
        "signature", f"{_BASE}/signature.zip",
        "6d7bfafd89bcfffb2fdf21f79df16e1988e30f5aa006b0a27d39b51f3bcae171", 11.9,
        "1 class (signature); 143 train / 35 val document scans",
        "AGPL-3.0 (as distributed by Ultralytics)", "signature.yaml"),
}


def cache_dir() -> Path:
    root = os.environ.get("DETEKTOR_DATA_DIR")
    path = Path(root) if root else Path.home() / ".cache" / "detektor" / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a zip, refusing members that would escape ``destination`` (zip-slip)."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe path in archive: {member.filename!r}")
        zf.extractall(destination)


def _download(url: str, out: Path) -> None:
    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0 and sys.stderr.isatty():
            sys.stderr.write(f"\r  {min(blocks * block_size, total) / 1e6:6.1f} / {total / 1e6:.1f} MB")

    urllib.request.urlretrieve(url, out, reporthook=progress)  # noqa: S310 - pinned, checksum-verified URL
    if sys.stderr.isatty():
        sys.stderr.write("\n")


def fetch(name: str, *, url: Optional[str] = None, sha256: Optional[str] = None) -> Path:
    """Ensure dataset ``name`` is available locally and return the path to its YAML.

    ``url`` / ``sha256`` override the registry (used by tests and for mirrors).
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(DATASETS)}")
    spec = DATASETS[name]
    target = cache_dir() / name
    yaml_path = target / spec.yaml_name
    if yaml_path.exists():
        return yaml_path

    expected = sha256 or spec.sha256
    target.mkdir(parents=True, exist_ok=True)
    archive = cache_dir() / f"{name}.zip.part"
    try:
        print(f"Downloading {name} ({spec.size_mb:.1f} MB, {spec.license}) ...", flush=True)
        _download(url or spec.url, archive)
        actual = sha256_of(archive)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {name}: expected {expected}, got {actual}")
        safe_extract(archive, target)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        archive.unlink(missing_ok=True)
    if not yaml_path.exists():
        raise RuntimeError(f"{name}: archive did not contain {spec.yaml_name}")
    return yaml_path


def make_subset(data_yaml: Path, out_dir: Path, *, max_train: Optional[int] = None, max_val: Optional[int] = None,
                seed: int = 0) -> Path:
    """Create a smaller dataset (deterministic random subset) next to ``out_dir``; returns its YAML.

    Handles both YOLO layouts. Files are copied (not symlinked) so it also works on Windows.
    """
    import random

    from utils.data_config import load_dataset_yaml, resolve_split_dirs

    info = load_dataset_yaml(str(data_yaml))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    payload: Dict[str, object] = {"names": info["class_names"], "nc": info["num_classes"]}
    for split, limit in (("train", max_train), ("val", max_val)):
        root = info.get(split)
        if not root:
            continue
        images_dir, labels_dir = resolve_split_dirs(root)
        images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        if limit and len(images) > limit:
            images = sorted(rng.sample(images, limit))
        new_images, new_labels = out_dir / split / "images", out_dir / split / "labels"
        new_images.mkdir(parents=True, exist_ok=True)
        new_labels.mkdir(parents=True, exist_ok=True)
        for image in images:
            shutil.copy2(image, new_images / image.name)
            label = labels_dir / f"{image.stem}.txt"
            if label.exists():
                shutil.copy2(label, new_labels / label.name)
        payload[split] = str(new_images.resolve())
    out_yaml = out_dir / "data.yaml"
    out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out_yaml
