"""Artifact packaging helpers for Detektor."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:  # torch is optional for metadata gathering when running tests
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch may be unavailable in some environments
    torch = None

MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass
class ArtifactPaths:
    """Resolved file names inside an artifact package."""

    model_pt: str
    model_onnx: Optional[str]
    config_yaml: Optional[str]
    data_yaml: Optional[str]
    class_names: Optional[str]
    metrics_summary: Optional[str]
    training_summary: Optional[str]
    environment_txt: str
    git_commit_txt: str


def _safe_read_yaml(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def extract_class_names(data_yaml_path: Optional[Path]) -> List[str]:
    """Extract class names from a YOLO-style dataset YAML file."""
    if not data_yaml_path or not data_yaml_path.exists():
        return []
    data_yaml = _safe_read_yaml(data_yaml_path)
    names = data_yaml.get("names")
    if isinstance(names, dict):
        # Roboflow exports may use dict mapping index->name
        names = [name for _, name in sorted(names.items())]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def gather_environment_info() -> Dict[str, Any]:
    """Capture key environment details for reproducibility."""
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "interpreter": sys.executable,
    }
    if torch is not None:
        info["torch_version"] = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda_available
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if cuda_available:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    else:  # pragma: no cover - torch unavailable
        info["torch_version"] = None
        info["cuda_available"] = False
        info["cuda_version"] = None
    return info


def get_git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Return the current git commit hash if inside a repo."""
    root = repo_root or Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover - git missing
        return None


def build_package_manifest(
    artifact_name: str,
    files: ArtifactPaths,
    config_metadata: Optional[Dict[str, Any]],
    data_metadata: Optional[Dict[str, Any]],
    class_names: Optional[List[str]],
    metrics_summary: Optional[Dict[str, Any]],
    training_summary: Optional[Dict[str, Any]],
    environment_info: Dict[str, Any],
    git_commit: Optional[str],
    source_paths: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Construct a manifest describing the packaged artifact."""
    created_at = datetime.now(timezone.utc).isoformat()
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_name": artifact_name,
        "created_at": created_at,
        "files": {
            "model_pt": files.model_pt,
            "model_onnx": files.model_onnx,
            "config_yaml": files.config_yaml,
            "data_yaml": files.data_yaml,
            "class_names": files.class_names,
            "metrics_summary": files.metrics_summary,
            "training_summary": files.training_summary,
            "environment_txt": files.environment_txt,
            "git_commit_txt": files.git_commit_txt,
        },
        "metadata": {
            "config": config_metadata,
            "data": data_metadata,
            "class_names": class_names,
            "metrics_summary": metrics_summary,
            "training_summary": training_summary,
        },
        "environment": environment_info,
        "git": {"commit": git_commit} if git_commit else {},
        "source_paths": source_paths,
    }
    return manifest


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def load_artifact_package(package_dir: str | Path) -> Dict[str, Any]:
    """Load a packaged artifact manifest and resolve file paths."""
    package_path = Path(package_dir)
    manifest_path = package_path / "package_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    files = manifest.get("files", {})
    resolved_files: Dict[str, Optional[str]] = {}
    for key, relative_path in files.items():
        if relative_path is None:
            resolved_files[key] = None
            continue
        resolved_path = package_path / relative_path
        resolved_files[key] = str(resolved_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Expected file {relative_path} missing inside artifact")

    manifest["resolved_files"] = resolved_files
    return manifest


__all__ = [
    "ArtifactPaths",
    "MANIFEST_SCHEMA_VERSION",
    "build_package_manifest",
    "extract_class_names",
    "gather_environment_info",
    "get_git_commit",
    "load_artifact_package",
    "write_manifest",
]
