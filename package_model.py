#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.artifacts import (
    ArtifactPaths,
    build_package_manifest,
    extract_class_names,
    gather_environment_info,
    get_git_commit,
    load_artifact_package,
    write_manifest,
)


def _copy_if_exists(src: Optional[Path], dst: Path) -> Optional[str]:
    if not src or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.name)


def package_model(
    weights_path: Path,
    output_dir: Path,
    config_path: Optional[Path],
    data_yaml_path: Optional[Path],
    metrics_summary_path: Optional[Path],
    training_summary_path: Optional[Path],
    onnx_path: Optional[Path],
) -> Path:
    artifact_name = output_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_pt_name = _copy_if_exists(weights_path, output_dir / "model.pt")
    model_onnx_name = _copy_if_exists(onnx_path, output_dir / "model.onnx")
    config_name = _copy_if_exists(config_path, output_dir / "config.yaml")
    data_name = _copy_if_exists(data_yaml_path, output_dir / "data.yaml")
    metrics_name = _copy_if_exists(metrics_summary_path, output_dir / "metrics_summary.json")
    training_name = _copy_if_exists(training_summary_path, output_dir / "training_summary.json")

    class_names = extract_class_names(data_yaml_path)
    class_names_path = None
    if class_names:
        class_names_path = output_dir / "class_names.json"
        class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    env_info = gather_environment_info()
    env_path = output_dir / "environment.txt"
    env_path.write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    git_commit = get_git_commit()
    git_path = output_dir / "git_commit.txt"
    git_path.write_text(git_commit or "unknown", encoding="utf-8")

    config_metadata = None
    if config_path and config_path.exists():
        config_metadata = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_metadata = None
    if data_yaml_path and data_yaml_path.exists():
        data_metadata = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))

    metrics_summary = None
    if metrics_summary_path and metrics_summary_path.exists():
        metrics_summary = json.loads(metrics_summary_path.read_text(encoding="utf-8"))

    training_summary = None
    if training_summary_path and training_summary_path.exists():
        training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))

    manifest = build_package_manifest(
        artifact_name=artifact_name,
        files=ArtifactPaths(
            model_pt=model_pt_name,
            model_onnx=model_onnx_name,
            config_yaml=config_name,
            data_yaml=data_name,
            class_names=str(class_names_path.name) if class_names_path else None,
            metrics_summary=metrics_name,
            training_summary=training_name,
            environment_txt=str(env_path.name),
            git_commit_txt=str(git_path.name),
        ),
        config_metadata=config_metadata,
        data_metadata=data_metadata,
        class_names=class_names,
        metrics_summary=metrics_summary,
        training_summary=training_summary,
        environment_info=env_info,
        git_commit=git_commit,
        source_paths={
            "weights": str(weights_path),
            "config": str(config_path) if config_path else None,
            "data_yaml": str(data_yaml_path) if data_yaml_path else None,
            "metrics_summary": str(metrics_summary_path) if metrics_summary_path else None,
            "training_summary": str(training_summary_path) if training_summary_path else None,
            "onnx": str(onnx_path) if onnx_path else None,
        },
    )

    manifest_path = output_dir / "package_manifest.json"
    write_manifest(manifest_path, manifest)

    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package Detektor model artifacts")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--config", type=str)
    parser.add_argument("--data-yaml", type=str)
    parser.add_argument("--metrics-summary", type=str, default="runs/chimera/metrics_summary.json")
    parser.add_argument("--training-summary", type=str, default="runs/chimera/train_summary.json")
    parser.add_argument("--onnx", type=str, help="Optional ONNX export path")
    parser.add_argument("--output-dir", type=str, help="Artifact output directory", default="artifacts/latest")
    parser.add_argument("--name", type=str, help="Override artifact folder name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    config_path = Path(args.config) if args.config else None
    data_yaml_path = Path(args.data_yaml) if args.data_yaml else None
    metrics_summary_path = Path(args.metrics_summary)
    training_summary_path = Path(args.training_summary)
    onnx_path = Path(args.onnx) if args.onnx else None

    output_dir = Path(args.output_dir)
    if args.name:
        output_dir = output_dir / args.name
    else:
        output_dir = output_dir / weights_path.stem

    manifest_path = package_model(
        weights_path=weights_path,
        output_dir=output_dir,
        config_path=config_path,
        data_yaml_path=data_yaml_path,
        metrics_summary_path=metrics_summary_path,
        training_summary_path=training_summary_path,
        onnx_path=onnx_path,
    )
    print(f"Artifact packaged at {manifest_path}")


if __name__ == "__main__":
    main()
