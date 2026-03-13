from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from models.factory import ARCHITECTURE_PROFILES
from train import train
from utils.architecture_compatibility import collect_compatibility_matrix


def _parse_profiles(raw_profiles: str) -> List[str]:
    if not raw_profiles.strip():
        return list(ARCHITECTURE_PROFILES.keys())
    profiles = [item.strip().lower() for item in raw_profiles.split(",") if item.strip()]
    invalid = [profile for profile in profiles if profile not in ARCHITECTURE_PROFILES]
    if invalid:
        raise ValueError(f"Unknown architecture profiles: {', '.join(invalid)}")
    return profiles


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _print_matrix(matrix: Dict[str, Any]) -> None:
    hardware = matrix["hardware"]
    print("=" * 72)
    print("ARCHITECTURE COMPATIBILITY")
    print("=" * 72)
    print(
        f"device={hardware.get('device')} "
        f"gpu={hardware.get('gpu_name')} "
        f"free_vram_gb={hardware.get('free_vram_gb', 0.0):.2f} "
        f"free_ram_gb={hardware.get('free_ram_gb', 0.0):.2f} "
        f"cpu_count={hardware.get('cpu_count', 0)}"
    )
    print(f"recommended_profile={matrix.get('recommended_profile')}")
    for device_name, entries in matrix.get("results", {}).items():
        print(f"\n[{device_name.upper()}]")
        for entry in entries:
            print(
                f"{entry['profile']:>10} | compatible={entry['compatible']!s:<5} "
                f"| max_batch={entry.get('max_batch_size', 0):>2} "
                f"| compatibility={entry.get('compatibility')} "
                f"| params={entry.get('model_info', {}).get('trainable_params', 0)} "
                f"| step_time_s={entry.get('step_time_s', 0.0):.3f}"
            )


def _run_training_sweep(
    *,
    data_yaml: str,
    profiles: List[str],
    epochs: int,
    base_output_dir: Path,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    sweep_start = time.perf_counter()
    remaining_profiles = len(profiles)

    for profile in profiles:
        run_output_dir = base_output_dir / profile
        print("\n" + "=" * 72)
        print(f"TRAINING PROFILE {profile} ({remaining_profiles} remaining including current)")
        print("=" * 72)
        started_at = time.perf_counter()
        summary = train(
            config_path=None,
            data_yaml=data_yaml,
            overrides={
                "model": {"profile": profile},
                "train": {
                    "epochs": int(epochs),
                    "warmup_epochs": 1,
                    "auto_tune": True,
                    "maximize_batch_size": True,
                    "batch_size_multiple": 4,
                },
                "logging": {"out_dir": str(run_output_dir)},
            },
            run_val=False,
        )
        elapsed_s = time.perf_counter() - started_at
        final_runtime = summary.get("smart_training", {}).get("final_runtime", {})
        runs.append(
            {
                "profile": profile,
                "elapsed_s": round(elapsed_s, 2),
                "elapsed_min": round(elapsed_s / 60.0, 2),
                "out_dir": summary.get("out_dir", ""),
                "best_metric": summary.get("best_metric"),
                "last_epoch_loss": summary.get("last_epoch_loss"),
                "resolved_batch_size": final_runtime.get("batch_size"),
                "resolved_grad_accum": final_runtime.get("grad_accum"),
                "resolved_effective_batch_size": final_runtime.get("effective_batch_size"),
                "resolved_img_size": final_runtime.get("img_size"),
                "resolved_device": final_runtime.get("device"),
                "resolved_model_display_name": final_runtime.get("model_display_name"),
            }
        )
        remaining_profiles -= 1

    total_elapsed_s = time.perf_counter() - sweep_start
    return {
        "epochs_per_profile": int(epochs),
        "profiles_run": profiles,
        "runs": runs,
        "total_elapsed_s": round(total_elapsed_s, 2),
        "total_elapsed_min": round(total_elapsed_s / 60.0, 2),
        "approx_time_line": f"Approximate total wall time for all profiles: {total_elapsed_s / 60.0:.2f} minutes",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List architecture compatibility and optionally benchmark all profiles")
    parser.add_argument("--data-yaml", type=str, default="", help="Dataset YAML used for real training sweeps")
    parser.add_argument("--profiles", type=str, default="", help="Comma-separated architecture profiles to include")
    parser.add_argument("--img-size", type=int, default=512, help="Image size used for compatibility probing")
    parser.add_argument("--batch-multiple", type=int, default=4, help="Batch-size multiple used during probing")
    parser.add_argument("--max-batch-probe", type=int, default=64, help="Maximum CUDA batch size to probe")
    parser.add_argument("--cpu-max-batch-probe", type=int, default=16, help="Maximum CPU batch size to probe")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per profile when running the training sweep")
    parser.add_argument("--run-train-sweep", action="store_true", help="Run real training for each selected profile")
    parser.add_argument("--output-dir", type=str, default="runs/architecture_matrix", help="Output directory for reports and optional sweep runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = _parse_profiles(args.profiles)
    output_dir = Path(args.output_dir)

    matrix = collect_compatibility_matrix(
        data_yaml=args.data_yaml or None,
        profiles=profiles,
        img_size=args.img_size,
        batch_multiple=args.batch_multiple,
        max_batch_probe=args.max_batch_probe,
        cpu_max_batch_probe=args.cpu_max_batch_probe,
    )
    _print_matrix(matrix)
    _write_json(output_dir / "compatibility_matrix.json", matrix)
    print(f"\nCompatibility summary saved to {output_dir / 'compatibility_matrix.json'}")

    if not args.run_train_sweep:
        return
    if not args.data_yaml:
        raise ValueError("--data-yaml is required when --run-train-sweep is used")

    sweep_summary = _run_training_sweep(
        data_yaml=args.data_yaml,
        profiles=profiles,
        epochs=args.epochs,
        base_output_dir=output_dir / "train_runs",
    )
    _write_json(output_dir / "train_sweep_summary.json", sweep_summary)
    print("\n" + sweep_summary["approx_time_line"])
    print(f"Training sweep summary saved to {output_dir / 'train_sweep_summary.json'}")


if __name__ == "__main__":
    main()
