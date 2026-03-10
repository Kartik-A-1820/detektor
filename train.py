from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_dataset
from engine.ema import ModelEMA
from models.chimera import ChimeraODIS
from utils.checkpoints import load_checkpoint, save_checkpoint
from utils.collate import detection_segmentation_collate_fn
from utils.data_config import apply_dataset_yaml_overrides, print_resolved_dataset_config
from utils.logging_utils import append_jsonl, append_metrics_row, write_json
from utils.model_info import get_model_info
from utils.repro import set_seed
from utils.vram import set_vram_cap, vram_report


def _resolve_device(device_name: str) -> torch.device:
    """Resolve the requested training device with a clear CUDA error when unavailable."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested in the config, but CUDA is not available on this machine")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _load_weights_if_provided(model: ChimeraODIS, weights: str | None, device: torch.device) -> None:
    """Load model weights from either a plain state dict or a checkpoint payload."""
    if not weights:
        return
    checkpoint = torch.load(weights, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)


def train(
    config_path: str,
    data_yaml: str | None = None,
    resume: str | None = None,
    weights: str | None = None,
    use_ema: bool = False,
    grad_clip_norm: float = 0.0,
    debug_loss: bool = False,
) -> Dict[str, Any]:
    """Run the production-hardened training loop with safe resume and logging helpers."""
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if data_yaml:
        resolved_dataset = apply_dataset_yaml_overrides(cfg, data_yaml)
        print_resolved_dataset_config(resolved_dataset)

    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
    set_vram_cap(cfg["train"].get("vram_cap", 0.8))
    device = _resolve_device(cfg.get("device", "cpu"))

    model = ChimeraODIS(
        num_classes=cfg["data"]["num_classes"],
        proto_k=cfg["model"]["proto_k"],
    ).to(device)
    model_info = get_model_info(model)
    if model_info["trainable_params"] == 0:
        print("warning: model has no trainable parameters")

    _load_weights_if_provided(model, weights, device)

    dataset = build_dataset(cfg, split="train")
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty")

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_segmentation_collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    amp_enabled = bool(cfg["train"].get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = ModelEMA(model, decay=float(cfg["train"].get("ema_decay", 0.9998))) if use_ema else None

    out_dir = Path(cfg.get("logging", {}).get("out_dir", "runs/chimera"))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = out_dir / "train_metrics.csv"
    metrics_jsonl_path = out_dir / "train_metrics.jsonl"
    summary_json_path = out_dir / "run_summary.json"
    last_checkpoint_path = out_dir / "chimera_last.pt"
    best_checkpoint_path = out_dir / "chimera_best.pt"

    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    if resume:
        resume_state = load_checkpoint(
            checkpoint_path=resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
        )
        print(resume_state["message"])
        if resume_state["loaded"]:
            start_epoch = int(resume_state["epoch"]) + 1
            global_step = int(resume_state["global_step"])
            if resume_state["best_metric"] is not None:
                best_metric = float(resume_state["best_metric"])
            if ema is not None and resume_state["ema_state"] is not None:
                ema.load_state_dict(resume_state["ema_state"])

    write_json(
        summary_json_path,
        {
            "config_path": config_path,
            "model_info": model_info,
            "device": str(device),
            "use_ema": use_ema,
            "grad_clip_norm": grad_clip_norm,
        },
    )

    epochs = int(cfg["train"]["epochs"])
    grad_accum = max(int(cfg["train"].get("grad_accum", 1)), 1)
    last_epoch_loss = 0.0

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for step_index, (imgs, targets) in enumerate(pbar, start=1):
            imgs = imgs.to(device, non_blocking=device.type == "cuda")

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = model.compute_loss(imgs, targets, debug=debug_loss)
                loss = loss / grad_accum

            if not torch.isfinite(loss).all():
                print(f"warning: non-finite loss encountered at epoch={epoch + 1}, step={step_index}; skipping step")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if step_index % grad_accum == 0 or step_index == len(loader):
                if grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            global_step += 1
            loss_value = float(loss.detach().item() * grad_accum)
            epoch_loss_sum += loss_value
            epoch_steps += 1
            current_lr = float(optimizer.param_groups[0]["lr"])
            pbar.set_postfix(loss=loss_value, lr=current_lr)

            log_row = {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": loss_value,
                "lr": current_lr,
            }
            append_metrics_row(metrics_csv_path, log_row)
            append_jsonl(metrics_jsonl_path, log_row)

        epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
        last_epoch_loss = epoch_loss
        is_best = epoch_loss < best_metric
        best_metric = min(best_metric, epoch_loss)

        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            config=cfg,
            ema_state=ema.state_dict() if ema is not None else None,
            is_best=is_best,
            best_checkpoint_path=best_checkpoint_path,
        )

        epoch_summary = {
            "epoch": epoch + 1,
            "epoch_loss": epoch_loss,
            "best_metric": best_metric,
            "global_step": global_step,
        }
        append_jsonl(out_dir / "epoch_summaries.jsonl", epoch_summary)
        print(f"epoch={epoch + 1} loss={epoch_loss:.6f} best={best_metric:.6f}")
        print(vram_report("After epoch: "))

    final_weights_path = out_dir / "chimera_final_weights.pt"
    torch.save(model.state_dict(), final_weights_path)
    if ema is not None:
        torch.save(ema.state_dict(), out_dir / "chimera_ema_weights.pt")

    final_summary = {
        "last_epoch_loss": last_epoch_loss,
        "best_metric": best_metric,
        "global_step": global_step,
        "final_weights": str(final_weights_path),
        "use_ema": use_ema,
    }
    write_json(out_dir / "train_summary.json", final_summary)
    return final_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Detektor on a configured dataset")
    parser.add_argument("--config", type=str, default="configs/chimera_s_512.yaml", help="Path to the base training config YAML")
    parser.add_argument("--data-yaml", type=str, default="", help="Optional YOLO/Roboflow dataset YAML used to override train/val roots and class metadata")
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint path to resume optimizer, scaler, and epoch state")
    parser.add_argument("--weights", type=str, default="", help="Optional model weights or checkpoint to initialize from before training")
    parser.add_argument("--ema", action="store_true", help="Enable exponential moving average tracking for model weights")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping max-norm; 0 disables clipping")
    parser.add_argument("--debug-loss", action="store_true", help="Enable detailed loss component debugging output")
    args = parser.parse_args()

    train(
        config_path=args.config,
        data_yaml=args.data_yaml or None,
        resume=args.resume or None,
        weights=args.weights or None,
        use_ema=args.ema,
        grad_clip_norm=args.grad_clip,
        debug_loss=args.debug_loss,
    )
