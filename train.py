from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_dataset
from models.chimera import ChimeraODIS
from models.factory import build_model_from_config, load_model_weights
from utils.auto_train_config import resolve_training_config, write_resolved_config
from utils.checkpoints import load_checkpoint, save_checkpoint
from utils.collate import detection_segmentation_collate_fn
from engine.ema import ModelEMA
from utils.logging_utils import append_jsonl, append_metrics_row, write_json
from utils.model_info import get_model_info
from utils.repro import set_seed
from utils.reporting import (
    load_train_metrics,
    plot_loss_curves,
    plot_learning_rate,
    plot_epoch_metrics,
    generate_metrics_summary,
)
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
    load_model_weights(model, checkpoint, strict=True)


def _has_non_finite_loss_components(loss_dict: Dict[str, torch.Tensor]) -> bool:
    """Return True when any tensor-valued loss component contains NaN/Inf."""
    for value in loss_dict.values():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            return True
    return False


def _build_train_loader(dataset: Any, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_segmentation_collate_fn,
    )


def train(
    config_path: str | None,
    data_yaml: str | None = None,
    resume: str | None = None,
    weights: str | None = None,
    use_ema: bool = False,
    grad_clip_norm: float = 0.0,
    debug_loss: bool = False,
    run_val: bool = False,
    val_freq: int = 1,
) -> Dict[str, Any]:
    """Run the production-hardened training loop with safe resume and logging helpers."""
    cfg, resolved_runtime = resolve_training_config(config_path, data_yaml)

    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
    set_vram_cap(cfg["train"].get("vram_cap", 0.8))
    device = _resolve_device(cfg.get("device", "cpu"))

    model = build_model_from_config(cfg).to(device)
    model_info = get_model_info(model)
    if model_info["trainable_params"] == 0:
        print("warning: model has no trainable parameters")

    _load_weights_if_provided(model, weights, device)

    dataset = build_dataset(cfg, split="train")
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty")
    
    # Get task mode from dataset
    task_mode = getattr(dataset, 'task_mode', None)
    if task_mode is not None:
        task_mode_str = task_mode.value if hasattr(task_mode, 'value') else str(task_mode)
    else:
        task_mode_str = "segment"  # Default to segment mode
    print(f"Training with task mode: {task_mode_str}")

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    if num_workers > 0:
        try:
            ctx = mp.get_context("spawn")
            probe_queue = ctx.Queue()
            probe_queue.close()
        except PermissionError:
            print("warning: data loader worker processes are unavailable here; falling back to num_workers=0")
            num_workers = 0
            cfg["train"]["num_workers"] = 0
            resolved_runtime["num_workers"] = 0
    loader = _build_train_loader(dataset, batch_size=batch_size, num_workers=num_workers, device=device)

    optimizer_type = cfg["train"].get("optimizer", "adamw").lower()
    lr = cfg["train"]["lr"]
    weight_decay = cfg["train"]["weight_decay"]
    
    if optimizer_type == "sgd":
        momentum = cfg["train"].get("momentum", 0.937)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}. Choose 'adamw' or 'sgd'.")
    
    scheduler_type = cfg["train"].get("scheduler", "cosine").lower()
    warmup_epochs = cfg["train"].get("warmup_epochs", 3)
    epochs = cfg["train"]["epochs"]
    
    if scheduler_type == "cosine":
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)).item())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}. Choose 'cosine' or 'none'.")
    
    amp_enabled = bool(cfg["train"].get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = ModelEMA(model, decay=float(cfg["train"].get("ema_decay", 0.9998))) if use_ema else None

    out_dir = Path(cfg.get("logging", {}).get("out_dir", "runs/chimera"))
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = write_resolved_config(cfg, out_dir / "resolved_train_config.yaml")
    print(f"resolved_device: {resolved_runtime['device']}")
    print(f"gpu_name: {resolved_runtime['gpu_name']}")
    print(f"total_vram_gb: {resolved_runtime['total_vram_gb']}")
    print(f"resolved_train_root: {resolved_runtime['resolved_train_root']}")
    print(f"resolved_val_root: {resolved_runtime['resolved_val_root']}")
    print(f"resolved_img_size: {resolved_runtime['img_size']}")
    print(f"resolved_batch_size: {resolved_runtime['batch_size']}")
    print(f"resolved_grad_accum: {resolved_runtime['grad_accum']}")
    print(f"resolved_effective_batch: {resolved_runtime['effective_batch_size']}")
    print(f"resolved_lr: {resolved_runtime['lr']}")
    print(f"resolved_amp: {resolved_runtime['amp']}")
    print(f"resolved_num_workers: {resolved_runtime['num_workers']}")
    print(f"resolved_out_dir: {resolved_runtime['out_dir']}")
    print(f"resolved_model_profile: {resolved_runtime['model_profile']}")
    print(f"resolved_model_name: {resolved_runtime['model_display_name']}")
    print(f"resolved_augment: {resolved_runtime['augment']}")
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
            scheduler=scheduler,
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
            "config_path": str(config_path or ""),
            "resolved_config_path": str(resolved_config_path),
            "resolved_runtime": resolved_runtime,
            "model_info": model_info,
            "device": str(device),
            "use_ema": use_ema,
            "grad_clip_norm": grad_clip_norm,
            "optimizer": optimizer_type,
            "scheduler": scheduler_type,
            "warmup_epochs": warmup_epochs,
        },
    )

    epochs = int(cfg["train"]["epochs"])
    grad_accum = max(int(cfg["train"].get("grad_accum", 1)), 1)
    last_epoch_loss = 0.0

    for epoch in range(start_epoch, epochs):
        model.train()
        
        # Update epoch for warmup in detection loss
        model.detection_loss.current_epoch = epoch
        
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for step_index, (imgs, targets) in enumerate(pbar, start=1):
            imgs = imgs.to(device, non_blocking=device.type == "cuda")
            used_amp_for_step = amp_enabled

            def compute_loss_dict(use_amp: bool) -> Dict[str, torch.Tensor]:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    return model.compute_loss(imgs, targets, return_dict=True, debug=debug_loss, task=task_mode_str)

            loss_dict = compute_loss_dict(used_amp_for_step)
            if _has_non_finite_loss_components(loss_dict):
                if used_amp_for_step:
                    print(
                        f"warning: non-finite loss components at epoch={epoch + 1}, step={step_index} with AMP; retrying in fp32"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    amp_enabled = False
                    used_amp_for_step = False
                    loss_dict = compute_loss_dict(False)
                if _has_non_finite_loss_components(loss_dict):
                    print(f"warning: non-finite loss encountered at epoch={epoch + 1}, step={step_index}; skipping step")
                    optimizer.zero_grad(set_to_none=True)
                    continue

            loss = loss_dict["loss_total"] / grad_accum
            if not torch.isfinite(loss).all():
                print(f"warning: non-finite loss encountered at epoch={epoch + 1}, step={step_index}; skipping step")
                optimizer.zero_grad(set_to_none=True)
                continue

            if used_amp_for_step:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step_index % grad_accum == 0 or step_index == len(loader):
                if used_amp_for_step:
                    scaler.unscale_(optimizer)

                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        has_nan_grad = True
                        break

                if has_nan_grad and used_amp_for_step:
                    print(
                        f"warning: non-finite gradients at epoch={epoch + 1}, step={step_index} with AMP; retrying in fp32"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    amp_enabled = False
                    used_amp_for_step = False
                    loss_dict = compute_loss_dict(False)
                    if _has_non_finite_loss_components(loss_dict):
                        print(f"warning: non-finite loss encountered at epoch={epoch + 1}, step={step_index}; skipping step")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    loss = loss_dict["loss_total"] / grad_accum
                    if not torch.isfinite(loss).all():
                        print(f"warning: non-finite loss encountered at epoch={epoch + 1}, step={step_index}; skipping step")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    loss.backward()
                    has_nan_grad = False
                    for param in model.parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            has_nan_grad = True
                            break

                if has_nan_grad:
                    print(f"warning: non-finite gradients at epoch={epoch + 1}, step={step_index}; skipping step")
                    optimizer.zero_grad(set_to_none=True)
                    if used_amp_for_step:
                        scaler.update()
                    continue

                if grad_clip_norm > 0.0:
                    total_norm = clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    if total_norm > grad_clip_norm * 10:
                        print(f"warning: large gradient norm {total_norm:.2f} at epoch={epoch + 1}, step={step_index}")

                if used_amp_for_step:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
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
                "loss_total": loss_value,
                "loss_cls": float(loss_dict["loss_cls"].detach()),
                "loss_box": float(loss_dict["loss_box"].detach()),
                "loss_obj": float(loss_dict["loss_obj"].detach()),
                "loss_mask": float(loss_dict["loss_mask"].detach()),
                "loss_mask_bce": float(loss_dict["loss_mask_bce"].detach()),
                "loss_mask_dice": float(loss_dict["loss_mask_dice"].detach()),
                "num_fg": float(loss_dict["num_fg"].detach()),
                "num_mask_pos": float(loss_dict["num_mask_pos"].detach()),
                "lr": current_lr,
            }
            append_metrics_row(metrics_csv_path, log_row)
            append_jsonl(metrics_jsonl_path, log_row)

        epoch_loss_avg = epoch_loss_sum / max(epoch_steps, 1)
        last_epoch_loss = epoch_loss_avg
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({"loss": f"{epoch_loss_avg:.3g}", "lr": f"{current_lr:.3g}"})
        pbar.close()
        
        if scheduler is not None:
            scheduler.step()

        is_best = epoch_loss_avg < best_metric
        best_metric = min(best_metric, epoch_loss_avg)

        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
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
            "epoch_loss": epoch_loss_avg,
            "best_metric": best_metric,
            "global_step": global_step,
        }
        append_jsonl(out_dir / "epoch_summaries.jsonl", epoch_summary)
        print(f"epoch={epoch + 1} loss={epoch_loss_avg:.6f} best={best_metric:.6f}")
        print(vram_report("After epoch: "))
        
        # Run validation if enabled
        if run_val and (epoch + 1) % val_freq == 0:
            print(f"\nRunning validation at epoch {epoch + 1}...")
            try:
                from validate import validate
                
                val_metrics = validate(
                    config_path=str(resolved_config_path),
                    data_yaml=data_yaml if data_yaml else None,
                    weights=str(last_checkpoint_path),
                    batch_size=cfg["train"].get("batch_size", 8),
                    conf_thresh=0.25,
                    iou_thresh=0.6,
                )
                
                # Log validation metrics
                val_log = {
                    "epoch": epoch + 1,
                    "val_precision": val_metrics.get("detection", {}).get("precision", 0.0),
                    "val_recall": val_metrics.get("detection", {}).get("recall", 0.0),
                    "val_map50": val_metrics.get("detection", {}).get("ap50", 0.0),
                    "val_mean_iou": val_metrics.get("detection", {}).get("mean_iou", 0.0),
                }
                append_jsonl(out_dir / "val_metrics.jsonl", val_log)
                print(f"Validation: P={val_log['val_precision']:.3f} R={val_log['val_recall']:.3f} mAP50={val_log['val_map50']:.3f}")
            except Exception as e:
                print(f"warning: validation failed: {e}")
        
        # Auto-generate plots after each epoch
        try:
            metrics_df = load_train_metrics(metrics_csv_path)
            if metrics_df is not None and len(metrics_df) > 0:
                plots_dir = out_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                
                # Generate plots
                plot_loss_curves(metrics_df, plots_dir)
                plot_learning_rate(metrics_df, plots_dir)
                plot_epoch_metrics(metrics_df, plots_dir)
                
                print(f"Generated training plots in {plots_dir}")
        except Exception as e:
            print(f"warning: failed to generate plots: {e}")

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
    
    # Generate final comprehensive training report
    print("\n" + "=" * 60)
    print("GENERATING FINAL TRAINING REPORT")
    print("=" * 60)
    try:
        metrics_df = load_train_metrics(metrics_csv_path)
        if metrics_df is not None and len(metrics_df) > 0:
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Generate all plots
            plot_loss_curves(metrics_df, plots_dir)
            plot_learning_rate(metrics_df, plots_dir)
            plot_epoch_metrics(metrics_df, plots_dir)
            
            # Generate training summary
            summary_path = out_dir / "training_summary.json"
            generate_metrics_summary(metrics_df, None, None, summary_path)
            print(f"resolved_config_path: {resolved_config_path}")
            
            print(f"\n[OK] Training plots saved to: {plots_dir}")
            print(f"[OK] Training summary saved to: {summary_path}")
            print(f"[OK] Metrics CSV: {metrics_csv_path}")
            print(f"[OK] Best checkpoint: {best_checkpoint_path}")
            print(f"[OK] Final weights: {final_weights_path}")
        else:
            print("warning: no metrics data available for report generation")
    except Exception as e:
        print(f"warning: failed to generate final report: {e}")
    
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return final_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Detektor on a configured dataset")
    parser.add_argument("--config", type=str, default="", help="Optional base training config YAML. If omitted, Detektor auto-tunes settings from the dataset and available hardware.")
    parser.add_argument("--data-yaml", type=str, default="", help="YOLO/Roboflow dataset YAML used to resolve train/val roots and class metadata")
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint path to resume optimizer, scaler, and epoch state")
    parser.add_argument("--weights", type=str, default="", help="Optional model weights or checkpoint to initialize from before training")
    parser.add_argument("--ema", action="store_true", help="Enable exponential moving average tracking for model weights")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping max-norm; 0 disables clipping")
    parser.add_argument("--debug-loss", action="store_true", help="Enable detailed loss component debugging output")
    parser.add_argument("--run-val", action="store_true", help="Run validation during training")
    parser.add_argument("--val-freq", type=int, default=1, help="Validation frequency in epochs (default: 1)")
    args = parser.parse_args()

    train(
        config_path=args.config or None,
        data_yaml=args.data_yaml or None,
        resume=args.resume or None,
        weights=args.weights or None,
        use_ema=args.ema,
        grad_clip_norm=args.grad_clip,
        debug_loss=args.debug_loss,
        run_val=args.run_val,
        val_freq=args.val_freq,
    )
