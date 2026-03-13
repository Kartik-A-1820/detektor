from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_dataset
from engine.ema import ModelEMA
from models.chimera import ChimeraODIS
from models.factory import build_model_from_config, load_model_weights
from utils.auto_train_config import plan_smart_retry, resolve_training_config, summarize_resolved_training, write_resolved_config
from utils.checkpoints import build_checkpoint_payload, load_checkpoint, save_checkpoint
from utils.collate import detection_segmentation_collate_fn
from utils.logging_utils import append_jsonl, append_metrics_row, write_json
from utils.model_info import get_model_info
from utils.reporting import (
    generate_metrics_summary,
    load_train_metrics,
    plot_epoch_metrics,
    plot_learning_rate,
    plot_loss_curves,
)
from utils.repro import set_seed
from utils.vram import set_vram_cap, vram_report


class RecoverableTrainingError(RuntimeError):
    """Signals a training failure that may be fixed by retrying with a safer config."""

    def __init__(self, failure_type: str, message: str):
        super().__init__(message)
        self.failure_type = failure_type


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


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(value, 1)
    return max(((max(value, 1) + multiple - 1) // multiple) * multiple, multiple)


def _auto_maximize_batch_size(
    cfg: Dict[str, Any],
    model: ChimeraODIS,
    dataset: Any,
    device: torch.device,
    task_mode: str,
) -> Dict[str, Any] | None:
    train_cfg = cfg.get("train", {})
    if device.type != "cuda" or not bool(train_cfg.get("auto_tune", True)):
        return None
    if not bool(train_cfg.get("maximize_batch_size", True)):
        return None
    if len(dataset) == 0:
        return None

    batch_multiple = max(int(train_cfg.get("batch_size_multiple", 4)), 1)
    current_batch = max(int(train_cfg.get("batch_size", batch_multiple)), 1)
    start_batch = _round_up_to_multiple(current_batch, batch_multiple)
    max_probe = max(int(train_cfg.get("max_batch_probe", 64)), start_batch)
    max_candidate = min(len(dataset), max_probe)
    if max_candidate < start_batch:
        return None

    amp_enabled = bool(train_cfg.get("amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_batch = 0
    last_error = ""

    for candidate in range(start_batch, max_candidate + 1, batch_multiple):
        try:
            loader = _build_train_loader(dataset, batch_size=candidate, num_workers=0, device=device)
            imgs, targets = next(iter(loader))
            imgs = imgs.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss_dict = model.compute_loss(imgs, targets, return_dict=True, task=task_mode)
            loss = loss_dict["loss_total"]
            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            best_batch = candidate
            last_error = ""

            del loader, imgs, targets, loss, loss_dict
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except RuntimeError as error:
            last_error = str(error)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if _is_oom_error(error):
                break
            return None

    if best_batch <= 0 or best_batch == current_batch:
        return None

    previous_grad_accum = max(int(train_cfg.get("grad_accum", 1)), 1)
    previous_effective_batch = max(current_batch * previous_grad_accum, 1)
    train_cfg["batch_size"] = best_batch
    train_cfg["grad_accum"] = 1

    current_lr = float(train_cfg.get("lr", 0.002))
    scaled_lr = current_lr * (best_batch / float(previous_effective_batch))
    train_cfg["lr"] = round(min(max(scaled_lr, 0.0005), 0.004), 6)

    return {
        "previous_batch_size": current_batch,
        "new_batch_size": best_batch,
        "previous_grad_accum": previous_grad_accum,
        "new_grad_accum": 1,
        "last_error": last_error,
        "new_lr": float(train_cfg["lr"]),
    }


def _clear_cuda_state() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _is_oom_error(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    text = str(error).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _log_resolved_runtime(resolved_runtime: Dict[str, Any]) -> None:
    label_map = (
        ("device", "resolved_device"),
        ("gpu_name", "resolved_gpu_name"),
        ("total_vram_gb", "resolved_total_vram_gb"),
        ("free_vram_gb", "resolved_free_vram_gb"),
        ("free_ram_gb", "resolved_free_ram_gb"),
        ("cpu_count", "resolved_cpu_count"),
        ("effective_vram_cap", "resolved_effective_vram_cap"),
        ("resolved_train_root", "resolved_train_root"),
        ("resolved_val_root", "resolved_val_root"),
        ("img_size", "resolved_img_size"),
        ("batch_size", "resolved_batch_size"),
        ("grad_accum", "resolved_grad_accum"),
        ("effective_batch_size", "resolved_effective_batch_size"),
        ("lr", "resolved_lr"),
        ("amp", "resolved_amp"),
        ("num_workers", "resolved_num_workers"),
        ("out_dir", "resolved_out_dir"),
        ("model_profile", "resolved_model_profile"),
        ("model_display_name", "resolved_model_display_name"),
        ("augment", "resolved_augment"),
        ("smart_training", "resolved_smart_training"),
    )
    for key, label in label_map:
        print(f"{label}: {resolved_runtime.get(key)}")


def _build_optimizer(cfg: Dict[str, Any], model: ChimeraODIS) -> torch.optim.Optimizer:
    optimizer_type = cfg["train"].get("optimizer", "adamw").lower()
    lr = cfg["train"]["lr"]
    weight_decay = cfg["train"]["weight_decay"]

    if optimizer_type == "sgd":
        momentum = cfg["train"].get("momentum", 0.937)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_type}. Choose 'adamw' or 'sgd'.")


def _build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_type = cfg["train"].get("scheduler", "cosine").lower()
    warmup_epochs = cfg["train"].get("warmup_epochs", 3)
    epochs = cfg["train"]["epochs"]

    if scheduler_type == "cosine":
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)).item())

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if scheduler_type == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {scheduler_type}. Choose 'cosine' or 'none'.")


def _resolve_task_mode(dataset: Any) -> str:
    task_mode = getattr(dataset, "task_mode", None)
    if task_mode is not None:
        return task_mode.value if hasattr(task_mode, "value") else str(task_mode)
    return "segment"


def _train_once(
    cfg: Dict[str, Any],
    resolved_runtime: Dict[str, Any],
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
    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
    effective_vram_cap = set_vram_cap(cfg["train"].get("vram_cap", 0.95))
    resolved_runtime["effective_vram_cap"] = round(float(effective_vram_cap), 4)
    device = _resolve_device(cfg.get("device", "cpu"))

    model = build_model_from_config(cfg).to(device)
    model_info = get_model_info(model)
    if model_info["trainable_params"] == 0:
        print("warning: model has no trainable parameters")

    _load_weights_if_provided(model, weights, device)

    dataset = build_dataset(cfg, split="train")
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty")

    task_mode_str = _resolve_task_mode(dataset)
    print(f"Training with task mode: {task_mode_str}")

    auto_batch_result = _auto_maximize_batch_size(
        cfg=cfg,
        model=model,
        dataset=dataset,
        device=device,
        task_mode=task_mode_str,
    )
    if auto_batch_result is not None:
        resolved_runtime["batch_size"] = int(cfg["train"]["batch_size"])
        resolved_runtime["grad_accum"] = int(cfg["train"]["grad_accum"])
        resolved_runtime["effective_batch_size"] = int(cfg["train"]["batch_size"]) * int(cfg["train"]["grad_accum"])
        resolved_runtime["lr"] = float(cfg["train"]["lr"])
        print(
            "auto_batch_probe: "
            f"batch_size {auto_batch_result['previous_batch_size']}->{auto_batch_result['new_batch_size']}, "
            f"grad_accum {auto_batch_result['previous_grad_accum']}->{auto_batch_result['new_grad_accum']}, "
            f"lr -> {auto_batch_result['new_lr']}"
        )

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

    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    amp_enabled = bool(cfg["train"].get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = ModelEMA(model, decay=float(cfg["train"].get("ema_decay", 0.9998))) if use_ema else None

    out_dir = Path(cfg.get("logging", {}).get("out_dir", "runs/chimera"))
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = write_resolved_config(cfg, out_dir / "resolved_train_config.yaml")
    _log_resolved_runtime(resolved_runtime)

    metrics_csv_path = out_dir / "train_metrics.csv"
    metrics_jsonl_path = out_dir / "train_metrics.jsonl"
    summary_json_path = out_dir / "run_summary.json"
    last_checkpoint_path = out_dir / "chimera_last.pt"
    best_checkpoint_path = out_dir / "chimera_best.pt"

    start_epoch = 0
    global_step = 0
    best_metric_mode = "val_map50" if run_val else "train_loss"
    best_metric = float("-inf") if best_metric_mode == "val_map50" else float("inf")
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
            "optimizer": cfg["train"].get("optimizer", "adamw").lower(),
            "scheduler": cfg["train"].get("scheduler", "cosine").lower(),
            "warmup_epochs": cfg["train"].get("warmup_epochs", 3),
            "best_metric_mode": best_metric_mode,
        },
    )

    epochs = int(cfg["train"]["epochs"])
    grad_accum = max(int(cfg["train"].get("grad_accum", 1)), 1)
    non_finite_patience = max(int(cfg.get("smart_training", {}).get("non_finite_patience", 2)), 1)
    non_finite_events = 0
    last_epoch_loss = 0.0

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            model.detection_loss.current_epoch = epoch

            optimizer.zero_grad(set_to_none=True)
            epoch_loss_sum = 0.0
            epoch_steps = 0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for step_index, (imgs, targets) in enumerate(pbar, start=1):
                try:
                    imgs = imgs.to(device, non_blocking=device.type == "cuda")

                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        loss_dict = model.compute_loss(imgs, targets, return_dict=True, debug=debug_loss, task=task_mode_str)

                    if _has_non_finite_loss_components(loss_dict):
                        non_finite_events += 1
                        if amp_enabled:
                            raise RecoverableTrainingError(
                                "amp_instability",
                                f"non-finite loss components under AMP at epoch={epoch + 1}, step={step_index}",
                            )
                        if non_finite_events >= non_finite_patience:
                            raise RecoverableTrainingError(
                                "non_finite_loss",
                                f"non-finite loss components in fp32 at epoch={epoch + 1}, step={step_index}",
                            )
                        print(f"warning: non-finite loss encountered at epoch={epoch + 1}, step={step_index}; skipping step")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    loss = loss_dict["loss_total"] / grad_accum
                    if not torch.isfinite(loss).all():
                        non_finite_events += 1
                        if amp_enabled:
                            raise RecoverableTrainingError(
                                "amp_instability",
                                f"non-finite total loss under AMP at epoch={epoch + 1}, step={step_index}",
                            )
                        if non_finite_events >= non_finite_patience:
                            raise RecoverableTrainingError(
                                "non_finite_loss",
                                f"non-finite total loss in fp32 at epoch={epoch + 1}, step={step_index}",
                            )
                        print(f"warning: non-finite loss encountered at epoch={epoch + 1}, step={step_index}; skipping step")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    if amp_enabled:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if step_index % grad_accum == 0 or step_index == len(loader):
                        if amp_enabled:
                            scaler.unscale_(optimizer)

                        has_nan_grad = False
                        for param in model.parameters():
                            if param.grad is not None and not torch.isfinite(param.grad).all():
                                has_nan_grad = True
                                break

                        if has_nan_grad:
                            non_finite_events += 1
                            if amp_enabled:
                                raise RecoverableTrainingError(
                                    "amp_instability",
                                    f"non-finite gradients under AMP at epoch={epoch + 1}, step={step_index}",
                                )
                            if non_finite_events >= non_finite_patience:
                                raise RecoverableTrainingError(
                                    "non_finite_grad",
                                    f"non-finite gradients in fp32 at epoch={epoch + 1}, step={step_index}",
                                )
                            print(f"warning: non-finite gradients at epoch={epoch + 1}, step={step_index}; skipping step")
                            optimizer.zero_grad(set_to_none=True)
                            continue

                        if grad_clip_norm > 0.0:
                            total_norm = clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                            if total_norm > grad_clip_norm * 10:
                                print(f"warning: large gradient norm {total_norm:.2f} at epoch={epoch + 1}, step={step_index}")

                        if amp_enabled:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if ema is not None:
                            ema.update(model)

                    global_step += 1
                    non_finite_events = 0
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
                except RecoverableTrainingError:
                    optimizer.zero_grad(set_to_none=True)
                    raise
                except RuntimeError as error:
                    optimizer.zero_grad(set_to_none=True)
                    if _is_oom_error(error):
                        raise RecoverableTrainingError(
                            "oom",
                            f"out of memory at epoch={epoch + 1}, step={step_index}: {error}",
                        ) from error
                    raise

            epoch_loss_avg = epoch_loss_sum / max(epoch_steps, 1)
            last_epoch_loss = epoch_loss_avg
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{epoch_loss_avg:.3g}", "lr": f"{current_lr:.3g}"})
            pbar.close()

            if scheduler is not None:
                scheduler.step()

            selection_metric = epoch_loss_avg
            is_best = selection_metric < best_metric if best_metric_mode == "train_loss" else False

            val_log = None
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
                    val_log = {
                        "epoch": epoch + 1,
                        "val_precision": val_metrics.get("detection", {}).get("precision", 0.0),
                        "val_recall": val_metrics.get("detection", {}).get("recall", 0.0),
                        "val_map50": val_metrics.get("detection", {}).get("ap50", 0.0),
                        "val_mean_iou": val_metrics.get("detection", {}).get("mean_iou", 0.0),
                    }
                    selection_metric = float(val_log["val_map50"])
                    is_best = selection_metric > best_metric
                    append_jsonl(out_dir / "val_metrics.jsonl", val_log)
                    print(
                        f"Validation: P={val_log['val_precision']:.3f} "
                        f"R={val_log['val_recall']:.3f} mAP50={val_log['val_map50']:.3f}"
                    )
                except Exception as error:
                    print(f"warning: validation failed: {error}")

            if is_best:
                best_metric = selection_metric

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
                "best_metric_mode": best_metric_mode,
                "selection_metric": selection_metric,
                "global_step": global_step,
            }
            if val_log is not None:
                epoch_summary.update(val_log)
            append_jsonl(out_dir / "epoch_summaries.jsonl", epoch_summary)
            print(f"epoch={epoch + 1} loss={epoch_loss_avg:.6f} best_{best_metric_mode}={best_metric:.6f}")
            print(vram_report("After epoch: "))

            try:
                metrics_df = load_train_metrics(metrics_csv_path)
                if metrics_df is not None and len(metrics_df) > 0:
                    plots_dir = out_dir / "plots"
                    plots_dir.mkdir(exist_ok=True)
                    plot_loss_curves(metrics_df, plots_dir)
                    plot_learning_rate(metrics_df, plots_dir)
                    plot_epoch_metrics(metrics_df, plots_dir)
                    print(f"Generated training plots in {plots_dir}")
            except Exception as error:
                print(f"warning: failed to generate plots: {error}")
    except RecoverableTrainingError:
        _clear_cuda_state()
        raise
    except RuntimeError as error:
        _clear_cuda_state()
        if _is_oom_error(error):
            raise RecoverableTrainingError("oom", f"out of memory while setting up training: {error}") from error
        raise
    finally:
        _clear_cuda_state()

    final_weights_path = out_dir / "chimera_final_weights.pt"
    torch.save(
        build_checkpoint_payload(
            model,
            epoch=max(epochs - 1, 0),
            global_step=global_step,
            best_metric=best_metric,
            config=cfg,
            ema_state=ema.state_dict() if ema is not None else None,
        ),
        final_weights_path,
    )
    if ema is not None:
        torch.save(
            {
                "format_version": 2,
                "ema_state": ema.state_dict(),
                "config": cfg,
                "model_config": cfg.get("model", {}),
            },
            out_dir / "chimera_ema_weights.pt",
        )

    final_summary = {
        "last_epoch_loss": last_epoch_loss,
        "best_metric": best_metric,
        "global_step": global_step,
        "final_weights": str(final_weights_path),
        "use_ema": use_ema,
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "train_summary.json", final_summary)

    print("\n" + "=" * 60)
    print("GENERATING FINAL TRAINING REPORT")
    print("=" * 60)
    try:
        metrics_df = load_train_metrics(metrics_csv_path)
        if metrics_df is not None and len(metrics_df) > 0:
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            plot_loss_curves(metrics_df, plots_dir)
            plot_learning_rate(metrics_df, plots_dir)
            plot_epoch_metrics(metrics_df, plots_dir)
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
    except Exception as error:
        print(f"warning: failed to generate final report: {error}")

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    return final_summary


def train(
    config_path: str | None,
    data_yaml: str | None = None,
    overrides: Dict[str, Any] | None = None,
    resume: str | None = None,
    weights: str | None = None,
    use_ema: bool = False,
    grad_clip_norm: float = 0.0,
    debug_loss: bool = False,
    run_val: bool = False,
    val_freq: int = 1,
) -> Dict[str, Any]:
    """Run training with a smart orchestrator that retries recoverable failures."""
    base_cfg, base_runtime = resolve_training_config(config_path, data_yaml, overrides=overrides)
    attempt = 1
    current_cfg = base_cfg
    attempt_history: list[Dict[str, Any]] = []

    while True:
        current_cfg.setdefault("smart_training", {})
        current_cfg["smart_training"]["attempt"] = attempt
        resolved_runtime = summarize_resolved_training(current_cfg, base_runtime)
        print(f"smart_training_attempt: {attempt}/{current_cfg['smart_training'].get('max_attempts', 1)}")

        try:
            final_summary = _train_once(
                cfg=current_cfg,
                resolved_runtime=resolved_runtime,
                config_path=config_path,
                data_yaml=data_yaml,
                resume=resume if attempt == 1 else None,
                weights=weights,
                use_ema=use_ema,
                grad_clip_norm=grad_clip_norm,
                debug_loss=debug_loss,
                run_val=run_val,
                val_freq=val_freq,
            )
            final_summary["smart_training"] = {
                "attempts_used": attempt,
                "attempt_history": attempt_history,
                "final_runtime": resolved_runtime,
            }
            return final_summary
        except RecoverableTrainingError as error:
            retry_plan = plan_smart_retry(
                cfg=current_cfg,
                failure_type=error.failure_type,
                failure_message=str(error),
                attempt_index=attempt,
            )
            attempt_record = {
                "attempt": attempt,
                "failure_type": error.failure_type,
                "failure_message": str(error),
                "out_dir": current_cfg.get("logging", {}).get("out_dir", ""),
            }
            if retry_plan is None:
                attempt_history.append(attempt_record)
                raise RuntimeError(
                    f"smart training could not recover from {error.failure_type} after {attempt} attempts: {error}"
                ) from error

            current_cfg, retry_info = retry_plan
            attempt_record["retry"] = retry_info
            attempt_history.append(attempt_record)
            print(
                f"warning: attempt {attempt} failed with {error.failure_type}; "
                f"retrying as attempt {retry_info['next_attempt']} with changes: {', '.join(retry_info['changes'])}"
            )
            attempt += 1


if __name__ == "__main__":
    import argparse

    def _update_nested(payload: Dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        cursor = payload
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = value

    def _build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        mapping = (
            ("device", ("device",)),
            ("seed", ("seed",)),
            ("deterministic", ("deterministic",)),
            ("img_size", ("train", "img_size")),
            ("epochs", ("train", "epochs")),
            ("batch_size", ("train", "batch_size")),
            ("num_workers", ("train", "num_workers")),
            ("lr", ("train", "lr")),
            ("weight_decay", ("train", "weight_decay")),
            ("optimizer", ("train", "optimizer")),
            ("scheduler", ("train", "scheduler")),
            ("warmup_epochs", ("train", "warmup_epochs")),
            ("momentum", ("train", "momentum")),
            ("amp", ("train", "amp")),
            ("grad_accum", ("train", "grad_accum")),
            ("vram_cap", ("train", "vram_cap")),
            ("conf_thresh", ("train", "conf_thresh")),
            ("iou_thresh", ("train", "iou_thresh")),
            ("auto_tune", ("train", "auto_tune")),
            ("maximize_batch_size", ("train", "maximize_batch_size")),
            ("batch_size_multiple", ("train", "batch_size_multiple")),
            ("max_batch_probe", ("train", "max_batch_probe")),
            ("hsv_h", ("augment", "hsv_h")),
            ("hsv_s", ("augment", "hsv_s")),
            ("hsv_v", ("augment", "hsv_v")),
            ("fliplr", ("augment", "fliplr")),
            ("flipud", ("augment", "flipud")),
            ("translate", ("augment", "translate")),
            ("scale", ("augment", "scale")),
            ("mosaic", ("augment", "mosaic")),
            ("cutmix", ("augment", "cutmix")),
            ("random_cut", ("augment", "random_cut")),
            ("random_cut_holes", ("augment", "random_cut_holes")),
            ("random_cut_scale", ("augment", "random_cut_scale")),
            ("augment_enabled", ("augment", "enabled")),
            ("model_profile", ("model", "profile")),
            ("proto_k", ("model", "proto_k")),
            ("out_dir", ("logging", "out_dir")),
        )
        for arg_name, path in mapping:
            value = getattr(args, arg_name, None)
            if value is not None:
                _update_nested(overrides, path, value)
        return overrides

    parser = argparse.ArgumentParser(description="Train Detektor on a configured dataset")
    parser.add_argument("--config", type=str, default="", help="Optional base training config YAML. If omitted, Detektor auto-tunes settings from the dataset and available hardware.")
    parser.add_argument("--data-yaml", type=str, default="", help="YOLO/Roboflow dataset YAML used to resolve train/val roots and class metadata")
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"), help="Override device selection")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training where possible")
    parser.add_argument("--img-size", dest="img_size", type=int, help="Override training image size")
    parser.add_argument("--epochs", type=int, help="Override epoch count")
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Override physical batch size")
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Override data loader worker count")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, help="Override weight decay")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), help="Override optimizer")
    parser.add_argument("--scheduler", choices=("cosine", "none"), help="Override scheduler")
    parser.add_argument("--warmup-epochs", dest="warmup_epochs", type=int, help="Override warmup epochs")
    parser.add_argument("--momentum", type=float, help="Override SGD momentum")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP explicitly")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP explicitly")
    parser.add_argument("--grad-accum", dest="grad_accum", type=int, help="Override gradient accumulation steps")
    parser.add_argument("--vram-cap", dest="vram_cap", type=float, help="Target fraction of currently free VRAM to reserve for this process")
    parser.add_argument("--conf-thresh", dest="conf_thresh", type=float, help="Override confidence threshold used during validation hooks")
    parser.add_argument("--iou-thresh", dest="iou_thresh", type=float, help="Override IoU threshold used during validation hooks")
    parser.add_argument("--auto-tune", dest="auto_tune", action="store_true", help="Enable hardware-aware auto configuration")
    parser.add_argument("--no-auto-tune", dest="auto_tune", action="store_false", help="Disable hardware-aware auto configuration")
    parser.add_argument("--maximize-batch-size", dest="maximize_batch_size", action="store_true", help="Probe for the largest safe CUDA batch size")
    parser.add_argument("--no-maximize-batch-size", dest="maximize_batch_size", action="store_false", help="Disable CUDA batch probing")
    parser.add_argument("--batch-size-multiple", dest="batch_size_multiple", type=int, help="Force CUDA batch probing to use multiples of this value")
    parser.add_argument("--max-batch-probe", dest="max_batch_probe", type=int, help="Upper bound for CUDA batch probing")
    parser.add_argument("--augment", dest="augment_enabled", action="store_true", help="Enable training augmentations explicitly")
    parser.add_argument("--no-augment", dest="augment_enabled", action="store_false", help="Disable training augmentations explicitly")
    parser.add_argument("--hsv-h", dest="hsv_h", type=float, help="Override HSV hue augmentation strength")
    parser.add_argument("--hsv-s", dest="hsv_s", type=float, help="Override HSV saturation augmentation strength")
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, help="Override HSV value augmentation strength")
    parser.add_argument("--fliplr", type=float, help="Override horizontal flip probability")
    parser.add_argument("--flipud", type=float, help="Override vertical flip probability")
    parser.add_argument("--translate", type=float, help="Override translation augmentation strength")
    parser.add_argument("--scale", type=float, help="Override scale augmentation strength")
    parser.add_argument("--mosaic", type=float, help="Override mosaic augmentation probability")
    parser.add_argument("--cutmix", type=float, help="Override CutMix augmentation probability")
    parser.add_argument("--random-cut", dest="random_cut", type=float, help="Override random cut augmentation probability")
    parser.add_argument("--random-cut-holes", dest="random_cut_holes", type=int, help="Override random cut hole count")
    parser.add_argument("--random-cut-scale", dest="random_cut_scale", type=float, help="Override random cut size ratio")
    parser.add_argument("--model", dest="model_profile", type=str, help="Override architecture profile name")
    parser.add_argument("--proto-k", dest="proto_k", type=int, help="Override prototype mask count")
    parser.add_argument("--out-dir", dest="out_dir", type=str, help="Override run output directory")
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint path to resume optimizer, scaler, and epoch state")
    parser.add_argument("--weights", type=str, default="", help="Optional model weights or checkpoint to initialize from before training")
    parser.add_argument("--ema", action="store_true", help="Enable exponential moving average tracking for model weights")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping max-norm; 0 disables clipping")
    parser.add_argument("--debug-loss", action="store_true", help="Enable detailed loss component debugging output")
    parser.add_argument("--run-val", action="store_true", help="Run validation during training")
    parser.add_argument("--val-freq", type=int, default=1, help="Validation frequency in epochs (default: 1)")
    parser.set_defaults(amp=None, auto_tune=None, maximize_batch_size=None, augment_enabled=None, deterministic=None)
    args = parser.parse_args()

    train(
        config_path=args.config or None,
        data_yaml=args.data_yaml or None,
        overrides=_build_cli_overrides(args),
        resume=args.resume or None,
        weights=args.weights or None,
        use_ema=args.ema,
        grad_clip_norm=args.grad_clip,
        debug_loss=args.debug_loss,
        run_val=args.run_val,
        val_freq=args.val_freq,
    )
