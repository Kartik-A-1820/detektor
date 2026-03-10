from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float | None = None,
    config: Dict[str, Any] | None = None,
    ema_state: Dict[str, Any] | None = None,
    is_best: bool = False,
    best_checkpoint_path: str | Path | None = None,
) -> None:
    """Save a training checkpoint and optionally update the best checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "config": config,
        "ema_state": ema_state,
    }
    torch.save(payload, checkpoint_path)

    if is_best and best_checkpoint_path is not None:
        best_checkpoint_path = Path(best_checkpoint_path)
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, best_checkpoint_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint safely and restore optimizer and scaler state when available."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return {
            "loaded": False,
            "epoch": 0,
            "global_step": 0,
            "best_metric": None,
            "ema_state": None,
            "scheduler_state": None,
            "message": f"Checkpoint not found: {checkpoint_path}",
        }

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"], strict=strict)

    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scaler_state = checkpoint.get("scaler_state")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    scheduler_state = checkpoint.get("scheduler_state")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    return {
        "loaded": True,
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_metric": checkpoint.get("best_metric"),
        "scheduler_state": checkpoint.get("scheduler_state"),
        "config": checkpoint.get("config"),
        "ema_state": checkpoint.get("ema_state"),
        "message": f"Loaded checkpoint from {checkpoint_path}",
    }
