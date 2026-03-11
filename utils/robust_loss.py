"""Production-grade robustness utilities for loss computation.

This module provides comprehensive safeguards against common training failures:
- NaN/Inf sanitization
- Division by zero protection
- Empty batch handling
- Gradient explosion prevention
"""

from __future__ import annotations

import torch
from torch import Tensor


EPS = 1e-7  # Epsilon for numerical stability


def sanitize_tensor(
    tensor: Tensor,
    nan_value: float = 0.0,
    posinf_value: float = 1e6,
    neginf_value: float = -1e6,
    name: str = "tensor",
) -> Tensor:
    """Sanitize tensor by replacing NaN/Inf values.
    
    Args:
        tensor: Input tensor to sanitize
        nan_value: Replacement value for NaN
        posinf_value: Replacement value for +Inf
        neginf_value: Replacement value for -Inf
        name: Tensor name for logging
        
    Returns:
        Sanitized tensor
    """
    if not torch.isfinite(tensor).all():
        num_nan = torch.isnan(tensor).sum().item()
        num_inf = torch.isinf(tensor).sum().item()
        if num_nan > 0 or num_inf > 0:
            print(f"warning: sanitizing {name}: {num_nan} NaN, {num_inf} Inf values")
        tensor = torch.nan_to_num(
            tensor,
            nan=nan_value,
            posinf=posinf_value,
            neginf=neginf_value,
        )
    return tensor


def safe_divide(
    numerator: Tensor,
    denominator: Tensor,
    eps: float = EPS,
    default: float = 0.0,
) -> Tensor:
    """Safe division with epsilon and zero handling.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        eps: Epsilon to add to denominator
        default: Default value when denominator is zero
        
    Returns:
        Result of safe division
    """
    # Add epsilon to prevent division by zero
    safe_denom = denominator + eps
    result = numerator / safe_denom
    
    # Replace any remaining NaN/Inf
    result = sanitize_tensor(result, nan_value=default, name="division_result")
    
    return result


def safe_mean(tensor: Tensor, dim: int | None = None, eps: float = EPS) -> Tensor:
    """Compute mean with safeguards against empty tensors.
    
    Args:
        tensor: Input tensor
        dim: Dimension to reduce (None for all)
        eps: Epsilon for numerical stability
        
    Returns:
        Safe mean value
    """
    if tensor.numel() == 0:
        return tensor.new_tensor(0.0)
    
    result = tensor.mean(dim=dim)
    result = sanitize_tensor(result, name="mean_result")
    return result


def safe_sum(tensor: Tensor, dim: int | None = None) -> Tensor:
    """Compute sum with NaN/Inf sanitization.
    
    Args:
        tensor: Input tensor
        dim: Dimension to reduce (None for all)
        
    Returns:
        Safe sum value
    """
    if tensor.numel() == 0:
        return tensor.new_tensor(0.0)
    
    result = tensor.sum(dim=dim)
    result = sanitize_tensor(result, name="sum_result")
    return result


def clamp_loss(
    loss: Tensor,
    min_value: float = 0.0,
    max_value: float = 1e4,
    name: str = "loss",
) -> Tensor:
    """Clamp loss values to prevent explosion.
    
    Args:
        loss: Loss tensor
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Loss name for logging
        
    Returns:
        Clamped loss
    """
    # First sanitize NaN/Inf
    loss = sanitize_tensor(loss, nan_value=max_value, name=name)
    
    # Then clamp to reasonable range
    original_max = loss.max().item() if loss.numel() > 0 else 0.0
    loss = loss.clamp(min=min_value, max=max_value)
    
    if original_max > max_value:
        print(f"warning: clamped {name} from {original_max:.2f} to {max_value:.2f}")
    
    return loss


def validate_batch(
    batch_size: int,
    num_targets: int,
    min_batch_size: int = 1,
) -> bool:
    """Validate batch consistency.
    
    Args:
        batch_size: Batch size from images
        num_targets: Number of targets
        min_batch_size: Minimum allowed batch size
        
    Returns:
        True if batch is valid
    """
    if batch_size < min_batch_size:
        print(f"warning: batch size {batch_size} below minimum {min_batch_size}")
        return False
    
    if batch_size != num_targets:
        print(f"warning: batch size mismatch: {batch_size} images vs {num_targets} targets")
        return False
    
    return True


def validate_boxes(boxes: Tensor, image_size: tuple[int, int]) -> Tensor:
    """Validate and sanitize bounding boxes.
    
    Args:
        boxes: Boxes tensor [N, 4] in xyxy format
        image_size: (height, width) of image
        
    Returns:
        Validated boxes
    """
    if boxes.numel() == 0:
        return boxes
    
    h, w = image_size
    
    # Clamp to image bounds
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0.0, max=float(w))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0.0, max=float(h))
    
    # Check for invalid boxes (x2 <= x1 or y2 <= y1)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid = (widths > 0) & (heights > 0)
    
    if not valid.all():
        num_invalid = (~valid).sum().item()
        print(f"warning: removing {num_invalid} invalid boxes (zero or negative area)")
        boxes = boxes[valid]
    
    return boxes


def check_gradients(
    model: torch.nn.Module,
    max_grad_norm: float = 100.0,
) -> dict[str, float]:
    """Check model gradients for issues.
    
    Args:
        model: PyTorch model
        max_grad_norm: Maximum allowed gradient norm
        
    Returns:
        Dictionary with gradient statistics
    """
    total_norm = 0.0
    num_params = 0
    num_nan = 0
    num_inf = 0
    
    for param in model.parameters():
        if param.grad is not None:
            num_params += 1
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            
            if torch.isnan(param.grad).any():
                num_nan += 1
            if torch.isinf(param.grad).any():
                num_inf += 1
    
    total_norm = total_norm ** 0.5
    
    stats = {
        "total_norm": total_norm,
        "num_params": num_params,
        "num_nan": num_nan,
        "num_inf": num_inf,
        "is_healthy": num_nan == 0 and num_inf == 0 and total_norm < max_grad_norm,
    }
    
    if not stats["is_healthy"]:
        print(f"warning: unhealthy gradients detected:")
        print(f"  total_norm={total_norm:.2f}, nan={num_nan}, inf={num_inf}")
    
    return stats


def safe_bce_loss(
    pred: Tensor,
    target: Tensor,
    reduction: str = "mean",
    eps: float = EPS,
) -> Tensor:
    """Binary cross-entropy with numerical stability.
    
    Args:
        pred: Predicted logits
        target: Target values [0, 1]
        reduction: Reduction method
        eps: Epsilon for stability
        
    Returns:
        BCE loss
    """
    # Clamp predictions to prevent log(0)
    pred = pred.clamp(min=-100, max=100)
    
    # Compute BCE
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, target, reduction="none"
    )
    
    # Sanitize
    loss = sanitize_tensor(loss, name="bce_loss")
    
    # Reduce
    if reduction == "mean":
        return safe_mean(loss)
    elif reduction == "sum":
        return safe_sum(loss)
    else:
        return loss
