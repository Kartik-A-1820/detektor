from __future__ import annotations

import base64
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from models.chimera import ChimeraODIS
from utils.visualize import draw_boxes


def resolve_device(device_name: str | None = None) -> torch.device:
    """Resolve an inference device for local serving."""
    if device_name is None or device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available on this machine")
    return torch.device(device_name)


def _detect_num_classes(checkpoint: Dict) -> int:
    """Auto-detect num_classes from checkpoint by inspecting classification head."""
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    
    for key in state_dict.keys():
        if "cls_preds" in key and "bias" in key:
            return state_dict[key].shape[0]
    return 1


def load_model(
    weights: str,
    num_classes: Optional[int] = None,
    proto_k: int = 24,
    device_name: str | None = None,
) -> Tuple[ChimeraODIS, torch.device]:
    """Load a ChimeraODIS model once for service or CLI inference."""
    device = resolve_device(device_name)
    checkpoint = torch.load(weights, map_location=device)
    
    if num_classes is None:
        num_classes = _detect_num_classes(checkpoint)
        print(f"[INFO] auto-detected num_classes={num_classes} from checkpoint")
    
    model = ChimeraODIS(num_classes=num_classes, proto_k=proto_k).to(device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model, device


def preprocess_image_bytes(image_bytes: bytes, image_size: int = 512) -> Tuple[Tensor, np.ndarray, Tuple[int, int]]:
    """Decode uploaded image bytes and prepare a model input tensor."""
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode uploaded image")

    original_height, original_width = image_bgr.shape[:2]
    resized_bgr = cv2.resize(image_bgr, (image_size, image_size))
    image_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
    return image_tensor.unsqueeze(0), cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), (original_height, original_width)


def encode_mask_to_base64_png(mask: Tensor) -> str:
    """Encode a binary mask tensor to a compact base64 PNG string."""
    mask_uint8 = mask.detach().cpu().numpy().astype(np.uint8) * 255
    success, encoded = cv2.imencode(".png", mask_uint8)
    if not success:
        raise ValueError("Failed to encode mask to PNG")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def prediction_to_service_response(
    prediction: Dict[str, Tensor],
    image_size: Tuple[int, int],
    include_masks: bool = False,
) -> Dict[str, object]:
    """Convert a model prediction dict into a JSON-friendly service response."""
    response: Dict[str, object] = {
        "num_detections": int(prediction["boxes"].shape[0]),
        "boxes": prediction["boxes"].detach().cpu().tolist(),
        "scores": [float(score) for score in prediction["scores"].detach().cpu().tolist()],
        "labels": [int(label) for label in prediction["labels"].detach().cpu().tolist()],
        "image_width": int(image_size[1]),
        "image_height": int(image_size[0]),
    }
    if include_masks:
        response["masks"] = [encode_mask_to_base64_png(mask) for mask in prediction["masks"]]
    return response


def overlay_prediction(image_rgb: np.ndarray, prediction: Dict[str, Tensor]) -> np.ndarray:
    """Render boxes and masks on an RGB image for CLI saving."""
    overlay = image_rgb.copy()
    masks = (
        prediction["masks"].detach().cpu().numpy()
        if prediction["masks"].numel() > 0
        else np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    )
    colors = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
        ],
        dtype=np.uint8,
    )
    for index, mask in enumerate(masks):
        color = colors[index % len(colors)]
        overlay[mask.astype(bool)] = (overlay[mask.astype(bool)] * 0.6 + color * 0.4).astype(np.uint8)
    overlay = draw_boxes(overlay, prediction["boxes"].detach().cpu().numpy())
    return overlay
