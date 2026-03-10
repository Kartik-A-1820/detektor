from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import torch
from torch import Tensor

from models.chimera import ChimeraODIS
from utils.visualize import draw_boxes


def _overlay_masks(image: Tensor | None, masks: Tensor) -> Tensor | None:
    """Overlay binary masks on an RGB image tensor in-place-friendly form."""
    if image is None:
        return None
    if masks.numel() == 0:
        return image

    overlay = image.clone()
    colors = torch.tensor(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
        ],
        device=overlay.device,
        dtype=overlay.dtype,
    )

    for index, mask in enumerate(masks):
        color = colors[index % colors.shape[0]].view(1, 1, 3)
        mask_3d = mask.unsqueeze(-1).to(dtype=overlay.dtype)
        overlay = torch.where(mask_3d > 0, overlay * 0.6 + color * 0.4, overlay)
    return overlay


def _prepare_image(source: str, image_size: int = 512) -> tuple[Tensor, Tensor, tuple[int, int]]:
    """Read and preprocess an image for ChimeraODIS inference."""
    image_bgr = cv2.imread(source)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image from path: {source}")

    original_height, original_width = image_bgr.shape[:2]
    resized_bgr = cv2.resize(image_bgr, (image_size, image_size))
    image_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
    return image_tensor.unsqueeze(0), torch.from_numpy(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)), (original_height, original_width)


def infer(
    weights: str,
    source: str,
    num_classes: int = 1,
    image_size: int = 512,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.6,
    topk_pre_nms: int = 300,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    save_path: Optional[str] = None,
) -> Dict[str, Tensor]:
    """Run ChimeraODIS inference on one image and optionally save a visualization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChimeraODIS(num_classes=num_classes).to(device)
    checkpoint = torch.load(weights, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()

    image_tensor, original_rgb, original_size = _prepare_image(source, image_size=image_size)
    image_tensor = image_tensor.to(device)

    predictions = model.predict(
        image_tensor,
        original_sizes=[original_size],
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        topk_pre_nms=topk_pre_nms,
        max_det=max_det,
        mask_thresh=mask_thresh,
    )[0]

    num_det = int(predictions["boxes"].shape[0])
    score_list = [round(float(score), 4) for score in predictions["scores"].detach().cpu()]
    label_list = [int(label) for label in predictions["labels"].detach().cpu()]
    print(f"detections: {num_det}")
    print(f"labels: {label_list}")
    print(f"scores: {score_list}")
    print(f"mask_count: {int(predictions['masks'].shape[0])}")

    vis_tensor = original_rgb.to(device=device, dtype=torch.float32)
    vis_tensor = _overlay_masks(vis_tensor, predictions["masks"].to(device=device))
    if vis_tensor is not None:
        vis_image = vis_tensor.clamp(0, 255).byte().cpu().numpy()
        vis_image = draw_boxes(vis_image, predictions["boxes"].detach().cpu().numpy())
        vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, vis_bgr)
            print(f"saved: {save_path}")

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single-image Detektor inference")
    parser.add_argument("--weights", type=str, default="chimera_last.pt", help="Path to model weights or checkpoint")
    parser.add_argument("--source", type=str, required=True, help="Path to the input image")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes expected by the checkpoint")
    parser.add_argument("--img-size", type=int, default=512, help="Square model input size")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold used before NMS")
    parser.add_argument("--iou-thresh", type=float, default=0.6, help="IoU threshold used by NMS")
    parser.add_argument("--topk-pre-nms", type=int, default=300, help="Maximum candidates kept before NMS")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="Threshold used to binarize predicted masks")
    parser.add_argument("--save-path", type=str, default="", help="Optional output path for a visualization image")
    args = parser.parse_args()

    infer(
        weights=args.weights,
        source=args.source,
        num_classes=args.num_classes,
        image_size=args.img_size,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        topk_pre_nms=args.topk_pre_nms,
        max_det=args.max_det,
        mask_thresh=args.mask_thresh,
        save_path=args.save_path or None,
    )
