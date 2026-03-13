from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
from torch import Tensor

from models.factory import build_model_from_checkpoint, infer_num_classes_from_checkpoint, load_model_weights
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


def _prepare_image(source: str, image_size: int = 512) -> tuple[Tensor, "np.ndarray", tuple[int, int]]:
    """Read and preprocess an image for ChimeraODIS inference."""
    image_bgr = cv2.imread(source)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image from path: {source}")

    original_height, original_width = image_bgr.shape[:2]
    resized_bgr = cv2.resize(image_bgr, (image_size, image_size))
    image_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
    original_rgb_numpy = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_tensor.unsqueeze(0), original_rgb_numpy, (original_height, original_width)


def _detect_num_classes(checkpoint: Dict) -> int:
    """Auto-detect num_classes from checkpoint by inspecting classification head."""
    return infer_num_classes_from_checkpoint(checkpoint)


def infer(
    weights: str,
    source: str,
    num_classes: Optional[int] = None,
    image_size: int = 512,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.6,
    topk_pre_nms: int = 300,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    task: str = "segment",
) -> Dict[str, Tensor]:
    """Run ChimeraODIS inference on one image and optionally save a visualization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights, map_location=device)
    
    if num_classes is None:
        num_classes = _detect_num_classes(checkpoint)
        print(f"auto-detected num_classes={num_classes} from checkpoint")
    
    model = build_model_from_checkpoint(checkpoint, num_classes=num_classes).to(device)
    load_model_weights(model, checkpoint, strict=True)
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
        task=task,
    )[0]

    num_det = int(predictions["boxes"].shape[0])
    score_list = [round(float(score), 4) for score in predictions["scores"].detach().cpu()]
    label_list = [int(label) for label in predictions["labels"].detach().cpu()]
    
    if class_names:
        label_names = [class_names[label] for label in label_list]
        print(f"detections: {num_det}")
        print(f"labels: {label_names}")
        print(f"scores: {score_list}")
    else:
        print(f"detections: {num_det}")
        print(f"labels: {label_list}")
        print(f"scores: {score_list}")
    
    if task == "segment" and "masks" in predictions:
        print(f"mask_count: {int(predictions['masks'].shape[0])}")

    vis_image = original_rgb.copy()
    vis_image = draw_boxes(vis_image, predictions["boxes"].detach().cpu().numpy())
    vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, vis_bgr)
        print(f"saved: {save_path}")

    return predictions


def infer_folder(
    weights: str,
    source_dir: str,
    output_dir: str,
    num_classes: Optional[int] = None,
    image_size: int = 512,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.6,
    topk_pre_nms: int = 300,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    class_names: Optional[List[str]] = None,
    task: str = "segment",
) -> None:
    """Run inference on all images in a folder and save visualizations."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_files = [f for f in source_path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"no images found in {source_dir}")
        return
    
    print(f"found {len(image_files)} images in {source_dir}")
    print(f"saving results to {output_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights, map_location=device)
    
    if num_classes is None:
        num_classes = _detect_num_classes(checkpoint)
        print(f"auto-detected num_classes={num_classes} from checkpoint")
    
    model = build_model_from_checkpoint(checkpoint, num_classes=num_classes).to(device)
    load_model_weights(model, checkpoint, strict=True)
    model.eval()
    
    for idx, image_file in enumerate(image_files, start=1):
        print(f"\n[{idx}/{len(image_files)}] processing: {image_file.name}")
        
        try:
            image_tensor, original_rgb, original_size = _prepare_image(str(image_file), image_size=image_size)
            image_tensor = image_tensor.to(device)
            
            predictions = model.predict(
                image_tensor,
                original_sizes=[original_size],
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                topk_pre_nms=topk_pre_nms,
                max_det=max_det,
                mask_thresh=mask_thresh,
                task=task,
            )[0]
            
            num_det = int(predictions["boxes"].shape[0])
            label_list = [int(label) for label in predictions["labels"].detach().cpu()]
            
            if class_names:
                label_names = [class_names[label] for label in label_list]
                print(f"  detections: {num_det}, labels: {label_names}")
            else:
                print(f"  detections: {num_det}, labels: {label_list}")
            
            vis_image = original_rgb.copy()
            vis_image = draw_boxes(vis_image, predictions["boxes"].detach().cpu().numpy())
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            
            output_file = output_path / f"{image_file.stem}_pred{image_file.suffix}"
            cv2.imwrite(str(output_file), vis_bgr)
            print(f"  saved: {output_file.name}")
        
        except Exception as e:
            print(f"  error processing {image_file.name}: {e}")
            continue
    
    print(f"\ncompleted: processed {len(image_files)} images")
    print(f"results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Detektor inference on image(s)")
    parser.add_argument("--weights", type=str, default="chimera_last.pt", help="Path to model weights or checkpoint")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or folder")
    parser.add_argument("--data-yaml", type=str, default="", help="Optional dataset YAML to load class names")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (auto-detected if not provided)")
    parser.add_argument("--img-size", type=int, default=512, help="Square model input size")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold used before NMS")
    parser.add_argument("--iou-thresh", type=float, default=0.6, help="IoU threshold used by NMS")
    parser.add_argument("--topk-pre-nms", type=int, default=300, help="Maximum candidates kept before NMS")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="Threshold used to binarize predicted masks")
    parser.add_argument("--save-path", type=str, default="", help="Output path for single image or folder for batch")
    parser.add_argument("--task", type=str, default="segment", choices=["detect", "segment"], help="Task mode: 'detect' (boxes only) or 'segment' (boxes + masks)")
    args = parser.parse_args()
    
    class_names = None
    if args.data_yaml:
        with open(args.data_yaml, "r") as f:
            data_config = yaml.safe_load(f)
            class_names = data_config.get("names", None)
            print(f"loaded class names: {class_names}")
    
    source_path = Path(args.source)
    
    if source_path.is_dir():
        output_dir = args.save_path if args.save_path else "runs/inference"
        infer_folder(
            weights=args.weights,
            source_dir=args.source,
            output_dir=output_dir,
            num_classes=args.num_classes,
            image_size=args.img_size,
            conf_thresh=args.conf_thresh,
            iou_thresh=args.iou_thresh,
            topk_pre_nms=args.topk_pre_nms,
            max_det=args.max_det,
            mask_thresh=args.mask_thresh,
            class_names=class_names,
            task=args.task,
        )
    else:
        save_path = args.save_path if args.save_path else f"runs/inference/{source_path.stem}_pred{source_path.suffix}"
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
            save_path=save_path,
            class_names=class_names,
            task=args.task,
        )
