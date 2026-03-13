from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from api.utils import load_model, overlay_prediction, prediction_to_service_response, preprocess_image_bytes


def main() -> None:
    """Run single-image local inference and optionally save an overlay image."""
    parser = argparse.ArgumentParser(description="Run local Detektor CLI prediction on a single image")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights or checkpoint")
    parser.add_argument("--source", type=str, required=True, help="Path to the input image")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto, cpu, or cuda")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold used before NMS")
    parser.add_argument("--iou-thresh", type=float, default=0.6, help="IoU threshold used by NMS")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image")
    parser.add_argument("--topk-pre-nms", type=int, default=300, help="Maximum candidates kept before NMS")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="Threshold used to binarize predicted masks")
    parser.add_argument("--save-path", type=str, default="", help="Optional output path for a visualization image")
    parser.add_argument("--include-masks", action="store_true", help="Include base64-encoded masks in the printed response payload")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes expected by the checkpoint")
    parser.add_argument("--proto-k", type=int, default=24, help="Number of prototype channels expected by the checkpoint")
    parser.add_argument("--img-size", type=int, default=512, help="Square model input size used for preprocessing")
    args = parser.parse_args()

    model, device = load_model(
        weights=args.weights,
        num_classes=args.num_classes,
        proto_k=args.proto_k,
        device_name=args.device,
    )

    image_bytes = Path(args.source).read_bytes()
    image_tensor, original_rgb, original_size = preprocess_image_bytes(image_bytes, image_size=args.img_size)
    image_tensor = image_tensor.to(device)

    prediction = model.predict(
        image_tensor,
        original_sizes=[original_size],
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        topk_pre_nms=args.topk_pre_nms,
        max_det=args.max_det,
        mask_thresh=args.mask_thresh,
    )[0]

    payload = prediction_to_service_response(
        prediction=prediction,
        image_size=original_size,
        include_masks=args.include_masks,
    )
    print(f"detections: {payload['num_detections']}")
    print(f"labels: {payload['labels']}")
    print(f"scores: {[round(score, 4) for score in payload['scores']]}")

    if args.save_path:
        overlay = overlay_prediction(original_rgb, prediction)
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
