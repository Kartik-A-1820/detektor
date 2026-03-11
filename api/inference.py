"""Inference service wrapper for production API."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from api.logging_utils import RequestTimer
from api.utils import encode_mask_to_base64_png, preprocess_image_bytes
from models.chimera import ChimeraODIS


class InferenceService:
    """Production-ready inference service wrapper."""
    
    def __init__(
        self,
        model: ChimeraODIS,
        device: torch.device,
        image_size: int = 512,
        default_conf_thresh: float = 0.25,
        default_iou_thresh: float = 0.6,
        default_max_det: int = 100,
        default_topk_pre_nms: int = 300,
        default_mask_thresh: float = 0.5,
        default_include_masks: bool = False,
    ) -> None:
        """Initialize the inference service.
        
        Args:
            model: Loaded ChimeraODIS model
            device: Inference device
            image_size: Input image size for model
            default_conf_thresh: Default confidence threshold
            default_iou_thresh: Default IoU threshold
            default_max_det: Default max detections
            default_topk_pre_nms: Default top-k before NMS
            default_mask_thresh: Default mask threshold
            default_include_masks: Default include masks flag
        """
        self.model = model
        self.device = device
        self.image_size = image_size
        self.default_conf_thresh = default_conf_thresh
        self.default_iou_thresh = default_iou_thresh
        self.default_max_det = default_max_det
        self.default_topk_pre_nms = default_topk_pre_nms
        self.default_mask_thresh = default_mask_thresh
        self.default_include_masks = default_include_masks
        self.num_classes = model.num_classes
    
    def warmup(self, num_iterations: int = 3) -> None:
        """Warmup the model with dummy inputs.
        
        Args:
            num_iterations: Number of warmup iterations
        """
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model.predict(
                    dummy_input,
                    original_sizes=[(self.image_size, self.image_size)],
                    conf_thresh=self.default_conf_thresh,
                    iou_thresh=self.default_iou_thresh,
                    topk_pre_nms=self.default_topk_pre_nms,
                    max_det=self.default_max_det,
                    mask_thresh=self.default_mask_thresh,
                )
    
    def predict_single(
        self,
        image_bytes: bytes,
        conf_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
        max_det: Optional[int] = None,
        include_masks: Optional[bool] = None,
    ) -> Tuple[Dict[str, object], float]:
        """Run inference on a single image.
        
        Args:
            image_bytes: Raw image bytes
            conf_thresh: Confidence threshold (uses default if None)
            iou_thresh: IoU threshold (uses default if None)
            max_det: Max detections (uses default if None)
            include_masks: Include masks flag (uses default if None)
            
        Returns:
            Tuple of (prediction dict, inference_time_ms)
        """
        # Use defaults if not provided
        conf_thresh = conf_thresh if conf_thresh is not None else self.default_conf_thresh
        iou_thresh = iou_thresh if iou_thresh is not None else self.default_iou_thresh
        max_det = max_det if max_det is not None else self.default_max_det
        include_masks = include_masks if include_masks is not None else self.default_include_masks
        
        # Preprocess image
        image_tensor, _, original_size = preprocess_image_bytes(image_bytes, self.image_size)
        image_tensor = image_tensor.to(self.device)
        
        # Run inference with timing
        with RequestTimer() as timer:
            with torch.no_grad():
                prediction = self.model.predict(
                    image_tensor,
                    original_sizes=[original_size],
                    conf_thresh=conf_thresh,
                    iou_thresh=iou_thresh,
                    topk_pre_nms=self.default_topk_pre_nms,
                    max_det=max_det,
                    mask_thresh=self.default_mask_thresh,
                )[0]
        
        inference_time_ms = timer.duration_ms
        
        # Convert to response format
        response = self._prediction_to_response(
            prediction=prediction,
            image_size=original_size,
            include_masks=include_masks,
            inference_time_ms=inference_time_ms,
        )
        
        return response, inference_time_ms
    
    def predict_batch(
        self,
        images_bytes: List[bytes],
        conf_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
        max_det: Optional[int] = None,
        include_masks: Optional[bool] = None,
    ) -> Tuple[List[Dict[str, object]], float]:
        """Run inference on a batch of images.
        
        Args:
            images_bytes: List of raw image bytes
            conf_thresh: Confidence threshold (uses default if None)
            iou_thresh: IoU threshold (uses default if None)
            max_det: Max detections (uses default if None)
            include_masks: Include masks flag (uses default if None)
            
        Returns:
            Tuple of (list of prediction dicts, total_inference_time_ms)
        """
        # Use defaults if not provided
        conf_thresh = conf_thresh if conf_thresh is not None else self.default_conf_thresh
        iou_thresh = iou_thresh if iou_thresh is not None else self.default_iou_thresh
        max_det = max_det if max_det is not None else self.default_max_det
        include_masks = include_masks if include_masks is not None else self.default_include_masks
        
        # Preprocess all images
        tensors = []
        original_sizes = []
        
        for img_bytes in images_bytes:
            tensor, _, orig_size = preprocess_image_bytes(img_bytes, self.image_size)
            tensors.append(tensor)
            original_sizes.append(orig_size)
        
        # Stack into batch
        batch_tensor = torch.cat(tensors, dim=0).to(self.device)
        
        # Run inference with timing
        with RequestTimer() as timer:
            with torch.no_grad():
                predictions = self.model.predict(
                    batch_tensor,
                    original_sizes=original_sizes,
                    conf_thresh=conf_thresh,
                    iou_thresh=iou_thresh,
                    topk_pre_nms=self.default_topk_pre_nms,
                    max_det=max_det,
                    mask_thresh=self.default_mask_thresh,
                )
        
        total_inference_time_ms = timer.duration_ms
        
        # Convert all predictions to response format
        responses = []
        for pred, orig_size in zip(predictions, original_sizes):
            response = self._prediction_to_response(
                prediction=pred,
                image_size=orig_size,
                include_masks=include_masks,
                inference_time_ms=None,  # Individual timing not available in batch
            )
            responses.append(response)
        
        return responses, total_inference_time_ms
    
    def _prediction_to_response(
        self,
        prediction: Dict[str, Tensor],
        image_size: Tuple[int, int],
        include_masks: bool,
        inference_time_ms: Optional[float] = None,
    ) -> Dict[str, object]:
        """Convert model prediction to API response format.
        
        Args:
            prediction: Model prediction dict
            image_size: Original image size (height, width)
            include_masks: Whether to include masks
            inference_time_ms: Inference time in milliseconds
            
        Returns:
            Response dictionary
        """
        boxes = prediction["boxes"].detach().cpu().tolist()
        scores = [float(s) for s in prediction["scores"].detach().cpu().tolist()]
        labels = [int(l) for l in prediction["labels"].detach().cpu().tolist()]
        
        # Build detections list (new format)
        detections = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            detection = {
                "box": box,
                "score": score,
                "label": label,
            }
            if include_masks and prediction["masks"].numel() > 0:
                detection["mask"] = encode_mask_to_base64_png(prediction["masks"][i])
            detections.append(detection)
        
        # Build response with both new and legacy formats
        response: Dict[str, object] = {
            "num_detections": len(boxes),
            "detections": detections,
            "image_width": int(image_size[1]),
            "image_height": int(image_size[0]),
            "inference_time_ms": inference_time_ms,
            # Legacy fields for backward compatibility
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
        
        if include_masks and prediction["masks"].numel() > 0:
            response["masks"] = [encode_mask_to_base64_png(m) for m in prediction["masks"]]
        
        return response
