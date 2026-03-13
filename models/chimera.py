from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

from losses import DetectionLoss, SegmentationLoss
from .blocks import ChimeraBackbone
from .heads import DetectionHead, PrototypeMaskHead
from .neck import ChimeraPANNeck
from utils.anchors import concatenate_points_and_strides, generate_points_for_features
from utils.box_ops import distances_to_boxes, flatten_prediction_levels
from utils.mask_ops import (
    boxes_to_proto_coordinates,
    compose_instance_masks,
    crop_mask_region,
    flatten_mask_coefficients,
    threshold_masks,
    upsample_masks_to_image,
    resize_instance_masks,
)
from utils.postprocess import build_empty_prediction, class_aware_nms, rescale_boxes, select_topk_candidates


class ChimeraODIS(nn.Module):
    """Lightweight multi-task architecture foundation for detection and instance segmentation."""

    def __init__(
        self,
        num_classes: int = 1,
        proto_k: int = 24,
        stem_channels: int = 32,
        backbone_channels: Tuple[int, int, int, int] = (64, 128, 192, 256),
        backbone_depths: Tuple[int, int, int, int] = (1, 2, 2, 2),
        neck_channels: Tuple[int, int, int] = (128, 192, 256),
        head_feat_channels: int = 128,
        mask_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.proto_k = proto_k
        self.backbone = ChimeraBackbone(
            stem_channels=stem_channels,
            stage_channels=backbone_channels,
            stage_depths=backbone_depths,
        )
        self.neck = ChimeraPANNeck(
            in_channels=(backbone_channels[1], backbone_channels[2], backbone_channels[3]),
            out_channels=neck_channels,
        )
        self.detect_head = DetectionHead(
            num_classes=num_classes,
            in_channels=neck_channels,
            feat_channels=head_feat_channels,
            num_mask_coeffs=proto_k,
        )
        self.proto_head = PrototypeMaskHead(
            in_channels=neck_channels[0],
            hidden_channels=max(neck_channels[0] // 2, 64),
            proto_k=proto_k,
        )
        self.detection_loss = DetectionLoss(
            num_classes=num_classes,
            label_smoothing=0.0,  # Can be tuned for stability (0.0-0.1)
        )
        self.segmentation_loss = SegmentationLoss()
        self.mask_weight = mask_weight

    def forward(self, x: Tensor) -> Dict[str, Any]:
        """Run the backbone, neck, and heads and return structured raw outputs."""
        if x.ndim != 4:
            raise AssertionError(f"Expected input tensor of shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise AssertionError(f"Expected input with 3 channels, got {x.shape[1]}")

        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        det_outputs = self.detect_head((n3, n4, n5))
        proto = self.proto_head(n3)

        return {
            "features": {
                "p3": p3,
                "p4": p4,
                "p5": p5,
                "n3": n3,
                "n4": n4,
                "n5": n5,
            },
            "det": det_outputs,
            "proto": proto,
        }

    def _normalize_targets(
        self,
        imgs: Tensor,
        targets: Any,
    ) -> List[Dict[str, Tensor]]:
        """Normalize collated or list-style targets into a per-image target list."""
        image_height, image_width = imgs.shape[-2:]
        device = imgs.device
        dtype = imgs.dtype

        normalized_targets: List[Dict[str, Tensor]] = []
        batch_size = imgs.shape[0]

        if isinstance(targets, Mapping):
            boxes = targets.get("boxes")
            labels = targets.get("labels")
            masks = targets.get("masks")
            if boxes is None or labels is None:
                raise AssertionError("Targets mapping must contain 'boxes' and 'labels'")
            if not isinstance(boxes, Tensor) or not isinstance(labels, Tensor):
                raise AssertionError("Collated targets must provide tensor boxes and labels")
            if boxes.ndim == 2:
                boxes = boxes.unsqueeze(0)
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)
            if boxes.shape[0] != batch_size or labels.shape[0] != batch_size:
                raise AssertionError("Collated target batch size does not match images")
            if masks is not None and isinstance(masks, Tensor) and masks.ndim == 3:
                masks = masks.unsqueeze(0)
            iterable_targets: Sequence[Dict[str, Tensor]] = [
                {
                    "boxes": boxes[index],
                    "labels": labels[index],
                    "masks": masks[index] if isinstance(masks, Tensor) and masks.shape[0] == batch_size else None,
                }
                for index in range(batch_size)
            ]
        else:
            iterable_targets = list(targets)
            if len(iterable_targets) != batch_size:
                raise AssertionError("Number of targets must match batch size")

        scale = torch.tensor(
            [float(image_width), float(image_height), float(image_width), float(image_height)],
            device=device,
            dtype=dtype,
        )

        for target in iterable_targets:
            boxes = target.get("boxes")
            labels = target.get("labels")
            if boxes is None or labels is None:
                raise AssertionError("Each target must contain 'boxes' and 'labels'")
            masks = target.get("masks")

            boxes = boxes.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.long)
            if isinstance(masks, Tensor):
                masks = masks.to(device=device)
                if masks.ndim == 2:
                    masks = masks.unsqueeze(0)
            else:
                masks = torch.zeros((boxes.shape[0], image_height, image_width), device=device, dtype=torch.float32)

            if boxes.numel() == 0:
                abs_boxes = boxes.reshape(0, 4)
            else:
                abs_boxes = boxes.reshape(-1, 4) * scale
                abs_boxes[:, 0::2] = abs_boxes[:, 0::2].clamp(min=0.0, max=float(image_width))
                abs_boxes[:, 1::2] = abs_boxes[:, 1::2].clamp(min=0.0, max=float(image_height))

            normalized_targets.append({"boxes": abs_boxes, "labels": labels.reshape(-1), "masks": masks})
        return normalized_targets

    def compute_loss(
        self,
        imgs: Tensor,
        targets: Sequence[Any],
        return_dict: bool = False,
        debug: bool = False,
        task: str = "segment",
    ) -> Tensor | Dict[str, Tensor]:
        """Compute detection loss from raw model outputs and normalized xyxy targets."""
        outputs = self.forward(imgs)
        n3 = outputs["features"]["n3"]
        n4 = outputs["features"]["n4"]
        n5 = outputs["features"]["n5"]

        cls_levels = outputs["det"]["cls"]
        box_levels = outputs["det"]["box"]
        obj_levels = outputs["det"]["obj"]
        mask_coeff_levels = outputs["det"]["mask_coeff"]
        proto = outputs["proto"]

        points_per_level, strides_per_level = generate_points_for_features(
            feature_maps=(n3, n4, n5),
            image_size=(imgs.shape[-2], imgs.shape[-1]),
        )
        points, strides = concatenate_points_and_strides(points_per_level, strides_per_level)

        flat_cls = flatten_prediction_levels(cls_levels)
        flat_box = flatten_prediction_levels(box_levels)
        flat_obj = flatten_prediction_levels(obj_levels)
        flat_mask_coeff = flatten_mask_coefficients(mask_coeff_levels)
        decoded_boxes = distances_to_boxes(points, flat_box)

        prepared_targets = self._normalize_targets(imgs, targets)
        det_loss_dict = self.detection_loss(
            pred_cls=flat_cls,
            pred_box=flat_box,
            pred_obj=flat_obj,
            decoded_boxes=decoded_boxes,
            points=points,
            strides=strides,
            targets=prepared_targets,
        )
        
        # Task-aware mask loss computation
        if task == "detect":
            # Detection only - skip mask loss
            zero = proto.new_zeros(())
            mask_loss_dict = {
                "loss_mask_bce": zero,
                "loss_mask_dice": zero,
                "loss_mask": zero,
                "num_mask_pos": zero,
            }
            loss_total = det_loss_dict["loss_total"]
        else:
            # Segmentation mode - compute mask loss
            mask_loss_dict = self._compute_mask_losses(
                prototypes=proto,
                mask_coefficients=flat_mask_coeff,
                assignments=det_loss_dict["assignments"],
                targets=prepared_targets,
                image_size=(imgs.shape[-2], imgs.shape[-1]),
            )
            loss_total = det_loss_dict["loss_total"] + self.mask_weight * mask_loss_dict["loss_mask"]
        
        loss_dict = {
            "loss_total": loss_total,
            "loss_cls": det_loss_dict["loss_cls"],
            "loss_box": det_loss_dict["loss_box"],
            "loss_obj": det_loss_dict["loss_obj"],
            "loss_mask": mask_loss_dict["loss_mask"],
            "loss_mask_bce": mask_loss_dict["loss_mask_bce"],
            "loss_mask_dice": mask_loss_dict["loss_mask_dice"],
            "num_fg": det_loss_dict["num_fg"],
            "num_mask_pos": mask_loss_dict["num_mask_pos"],
        }

        if debug or not torch.isfinite(loss_total):
            non_finite_components = []
            for name, value in loss_dict.items():
                if isinstance(value, Tensor) and not torch.isfinite(value).all():
                    non_finite_components.append(f"{name}={float(value.detach()):.6f}")
            if non_finite_components:
                print(f"non-finite loss components: {', '.join(non_finite_components)}")
                print(f"  num_fg={float(loss_dict['num_fg'])}, num_mask_pos={float(loss_dict['num_mask_pos'])}")

        if return_dict:
            return loss_dict
        return loss_dict["loss_total"]

    def forward_export(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run an ONNX-friendly tensor-only export forward path.

        Input:
            ``x``: image tensor with shape ``[B, 3, H, W]``.

        Returns:
            A tensor tuple containing:
            - ``cls_flat``: ``[B, total_points, num_classes]``
            - ``box_flat``: ``[B, total_points, 4]``
            - ``obj_flat``: ``[B, total_points, 1]``
            - ``mask_coeff_flat``: ``[B, total_points, proto_k]``
            - ``proto``: ``[B, proto_k, Hp, Wp]``

        Postprocessing such as decoding, NMS, confidence thresholding, and mask
        reconstruction remains outside the exported graph.
        """
        outputs = self.forward(x)
        cls_levels = outputs["det"]["cls"]
        box_levels = outputs["det"]["box"]
        obj_levels = outputs["det"]["obj"]
        mask_coeff_levels = outputs["det"]["mask_coeff"]
        proto = outputs["proto"]

        cls_flat = flatten_prediction_levels(cls_levels)
        box_flat = flatten_prediction_levels(box_levels)
        obj_flat = flatten_prediction_levels(obj_levels)
        mask_coeff_flat = flatten_mask_coefficients(mask_coeff_levels)
        return cls_flat, box_flat, obj_flat, mask_coeff_flat, proto

    @torch.no_grad()
    def predict(
        self,
        imgs: Tensor,
        original_sizes: Sequence[Tuple[int, int]] | None = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.6,
        topk_pre_nms: int = 300,
        max_det: int = 100,
        mask_thresh: float = 0.5,
        return_mask_probs: bool = False,
        task: str = "segment",
    ) -> List[Dict[str, Tensor]]:
        """Run lightweight inference postprocessing for detection and instance segmentation.

        For efficiency, multi-class scoring keeps only the best class per point before
        thresholding, top-k filtering, and class-aware NMS.
        """
        outputs = self.forward(imgs)
        n3 = outputs["features"]["n3"]
        n4 = outputs["features"]["n4"]
        n5 = outputs["features"]["n5"]

        cls_levels = outputs["det"]["cls"]
        box_levels = outputs["det"]["box"]
        obj_levels = outputs["det"]["obj"]
        mask_coeff_levels = outputs["det"]["mask_coeff"]
        proto = outputs["proto"]

        points_per_level, strides_per_level = generate_points_for_features(
            feature_maps=(n3, n4, n5),
            image_size=(imgs.shape[-2], imgs.shape[-1]),
        )
        points, _ = concatenate_points_and_strides(points_per_level, strides_per_level)

        flat_cls = flatten_prediction_levels(cls_levels)
        flat_box = flatten_prediction_levels(box_levels)
        flat_obj = flatten_prediction_levels(obj_levels)
        flat_mask_coeff = flatten_mask_coefficients(mask_coeff_levels)
        decoded_boxes = distances_to_boxes(points, flat_box)

        cls_prob = flat_cls.sigmoid()
        obj_prob = flat_obj.sigmoid().squeeze(-1)
        scores_all = cls_prob * obj_prob.unsqueeze(-1)
        scores, labels = scores_all.max(dim=-1)

        model_image_size = (imgs.shape[-2], imgs.shape[-1])
        if original_sizes is None:
            original_sizes = [model_image_size for _ in range(imgs.shape[0])]
        if len(original_sizes) != imgs.shape[0]:
            raise AssertionError("original_sizes length must match batch size")

        predictions: List[Dict[str, Tensor]] = []
        for batch_index in range(imgs.shape[0]):
            candidate_mask = scores[batch_index] >= conf_thresh
            if not candidate_mask.any():
                predictions.append(
                    build_empty_prediction(
                        image_size=original_sizes[batch_index],
                        device=imgs.device,
                        score_dtype=imgs.dtype,
                        mask_dtype=torch.bool if not return_mask_probs else imgs.dtype,
                    )
                )
                continue

            boxes_i = decoded_boxes[batch_index, candidate_mask]
            scores_i = scores[batch_index, candidate_mask]
            labels_i = labels[batch_index, candidate_mask]
            mask_coeff_i = flat_mask_coeff[batch_index, candidate_mask]

            boxes_i, scores_i, labels_i, mask_coeff_i = select_topk_candidates(
                boxes=boxes_i,
                scores=scores_i,
                labels=labels_i,
                mask_coeff=mask_coeff_i,
                topk_pre_nms=topk_pre_nms,
            )
            keep = class_aware_nms(
                boxes=boxes_i,
                scores=scores_i,
                labels=labels_i,
                iou_thresh=iou_thresh,
                max_det=max_det,
            )

            if keep.numel() == 0:
                predictions.append(
                    build_empty_prediction(
                        image_size=original_sizes[batch_index],
                        device=imgs.device,
                        score_dtype=imgs.dtype,
                        mask_dtype=torch.bool if not return_mask_probs else imgs.dtype,
                    )
                )
                continue

            kept_boxes = boxes_i[keep]
            kept_scores = scores_i[keep]
            kept_labels = labels_i[keep]
            kept_coeff = mask_coeff_i[keep]

            scaled_boxes = rescale_boxes(
                boxes=kept_boxes,
                from_size=model_image_size,
                to_size=original_sizes[batch_index],
            )
            
            # Task-aware mask generation
            result = {
                "boxes": scaled_boxes,
                "scores": kept_scores,
                "labels": kept_labels,
            }
            
            if task == "segment":
                # Generate masks for segmentation mode
                proto_per_image = proto[batch_index]
                proto_size = (proto_per_image.shape[-2], proto_per_image.shape[-1])
                mask_logits = compose_instance_masks(proto_per_image, kept_coeff)
                proto_boxes = boxes_to_proto_coordinates(
                    kept_boxes,
                    image_size=model_image_size,
                    proto_size=proto_size,
                )
                mask_logits = crop_mask_region(mask_logits, proto_boxes)
                mask_probs = upsample_masks_to_image(mask_logits.sigmoid(), model_image_size)
                masks = (
                    mask_probs
                    if return_mask_probs
                    else threshold_masks(mask_probs, threshold=mask_thresh)
                )

                if original_sizes[batch_index] != model_image_size:
                    masks = upsample_masks_to_image(
                        masks.to(dtype=imgs.dtype),
                        image_size=original_sizes[batch_index],
                    )
                    if not return_mask_probs:
                        masks = masks >= 0.5
                
                result["masks"] = masks

            predictions.append(result)
        return predictions

    def _compute_mask_losses(
        self,
        prototypes: Tensor,
        mask_coefficients: Tensor,
        assignments: Sequence[Dict[str, Tensor]],
        targets: Sequence[Dict[str, Tensor]],
        image_size: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        """Compute prototype-mask segmentation losses on positive assignments only."""
        batch_pred_masks: List[Tensor] = []
        batch_gt_masks: List[Tensor] = []
        total_mask_pos = prototypes.new_tensor(0.0)
        proto_size = (prototypes.shape[-2], prototypes.shape[-1])

        for batch_index, assignment in enumerate(assignments):
            fg_mask = assignment["fg_mask"]
            if not fg_mask.any():
                continue

            matched_gt_indices = assignment["matched_gt_indices"][fg_mask]
            valid = matched_gt_indices >= 0
            if not valid.any():
                continue

            pos_coeffs = mask_coefficients[batch_index, fg_mask][valid]
            assigned_boxes = assignment["assigned_boxes"][fg_mask][valid]
            prototypes_per_image = prototypes[batch_index]
            pred_mask_logits = compose_instance_masks(prototypes_per_image, pos_coeffs)

            gt_masks = targets[batch_index]["masks"]
            if gt_masks.numel() == 0:
                continue
            gt_masks = gt_masks[matched_gt_indices[valid]]
            gt_masks = resize_instance_masks(gt_masks, proto_size)

            proto_boxes = boxes_to_proto_coordinates(assigned_boxes, image_size=image_size, proto_size=proto_size)
            pred_mask_logits = crop_mask_region(pred_mask_logits, proto_boxes)
            gt_masks = crop_mask_region(gt_masks, proto_boxes)

            batch_pred_masks.append(pred_mask_logits)
            batch_gt_masks.append(gt_masks)
            total_mask_pos = total_mask_pos + valid.sum().to(dtype=prototypes.dtype)

        if batch_pred_masks:
            pred_masks = torch.cat(batch_pred_masks, dim=0)
            target_masks = torch.cat(batch_gt_masks, dim=0)
            loss_dict = self.segmentation_loss(pred_masks, target_masks)
        else:
            zero = prototypes.new_zeros(())
            loss_dict = {
                "loss_mask_bce": zero,
                "loss_mask_dice": zero,
                "loss_mask": zero,
            }
        loss_dict["num_mask_pos"] = total_mask_pos
        return loss_dict


def smoke_test() -> Dict[str, Any]:
    """Run a lightweight CPU smoke test and print output tensor shapes."""
    model = ChimeraODIS(num_classes=1, proto_k=24)
    model.eval()

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        outputs = model(x)

    expected_keys = {"features", "det", "proto"}
    if set(outputs.keys()) != expected_keys:
        raise AssertionError(f"Unexpected top-level keys: {outputs.keys()}")

    print("features:")
    for name, tensor in outputs["features"].items():
        print(f"  {name}: {tuple(tensor.shape)}")

    print("det:")
    for name, tensors in outputs["det"].items():
        shapes = [tuple(tensor.shape) for tensor in tensors]
        print(f"  {name}: {shapes}")

    print(f"proto: {tuple(outputs['proto'].shape)}")
    return outputs


def smoke_test_loss() -> Dict[str, Tensor]:
    """Run a lightweight CPU smoke test for detection and mask loss computation."""
    model = ChimeraODIS(num_classes=1, proto_k=24)
    model.train()

    imgs = torch.randn(2, 3, 512, 512)
    mask0_a = torch.zeros((512, 512), dtype=torch.float32)
    mask0_a[64:215, 52:184] = 1.0
    mask0_b = torch.zeros((512, 512), dtype=torch.float32)
    mask0_b[246:451, 281:420] = 1.0
    mask1_a = torch.zeros((512, 512), dtype=torch.float32)
    mask1_a[102:297, 112:225] = 1.0
    targets = [
        {
            "boxes": torch.tensor([[0.10, 0.12, 0.36, 0.42], [0.55, 0.48, 0.82, 0.88]], dtype=torch.float32),
            "labels": torch.tensor([0, 0], dtype=torch.long),
            "masks": torch.stack((mask0_a, mask0_b), dim=0),
        },
        {
            "boxes": torch.tensor([[0.22, 0.20, 0.44, 0.58]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
            "masks": mask1_a.unsqueeze(0),
        },
    ]

    loss_dict = model.compute_loss(imgs, targets, return_dict=True)
    print(f"loss_total: {float(loss_dict['loss_total'].detach())}")
    print(f"loss_cls: {float(loss_dict['loss_cls'].detach())}")
    print(f"loss_box: {float(loss_dict['loss_box'].detach())}")
    print(f"loss_obj: {float(loss_dict['loss_obj'].detach())}")
    print(f"loss_mask: {float(loss_dict['loss_mask'].detach())}")
    print(f"loss_mask_bce: {float(loss_dict['loss_mask_bce'].detach())}")
    print(f"loss_mask_dice: {float(loss_dict['loss_mask_dice'].detach())}")
    print(f"num_fg: {float(loss_dict['num_fg'].detach())}")
    print(f"num_mask_pos: {float(loss_dict['num_mask_pos'].detach())}")
    return loss_dict


def smoke_test_predict() -> List[Dict[str, Tensor]]:
    """Run a lightweight CPU smoke test for inference postprocessing."""
    model = ChimeraODIS(num_classes=1, proto_k=24)
    model.eval()

    imgs = torch.randn(1, 3, 512, 512)
    outputs = model.predict(imgs, original_sizes=[(512, 512)])
    if len(outputs) != 1:
        raise AssertionError(f"Expected one prediction dict, got {len(outputs)}")

    prediction = outputs[0]
    print(f"predict keys: {list(prediction.keys())}")
    print(f"boxes: {tuple(prediction['boxes'].shape)}")
    print(f"scores: {tuple(prediction['scores'].shape)}")
    print(f"labels: {tuple(prediction['labels'].shape)}")
    print(f"masks: {tuple(prediction['masks'].shape)}")
    return outputs


def smoke_test_export() -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run a lightweight CPU smoke test for export-forward outputs."""
    model = ChimeraODIS(num_classes=1, proto_k=24)
    model.eval()

    imgs = torch.randn(1, 3, 512, 512)
    outputs = model.forward_export(imgs)
    names = ("cls_flat", "box_flat", "obj_flat", "mask_coeff_flat", "proto")
    for name, tensor in zip(names, outputs):
        print(f"{name}: {tuple(tensor.shape)}")
    return outputs


if __name__ == "__main__":
    smoke_test()
    smoke_test_loss()
    smoke_test_predict()
    smoke_test_export()
