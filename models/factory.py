from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch

from .chimera import ChimeraODIS


@dataclass(frozen=True)
class ArchitectureProfile:
    tier_key: str
    display_name: str
    stem_channels: int
    backbone_channels: tuple[int, int, int, int]
    backbone_depths: tuple[int, int, int, int]
    neck_channels: tuple[int, int, int]
    head_feat_channels: int
    proto_k: int


ARCHITECTURE_PROFILES: Dict[str, ArchitectureProfile] = {
    "firefly": ArchitectureProfile(
        tier_key="firefly",
        display_name="Firefly",
        stem_channels=16,
        backbone_channels=(24, 48, 72, 96),
        backbone_depths=(1, 1, 1, 1),
        neck_channels=(48, 72, 96),
        head_feat_channels=48,
        proto_k=16,
    ),
    "comet": ArchitectureProfile(
        tier_key="comet",
        display_name="Comet",
        stem_channels=24,
        backbone_channels=(32, 64, 96, 128),
        backbone_depths=(1, 1, 2, 2),
        neck_channels=(64, 96, 128),
        head_feat_channels=64,
        proto_k=20,
    ),
    "nova": ArchitectureProfile(
        tier_key="nova",
        display_name="Nova",
        stem_channels=24,
        backbone_channels=(48, 96, 128, 160),
        backbone_depths=(1, 2, 2, 2),
        neck_channels=(96, 128, 160),
        head_feat_channels=96,
        proto_k=24,
    ),
    "pulsar": ArchitectureProfile(
        tier_key="pulsar",
        display_name="Pulsar",
        stem_channels=32,
        backbone_channels=(64, 128, 192, 224),
        backbone_depths=(1, 2, 2, 3),
        neck_channels=(128, 192, 224),
        head_feat_channels=128,
        proto_k=24,
    ),
    "quasar": ArchitectureProfile(
        tier_key="quasar",
        display_name="Quasar",
        stem_channels=32,
        backbone_channels=(64, 128, 192, 256),
        backbone_depths=(1, 2, 2, 2),
        neck_channels=(128, 192, 256),
        head_feat_channels=128,
        proto_k=24,
    ),
    "supernova": ArchitectureProfile(
        tier_key="supernova",
        display_name="Supernova",
        stem_channels=40,
        backbone_channels=(80, 160, 224, 320),
        backbone_depths=(2, 3, 3, 3),
        neck_channels=(160, 224, 320),
        head_feat_channels=160,
        proto_k=32,
    ),
}


def infer_num_classes_from_checkpoint(checkpoint: Dict[str, Any]) -> int:
    """Infer class count from a checkpoint payload or state dict."""
    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, Mapping):
        return 1
    for key, value in state_dict.items():
        if "cls_preds" in key and key.endswith("bias") and hasattr(value, "shape"):
            return int(value.shape[0])
    return 1


def infer_proto_k_from_checkpoint(checkpoint: Dict[str, Any]) -> int:
    """Infer prototype mask channel count from a checkpoint payload or state dict."""
    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, Mapping):
        return 24
    weight = state_dict.get("proto_head.pred.weight")
    if hasattr(weight, "shape") and len(weight.shape) >= 1:
        return int(weight.shape[0])
    return 24


def resolve_model_config(
    model_cfg: Optional[Mapping[str, Any]],
    *,
    num_classes: int,
    proto_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve a complete model config from a named profile plus any explicit overrides."""
    cfg = dict(model_cfg or {})
    profile_key = str(cfg.get("profile", "quasar")).lower()
    if profile_key not in ARCHITECTURE_PROFILES:
        raise ValueError(f"Unknown architecture profile '{profile_key}'")

    profile = ARCHITECTURE_PROFILES[profile_key]
    resolved = {
        "profile": profile.tier_key,
        "display_name": profile.display_name,
        "stem_channels": int(cfg.get("stem_channels", profile.stem_channels)),
        "backbone_channels": tuple(cfg.get("backbone_channels", profile.backbone_channels)),
        "backbone_depths": tuple(cfg.get("backbone_depths", profile.backbone_depths)),
        "neck_channels": tuple(cfg.get("neck_channels", profile.neck_channels)),
        "head_feat_channels": int(cfg.get("head_feat_channels", profile.head_feat_channels)),
        "proto_k": int(cfg.get("proto_k", proto_k if proto_k is not None else profile.proto_k)),
        "num_classes": int(num_classes),
    }
    return resolved


def build_model_from_model_config(model_cfg: Mapping[str, Any], *, num_classes: int) -> ChimeraODIS:
    """Instantiate ChimeraODIS from a resolved model config mapping."""
    resolved = resolve_model_config(model_cfg, num_classes=num_classes)
    return ChimeraODIS(
        num_classes=resolved["num_classes"],
        proto_k=resolved["proto_k"],
        stem_channels=resolved["stem_channels"],
        backbone_channels=tuple(resolved["backbone_channels"]),
        backbone_depths=tuple(resolved["backbone_depths"]),
        neck_channels=tuple(resolved["neck_channels"]),
        head_feat_channels=resolved["head_feat_channels"],
    )


def build_model_from_config(cfg: Mapping[str, Any]) -> ChimeraODIS:
    """Instantiate ChimeraODIS from the project config payload."""
    return build_model_from_model_config(
        cfg.get("model", {}),
        num_classes=int(cfg["data"]["num_classes"]),
    )


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    num_classes: Optional[int] = None,
    proto_k: Optional[int] = None,
) -> ChimeraODIS:
    """Instantiate ChimeraODIS from checkpoint metadata when available."""
    checkpoint_cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(checkpoint_cfg, Mapping):
        return build_model_from_config(checkpoint_cfg)

    checkpoint_model_cfg = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if isinstance(checkpoint_model_cfg, Mapping):
        resolved_num_classes = int(
            num_classes if num_classes is not None else checkpoint_model_cfg.get("num_classes", infer_num_classes_from_checkpoint(checkpoint))
        )
        resolved_proto_k = int(
            proto_k if proto_k is not None else checkpoint_model_cfg.get("proto_k", infer_proto_k_from_checkpoint(checkpoint))
        )
        model_cfg = dict(checkpoint_model_cfg)
        model_cfg["proto_k"] = resolved_proto_k
        return build_model_from_model_config(model_cfg, num_classes=resolved_num_classes)

    inferred_num_classes = int(num_classes if num_classes is not None else infer_num_classes_from_checkpoint(checkpoint))
    inferred_proto_k = int(proto_k if proto_k is not None else infer_proto_k_from_checkpoint(checkpoint))
    model_cfg = {
        "profile": "quasar",
        "proto_k": inferred_proto_k,
    }
    return build_model_from_model_config(model_cfg, num_classes=inferred_num_classes)


def load_model_weights(model: ChimeraODIS, checkpoint: Dict[str, Any] | Mapping[str, Any], *, strict: bool = True) -> None:
    """Load model weights from either a training checkpoint or a plain state dict."""
    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    model.load_state_dict(state_dict, strict=strict)


def describe_model_profile(model_cfg: Mapping[str, Any]) -> str:
    """Return a concise human-readable architecture description."""
    display_name = str(model_cfg.get("display_name", model_cfg.get("profile", "unknown")))
    return f"{display_name} ({model_cfg.get('profile', 'unknown')})"
