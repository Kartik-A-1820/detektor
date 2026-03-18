from .yolo_seg import YOLOSegDataset

def build_dataset(cfg, split):
    root = cfg["data"][split]
    is_train = split == "train"
    augment_cfg = cfg.get("augment", {}) if is_train else {}
    return YOLOSegDataset(
        root,
        img_size=cfg["train"]["img_size"],
        augment=is_train and bool(augment_cfg.get("enabled", False)),
        augment_cfg=augment_cfg,
    )
