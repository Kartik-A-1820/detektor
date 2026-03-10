from .yolo_seg import YOLOSegDataset

def build_dataset(cfg, split):
    root = cfg["data"][split]
    return YOLOSegDataset(root, img_size=cfg["train"]["img_size"])
