import os, cv2, torch
from torch.utils.data import Dataset
import numpy as np

class YOLOSegDataset(Dataset):
    def __init__(self, root, img_size=512):
        self.root = root
        self.img_size = img_size
        self.images = [f for f in os.listdir(os.path.join(root, "images")) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img[:, :, ::-1].transpose(2,0,1)).float()/255
        target = {"boxes": torch.zeros((0,4)), "labels": torch.zeros((0,), dtype=torch.long)}
        return img, target
