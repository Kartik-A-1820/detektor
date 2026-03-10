import torch, yaml
from torch.utils.data import DataLoader
from datasets import build_dataset
from models.chimera import ChimeraODIS

def validate(config_path):
    cfg = yaml.safe_load(open(config_path))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = ChimeraODIS(num_classes=cfg["data"]["num_classes"],
                        proto_k=cfg["model"]["proto_k"]).to(device)
    model.load_state_dict(torch.load("chimera_last.pt", map_location=device))
    model.eval()

    dataset = build_dataset(cfg, split="val")
    loader = DataLoader(dataset, batch_size=4)

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            print("Batch processed")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/chimera_s_512.yaml")
    args = p.parse_args()
    validate(args.config)
