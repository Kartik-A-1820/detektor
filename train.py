import torch, yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import build_dataset
from models.chimera import ChimeraODIS
from utils.vram import set_vram_cap, vram_report

def train(config_path):
    cfg = yaml.safe_load(open(config_path))
    set_vram_cap(cfg["train"]["vram_cap"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = ChimeraODIS(num_classes=cfg["data"]["num_classes"],
                        proto_k=cfg["model"]["proto_k"]).to(device)

    dataset = build_dataset(cfg, split="train")
    loader = DataLoader(dataset,
                        batch_size=cfg["train"]["batch_size"],
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
                loss = model.compute_loss(imgs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=float(loss.item()))
        print(vram_report("After epoch: "))

    torch.save(model.state_dict(), "chimera_last.pt")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/chimera_s_512.yaml")
    args = p.parse_args()
    train(args.config)
