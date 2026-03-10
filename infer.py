import torch, cv2
from models.chimera import ChimeraODIS
from utils.visualize import draw_boxes

def infer(weights, source):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChimeraODIS(num_classes=1).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    img = cv2.imread(source)
    img_resized = cv2.resize(img, (512, 512))
    img_tensor = torch.from_numpy(img_resized[:, :, ::-1].transpose(2,0,1)).float()/255
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)

    print("Inference done")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="chimera_last.pt")
    p.add_argument("--source", type=str, required=True)
    args = p.parse_args()
    infer(args.weights, args.source)
