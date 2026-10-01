#!/usr/bin/env python
"""Minimal Python client for a running Detektor API.

    python examples/api_client.py photo.jpg                       # print detections
    python examples/api_client.py photo.jpg --save annotated.jpg  # also draw boxes
    DETEKTOR_API_KEY=... python examples/api_client.py *.jpg --batch

Only needs ``requests`` (and ``opencv-python`` / ``pillow`` for ``--save``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests


def predict(base_url: str, path: Path, api_key: str | None, conf: float, iou: float) -> dict:
    headers = {"X-API-Key": api_key} if api_key else {}
    with path.open("rb") as handle:
        response = requests.post(
            f"{base_url}/v1/predict",
            params={"conf_thresh": conf, "iou_thresh": iou},
            files={"image": (path.name, handle, "image/jpeg")},
            headers=headers,
            timeout=60,
        )
    response.raise_for_status()
    return response.json()


def predict_batch(base_url: str, paths: list[Path], api_key: str | None, conf: float, iou: float) -> dict:
    headers = {"X-API-Key": api_key} if api_key else {}
    files = [("images", (p.name, p.read_bytes(), "image/jpeg")) for p in paths]
    response = requests.post(
        f"{base_url}/v1/predict_batch", params={"conf_thresh": conf, "iou_thresh": iou}, files=files, headers=headers, timeout=120
    )
    response.raise_for_status()
    return response.json()


def draw(path: Path, detections: list[dict], out: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(path).convert("RGB")
    canvas = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        canvas.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=3)
        canvas.text((x1 + 3, y1 + 2), f"{det['label']} {det['score']:.2f}", fill=(255, 140, 0))
    image.save(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--url", default=os.getenv("DETEKTOR_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("DETEKTOR_API_KEY"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--batch", action="store_true", help="send all images in one /v1/predict_batch request")
    parser.add_argument("--save", type=Path, help="write an annotated copy (single image only)")
    args = parser.parse_args()

    try:
        if args.batch:
            result = predict_batch(args.url, args.images, args.api_key, args.conf, args.iou)
            for path, pred in zip(args.images, result["predictions"]):
                print(f"{path.name}: {pred['num_detections']} detection(s)")
            print(f"total inference time: {result.get('total_inference_time_ms', 0):.1f} ms")
            return 0
        for path in args.images:
            pred = predict(args.url, path, args.api_key, args.conf, args.iou)
            print(f"{path.name}: {pred['num_detections']} detection(s) in {pred.get('inference_time_ms', 0):.1f} ms")
            for det in pred["detections"]:
                print(f"  label={det['label']} score={det['score']:.3f} box={[round(v, 1) for v in det['box']]}")
            if args.save and len(args.images) == 1:
                draw(path, pred["detections"], args.save)
                print(f"saved {args.save}")
    except requests.HTTPError as exc:
        print(f"HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    except requests.ConnectionError:
        print(f"Could not reach {args.url} — is `python serve.py` running?", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
