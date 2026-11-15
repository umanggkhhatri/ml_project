from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from shapeguard.data.transforms import letterbox, normalize
from shapeguard.engine.checkpoint import load_checkpoint
from shapeguard.models import build_model
from shapeguard.reproducibility import choose_device


def predict(checkpoint, image_path, output, device_request="auto"):
    state = load_checkpoint(checkpoint)
    config = state["config"]
    device = choose_device(device_request)
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    image = np.asarray(Image.open(image_path).convert("L"))
    h, w = image.shape
    resized, _, _, _, meta = letterbox(normalize(image), np.zeros_like(image), np.zeros((0, h, w), np.uint8), config["data"]["size"])
    with torch.inference_mode():
        probabilities = model(torch.from_numpy(resized)[None, None].to(device)).softmax(1)[0].cpu().numpy()
    top, left, nh, nw = meta["top"], meta["left"], meta["height"], meta["width"]
    native = np.stack([cv2.resize(p[top:top + nh, left:left + nw], (w, h), interpolation=cv2.INTER_LINEAR) for p in probabilities])
    mask = native.argmax(0).astype(np.uint8)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output / "mask.png")
    np.save(output / "probabilities.npy", native)
    from shapeguard.reporting.figures import overlay
    Image.fromarray(overlay(image, mask, config["data"]["task"])).save(output / "overlay.png")
    return output
