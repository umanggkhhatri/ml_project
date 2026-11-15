"""Synthetic software integration test; never a medical benchmark."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from shapeguard.config import DEFAULTS
from shapeguard.data.dataset import FractureDataset, collate
from shapeguard.engine.evaluate import evaluate
from shapeguard.engine.train import train
from shapeguard.losses.objective import objective
from shapeguard.models import build_model
from shapeguard.reporting.figures import figures
from shapeguard.reproducibility import choose_device, save_json, seed_everything, sha256


def synthetic_dataset(directory, count=12):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.csv").exists():
        raise FileExistsError("Synthetic dataset exists; choose a new smoke directory")
    rows = []
    rng = np.random.default_rng(42)
    for i in range(count):
        h, w = 48, 64
        bone = np.zeros((h, w), np.uint8)
        bone[8:40, 12:52] = 1
        fracture = np.zeros_like(bone)
        if i % 2 == 0:
            fracture[22:25, 14:50] = 1
        categorical = bone.copy()
        categorical[fracture > 0] = 2
        image = np.clip(20 + bone * 140 + fracture * 70 + rng.normal(0, 3, (h, w)), 0, 255).astype(np.uint8)
        image_id = f"SYNTHETIC_{i:03d}"
        paths = {key: directory / f"{image_id}_{key}.png" for key in ("image", "fracture", "bone", "mask")}
        for key, array in (("image", image), ("fracture", fracture), ("bone", bone), ("mask", categorical)):
            Image.fromarray(array).save(paths[key])
        instance_path = directory / f"{image_id}_instances.npz"
        stack = fracture[None] if fracture.any() else np.zeros((0, h, w), np.uint8)
        np.savez_compressed(instance_path, masks=stack)
        rows.append({
            "image_id": image_id, "image_path": paths["image"].name,
            "fracture_path": paths["fracture"].name, "bone_path": paths["bone"].name,
            "mask_path": paths["mask"].name, "instance_path": instance_path.name,
            "bone_status": "accepted", "split": "train" if i < 8 else "val" if i < 10 else "test",
            "group_id": image_id, "sha256": sha256(paths["image"]), "fractured": int(fracture.any()),
            "width": w, "height": h, "hardware": 0, "multiscan": 0, "hand": 0, "leg": 1,
        })
    path = directory / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    save_json(directory / "SYNTHETIC_ONLY.json", {"warning": "Synthetic software fixtures. Not X-ray data or clinical results."})
    return path


def smoke(directory="runs/smoke", device_request="cpu"):
    directory = Path(directory).resolve()
    manifest = synthetic_dataset(directory / "data")
    config = deepcopy(DEFAULTS)
    config["name"] = "SYNTHETIC_SMOKE"
    config["device"] = device_request
    config["data"].update(manifest=str(manifest), task="multiclass", size=48, augment=False)
    config["model"]["widths"] = [4, 8, 16, 32, 64]
    config["train"].update(epochs=2, patience=5, output=str(directory / "experiments"))
    config["loss"].update(instance=0.1, geometry=0.1, boundary=0.1)
    seed_everything(42)
    device = choose_device(device_request)
    if device.type == "cuda":
        config["train"]["cuda_memory_fraction"] = 0.075
        torch.cuda.set_per_process_memory_fraction(0.075, device)
    dataset = FractureDataset(manifest, "train", size=48)
    batch = collate([dataset[0], dataset[1]])
    for key in ("image", "mask", "valid"):
        batch[key] = batch[key].to(device)
    model = build_model(config).to(device)
    config["loss"]["center"] = 0.1
    value, terms = objective(model(batch["image"]), batch, config)
    value.backward()
    if not torch.isfinite(value) or not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()):
        raise AssertionError("Loss/gradient smoke test failed")
    config["loss"]["center"] = 0
    run = train(config)
    config["train"]["epochs"] = 3
    train(config, resume=True)
    evaluated = evaluate(run / "best.pt", device_request, robustness=True)
    plots = figures(run / "best.pt", evaluated)
    result = {"status": "passed", "device": str(device), "loss_terms": terms,
              "run": str(run), "evaluation": str(evaluated), "figures": str(plots),
              "warning": "Synthetic software verification only, not medical performance."}
    save_json(directory / "smoke_result.json", result)
    return result
