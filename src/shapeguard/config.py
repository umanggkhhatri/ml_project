"""Strict configuration loading with relative-path inheritance."""

from copy import deepcopy
from pathlib import Path

import yaml

DEFAULTS = {
    "name": "baseline",
    "seed": 42,
    "device": "auto",
    "data": {
        "manifest": "data/prepared/manifest.csv",
        "task": "binary",
        "size": 384,
        "workers": 0,
        "augment": True,
        "contrast_augment": False,
        "dual_view": False,
        "sampling": "natural",
    },
    "model": {"architecture": "unet", "widths": [16, 32, 64, 128, 256]},
    "train": {
        "epochs": 50,
        "batch_size": 2,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "patience": 10,
        "amp": False,
        "output": "runs",
        "threads": 4,
        "selection": "fracture_positive_dice",
        "cuda_memory_fraction": None,
    },
    "loss": {
        "ce": 1.0,
        "class_weights": None,
        "dice": 1.0,
        "instance": 0.0,
        "center": 0.0,
        "geometry": 0.0,
        "geometry_mode": "adaptive",
        "boundary": 0.0,
        "consistency": 0.0,
        "center_size": 7,
        "context": 0.25,
        "boundary_mode": "adaptive",
        "boundary_scale": 1.0,
        "fixed_tau": 3.0,
        "tau_min": 1.0,
        "tau_max": 8.0,
        "threshold": 0.5,
    },
}


def merge(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {key}")
        result[key] = merge(base[key], value) if isinstance(value, dict) else value
    return result


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Circular configuration inheritance: {path}")
    seen.add(path)
    with path.open() as handle:
        updates = yaml.safe_load(handle) or {}
    parent = updates.pop("extends", None)
    base = load_config(path.parent / parent, seen) if parent else DEFAULTS
    config = merge(base, updates)
    validate(config)
    return config


def validate(config):
    if config["data"]["task"] not in {"binary", "multiclass"}:
        raise ValueError("data.task must be binary or multiclass")
    if config["data"]["sampling"] not in {"natural", "balanced"}:
        raise ValueError("data.sampling must be natural or balanced")
    if config["model"]["architecture"] not in {"unet", "attention_unet"}:
        raise ValueError("Unsupported model architecture")
    widths = config["model"]["widths"]
    if len(widths) != 5 or any(int(w) < 2 for w in widths):
        raise ValueError("model.widths must contain five positive widths >= 2")
    if config["data"]["size"] < 32:
        raise ValueError("Image size must be >= 32")
    for key in ("epochs", "batch_size", "patience", "threads"):
        if config["train"][key] < 1:
            raise ValueError(f"train.{key} must be positive")
    if config["train"]["selection"] not in {"fracture_positive_dice", "foreground_dice"}:
        raise ValueError("Unsupported checkpoint selection metric")
    loss = config["loss"]
    classes = 2 if config["data"]["task"] == "binary" else 3
    if loss["class_weights"] is not None and (len(loss["class_weights"]) != classes or any(w <= 0 for w in loss["class_weights"])):
        raise ValueError("class_weights must contain one positive weight per class")
    fraction = config["train"]["cuda_memory_fraction"]
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("CUDA memory fraction must be in (0, 1]")
    if any(loss[k] < 0 for k in ("ce", "dice", "instance", "center", "geometry", "boundary", "consistency")):
        raise ValueError("Loss weights must be nonnegative")
    if loss["ce"] + loss["dice"] <= 0:
        raise ValueError("A global segmentation objective is required")
    if loss["boundary_mode"] not in {"fixed", "adaptive"}:
        raise ValueError("boundary_mode must be fixed or adaptive")
    if loss["geometry_mode"] not in {"adaptive", "square"}:
        raise ValueError("geometry_mode must be adaptive or square")
    if not 0 < loss["threshold"] < 1 or loss["context"] < 0:
        raise ValueError("Invalid threshold or context")
    if loss["center_size"] < 1 or loss["center_size"] % 2 != 1:
        raise ValueError("center_size must be a positive odd number")
    if not 0 < loss["tau_min"] <= loss["tau_max"] or loss["fixed_tau"] <= 0:
        raise ValueError("Boundary scales must be positive and ordered")
