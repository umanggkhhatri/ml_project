import cv2
import numpy as np


def corrupt(image, kind, severity, rng):
    if kind == "clean":
        return image
    if severity not in (1, 2, 3):
        raise ValueError("Corruption severity must be 1, 2, or 3")
    if kind == "contrast":
        return np.clip((image - 0.5) * (1 - severity * 0.2) + 0.5, 0, 1)
    if kind == "gamma":
        return image ** (1 + severity * 0.25)
    if kind == "noise":
        return np.clip(image + rng.normal(0, severity * 0.02, image.shape), 0, 1).astype(np.float32)
    if kind == "blur":
        return cv2.GaussianBlur(image, (0, 0), severity * 0.5)
    raise ValueError(f"Unknown corruption: {kind}")
