"""One shared letterbox geometry for images, categorical masks and polygon instances."""

import cv2
import numpy as np


def letterbox(image, mask, instances, size):
    h, w = image.shape
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.zeros((size, size), dtype=np.float32)
    target = np.full((size, size), 255, dtype=np.uint8)
    valid = np.zeros((size, size), dtype=bool)
    out[top:top + nh, left:left + nw] = cv2.resize(image.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    target[top:top + nh, left:left + nw] = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    valid[top:top + nh, left:left + nw] = True
    transformed = np.zeros((len(instances), size, size), dtype=np.uint8)
    vanished = 0
    for i, instance in enumerate(instances):
        small = cv2.resize(instance, (nw, nh), interpolation=cv2.INTER_NEAREST)
        transformed[i, top:top + nh, left:left + nw] = small
        vanished += int(instance.any() and not small.any())
    return out, target, transformed, valid, {
        "top": top, "left": left, "height": nh, "width": nw,
        "native_height": h, "native_width": w, "vanished_instances": vanished,
    }


def normalize(image):
    image = image.astype(np.float32)
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def augment(image, mask, instances, valid, contrast=False):
    # Modest affine perturbation; no medically unrealistic vertical flips.
    size = image.shape[0]
    angle = float(np.random.uniform(-10, 10))
    scale = float(np.random.uniform(0.95, 1.05))
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    image = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST, borderValue=255)
    valid = cv2.warpAffine(valid.astype(np.uint8), matrix, (size, size), flags=cv2.INTER_NEAREST).astype(bool)
    instances = np.stack([
        cv2.warpAffine(m, matrix, (size, size), flags=cv2.INTER_NEAREST) for m in instances
    ]) if len(instances) else instances
    if contrast:
        gamma = float(np.random.uniform(0.85, 1.15))
        image = np.clip(image, 0, 1) ** gamma
    image[~valid] = 0
    mask[~valid] = 255
    return image, mask, instances, valid
