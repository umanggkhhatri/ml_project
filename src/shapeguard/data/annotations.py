"""Annotation rasterization, geometry and categorical mask validation."""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def rasterize_polygons(segmentation, width, height):
    if not isinstance(segmentation, list):
        raise ValueError("Only COCO polygon annotations are supported, not compressed RLE")
    canvas = Image.new("L", (int(width), int(height)), 0)
    drawer = ImageDraw.Draw(canvas)
    for polygon in segmentation:
        coordinates = np.asarray(polygon, dtype=float)
        if coordinates.size < 6 or coordinates.size % 2 or not np.isfinite(coordinates).all():
            raise ValueError("Malformed polygon")
        points = coordinates.reshape(-1, 2)
        if (points < 0).any() or (points[:, 0] > width).any() or (points[:, 1] > height).any():
            raise ValueError("Polygon coordinates are outside the image")
        drawer.polygon([tuple(p) for p in points], fill=1)
    return np.asarray(canvas, dtype=np.uint8)


def combine_masks(bone, fracture):
    if bone.shape != fracture.shape:
        raise ValueError("Bone and fracture masks must have identical native dimensions")
    mask = np.zeros(bone.shape, dtype=np.uint8)
    mask[bone > 0] = 1
    mask[fracture > 0] = 2
    return mask


def components(mask):
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3)))
    return labels, count


def boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def geometry(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("An instance cannot be empty")
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return {
        "area": int(len(xs)),
        "perimeter": int(boundary(mask).sum()),
        "bbox": [y0, y1, x0, x1],
        "center": [float(ys.mean()), float(xs.mean())],
        "aspect_ratio": float(max(y1 - y0, x1 - x0) / min(y1 - y0, x1 - x0)),
    }


def expanded_box(mask, context=0.25):
    y0, y1, x0, x1 = geometry(mask)["bbox"]
    py = max(1, int(np.ceil((y1 - y0) * context)))
    px = max(1, int(np.ceil((x1 - x0) * context)))
    h, w = mask.shape
    return max(0, y0 - py), min(h, y1 + py), max(0, x0 - px), min(w, x1 + px)


def square_box(mask, context=0.25):
    """Matched-objective control: square support enclosing the instance, clipped at edges."""
    y0, y1, x0, x1 = geometry(mask)["bbox"]
    side = int(np.ceil(max(y1 - y0, x1 - x0) * (1 + 2 * context))) + 2
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    h, w = mask.shape
    return max(0, int(np.floor(cy - side / 2))), min(h, int(np.ceil(cy + side / 2))), max(0, int(np.floor(cx - side / 2))), min(w, int(np.ceil(cx + side / 2)))


def validate_mask(mask, classes):
    if mask.ndim != 2 or not set(np.unique(mask)).issubset(set(range(classes))):
        raise ValueError(f"Expected a 2D categorical mask with values 0..{classes - 1}")
