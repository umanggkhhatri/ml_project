import numpy as np
from scipy import ndimage

from shapeguard.data.annotations import boundary


def binary_metrics(prediction, target, valid=None, distances=True):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    valid = np.ones_like(target) if valid is None else np.asarray(valid, bool)
    if not valid.any():
        raise ValueError("No valid pixels to evaluate")
    prediction, target = prediction & valid, target & valid
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target & valid).sum())
    fn = int((~prediction & target & valid).sum())
    tn = int((~prediction & ~target & valid).sum())
    both_empty = tp + fp + fn == 0
    result = {
        "dice": 1.0 if both_empty else 2 * tp / (2 * tp + fp + fn),
        "iou": 1.0 if both_empty else tp / (tp + fp + fn),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "target_pixels": int(target.sum()), "prediction_pixels": int(prediction.sum()),
        "both_empty": both_empty, "one_empty": bool(target.any() != prediction.any()),
    }
    if distances:
        if both_empty:
            hd, hd95 = 0.0, 0.0
        elif result["one_empty"]:
            hd, hd95 = np.inf, np.inf
        else:
            a, b = boundary(prediction), boundary(target)
            ab = ndimage.distance_transform_edt(~b)[a]
            ba = ndimage.distance_transform_edt(~a)[b]
            hd = float(max(ab.max(), ba.max()))
            hd95 = float(max(np.percentile(ab, 95), np.percentile(ba, 95)))
        result.update(hausdorff=hd, hd95=hd95)
    return result
