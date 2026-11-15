"""One-to-one IoU matching; merged predictions cannot count as multiple detections."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from shapeguard.data.annotations import components, geometry


def instance_metrics(prediction, ground_truth_instances, thresholds, valid=None, minimum_iou=0.1):
    prediction = np.asarray(prediction, bool)
    if valid is not None:
        prediction &= valid
    labels, count = components(prediction)
    ground_truth = [np.asarray(m, bool) & (valid if valid is not None else True) for m in ground_truth_instances]
    ground_truth = [m for m in ground_truth if m.any()]
    scores = np.zeros((len(ground_truth), count), dtype=float)
    for i, gt in enumerate(ground_truth):
        for j in range(count):
            pred = labels == j + 1
            scores[i, j] = (gt & pred).sum() / max(1, (gt | pred).sum())
    matches = {}
    if scores.size:
        # Prioritize maximum number of valid matches, then their total IoU.
        utility = (scores >= minimum_iou).astype(float) * (min(scores.shape) + 1) + scores
        rows, cols = linear_sum_assignment(-utility)
        matches = {int(i): int(j) for i, j in zip(rows, cols) if scores[i, j] >= minimum_iou}
    records = []
    for i, gt in enumerate(ground_truth):
        info = geometry(gt)
        fraction = info["area"] / (int(valid.sum()) if valid is not None else gt.size)
        group = "small" if fraction <= thresholds[0] else "medium" if fraction <= thresholds[1] else "large"
        j = matches.get(i)
        iou = float(scores[i, j]) if j is not None else 0.0
        records.append({
            "instance": i, "area_fraction": fraction, "size_group": group,
            "aspect_ratio": info["aspect_ratio"], "matched": j is not None,
            "matched_iou": iou, "matched_dice": 2 * iou / (1 + iou),
        })
    return records, {"ground_truth_instances": len(ground_truth), "predicted_instances": count,
                     "matched_instances": len(matches), "missed_instances": len(ground_truth) - len(matches),
                     "false_instances": count - len(matches), "matching_iou_threshold": minimum_iou}
