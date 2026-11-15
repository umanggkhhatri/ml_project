"""Explicit ICI-inspired 2D adaptation, NOT a verbatim upstream implementation.

Exact CPU CCA selects components at a detached threshold. Selected soft probabilities
retain gradients. Missed components need the global term to recover. Fixed-size center
maps use component-mean probability, unlike upstream's constant hard assignments.
This confidence surrogate has gradients but not gradients through centroid positions.
See docs/methodology.md for the exact departures and reference commit.
"""

import numpy as np
import torch

from shapeguard.data.annotations import components
from shapeguard.losses.segmentation import soft_dice


def center_map(labels, probability=None, size=7):
    """Union of clipped centroid squares. Prediction values are soft component confidence."""
    if probability is None:
        output = torch.zeros(labels.shape, dtype=torch.float32)
    else:
        output = probability * 0
    radius = size // 2
    h, w = labels.shape
    for label in range(1, int(labels.max()) + 1):
        ys, xs = np.nonzero(labels == label)
        if not len(xs):
            continue
        cy, cx = int(np.ceil(ys.mean())), int(np.ceil(xs.mean()))
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        if probability is None:
            output[y0:y1, x0:x1] = 1
        else:
            indices = torch.as_tensor(labels == label, device=probability.device)
            value = probability[indices].mean()
            patch = output[y0:y1, x0:x1].clone()
            output[y0:y1, x0:x1] = torch.maximum(patch, value.expand_as(patch))
    return output


def ici_terms(probability, target, valid, threshold=0.5, center_size=7, compute_center=True):
    zero = probability.sum() * 0
    instance_terms, center_terms = [], []
    for p, y, v in zip(probability, target, valid):
        truth = (y.bool() & v).detach().cpu().numpy()
        predicted = ((p.detach() >= threshold) & v).cpu().numpy()
        gt_labels, count = components(truth)
        pred_labels, _ = components(predicted)
        for k in range(1, count + 1):
            gt = gt_labels == k
            overlaps = np.unique(pred_labels[gt])
            overlaps = overlaps[overlaps > 0]
            support = np.isin(pred_labels, overlaps)
            # Appendix B no-TP variant: another true instance is not a false positive.
            support &= ~truth | gt
            support_t = torch.as_tensor(support, device=p.device)
            gt_t = torch.as_tensor(gt, device=p.device, dtype=p.dtype)
            instance_terms.append(soft_dice(p * support_t, gt_t, v))
        if compute_center:
            predicted_centers = center_map(pred_labels, p, center_size)
            true_centers = center_map(gt_labels, size=center_size).to(p.device)
            center_terms.append(soft_dice(predicted_centers, true_centers, v))
    return (
        torch.stack(instance_terms).mean() if instance_terms else zero,
        torch.stack(center_terms).mean() if center_terms else zero,
    )
