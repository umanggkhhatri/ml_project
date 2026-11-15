"""C1: aspect-ratio-aware local supervision replaces the fixed-center surrogate."""

import torch
from torch.nn import functional as F

from shapeguard.data.annotations import expanded_box, square_box
from shapeguard.losses.segmentation import soft_dice


def geometry_loss(probability, target, instances, valid, context=0.25, mode="adaptive"):
    terms = []
    for p, y, masks, v in zip(probability, target, instances, valid):
        for instance in masks:
            mask = (instance.to(v.device) & v).detach().cpu().numpy()
            if not mask.any():
                continue
            y0, y1, x0, x1 = square_box(mask, context) if mode == "square" else expanded_box(mask, context)
            local_p = p[y0:y1, x0:x1]
            local_y = y[y0:y1, x0:x1].float()
            local_v = v[y0:y1, x0:x1]
            dice = soft_dice(local_p, local_y, local_v)
            bce = F.binary_cross_entropy(local_p.clamp(1e-6, 1 - 1e-6), local_y, reduction="none")
            terms.append(dice + bce[local_v].mean())
    return torch.stack(terms).mean() if terms else probability.sum() * 0
