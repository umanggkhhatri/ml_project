"""C2: normalized instance-local boundary weighting with fixed/adaptive controls."""

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from shapeguard.data.annotations import boundary, expanded_box, geometry


def boundary_weights(mask, config):
    shape = geometry(mask)
    if config["boundary_mode"] == "adaptive":
        tau = float(np.clip(
            config["boundary_scale"] * shape["area"] / max(shape["perimeter"], 1),
            config["tau_min"], config["tau_max"],
        ))
    else:
        tau = config["fixed_tau"]
    distance = ndimage.distance_transform_edt(~boundary(mask))
    return np.exp(-distance / tau).astype(np.float32)


def boundary_loss(probability, target, instances, valid, config):
    terms = []
    for p, y, masks, v in zip(probability, target, instances, valid):
        for instance in masks:
            mask = (instance.to(v.device) & v).detach().cpu().numpy()
            if not mask.any():
                continue
            y0, y1, x0, x1 = expanded_box(mask, config["context"])
            local_mask = mask[y0:y1, x0:x1]
            weights = torch.as_tensor(boundary_weights(local_mask, config), device=p.device)
            weights = weights * v[y0:y1, x0:x1]
            bce = F.binary_cross_entropy(
                p[y0:y1, x0:x1].clamp(1e-6, 1 - 1e-6),
                y[y0:y1, x0:x1].float(), reduction="none",
            )
            terms.append((bce * weights).sum() / weights.sum().clamp_min(1e-6))
    return torch.stack(terms).mean() if terms else probability.sum() * 0
