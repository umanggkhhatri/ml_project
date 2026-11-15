import torch
from torch.nn import functional as F


def soft_dice(probability, target, valid=None, epsilon=1e-6):
    if valid is not None:
        probability = probability * valid
        target = target * valid
    intersection = (probability * target).sum()
    return 1 - (2 * intersection + epsilon) / (probability.sum() + target.sum() + epsilon)


def global_loss(logits, target, valid, class_weights=None):
    if not valid.any():
        raise ValueError("No valid pixels in training batch")
    weights = torch.as_tensor(class_weights, device=logits.device, dtype=logits.dtype) if class_weights is not None else None
    ce = F.cross_entropy(logits, target, weight=weights, ignore_index=255, reduction="none")
    denominator = weights[target[valid]].sum() if weights is not None else valid.sum()
    ce = ce[valid].sum() / denominator.clamp_min(1)
    probability = logits.softmax(1)
    dice_terms = []
    for b in range(len(target)):
        for cls in range(1, logits.shape[1]):
            dice_terms.append(soft_dice(probability[b, cls], (target[b] == cls).float(), valid[b]))
    return ce, torch.stack(dice_terms).mean()
