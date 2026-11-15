"""C3: appearance-only perturbations; symmetric JS divergence on valid pixels."""

import torch


def contrast_view(image, valid):
    gamma = torch.empty((len(image), 1, 1, 1), device=image.device).uniform_(0.85, 1.15)
    view = image.clamp(0, 1).pow(gamma)
    return view * valid[:, None]


def js_divergence(first, second, valid):
    p, q = first.softmax(1).clamp_min(1e-7), second.softmax(1).clamp_min(1e-7)
    mean = (p + q) / 2
    divergence = 0.5 * ((p * (p.log() - mean.log())).sum(1) + (q * (q.log() - mean.log())).sum(1))
    return divergence[valid].mean()
