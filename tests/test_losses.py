from copy import deepcopy

import numpy as np
import pytest
import torch

from shapeguard.config import DEFAULTS
from shapeguard.losses.boundary import boundary_loss, boundary_weights
from shapeguard.losses.consistency import js_divergence
from shapeguard.losses.geometry import geometry_loss
from shapeguard.losses.ici import center_map, ici_terms
from shapeguard.losses.objective import objective
from shapeguard.losses.segmentation import soft_dice
from shapeguard.models import build_model


def sample_batch():
    target = torch.zeros((2, 32, 32), dtype=torch.long)
    target[:, 5:27, 8:24] = 1
    target[0, 15:17, 9:23] = 2
    instance = target[0] == 2
    return {"image": torch.rand((2, 1, 32, 32)), "mask": target,
            "valid": torch.ones_like(target, dtype=torch.bool),
            "instances": [instance[None], torch.zeros((0, 32, 32), dtype=torch.bool)]}


@pytest.mark.parametrize("architecture", ["unet", "attention_unet"])
def test_model_shapes(architecture):
    config = deepcopy(DEFAULTS)
    config["data"]["task"] = "multiclass"
    config["model"].update(widths=[2, 4, 8, 16, 32], architecture=architecture)
    model = build_model(config)
    assert model(torch.rand((2, 1, 33, 47))).shape == (2, 3, 33, 47)


def test_ici_component_selection_and_gradients():
    p = torch.zeros((1, 12, 12), requires_grad=True)
    with torch.no_grad():
        p[:, 2:5, 2:5] = 0.8
        p[:, 8:10, 8:10] = 0.9  # Spurious, disconnected; instance term must exclude it.
    y = torch.zeros_like(p, dtype=torch.bool)
    y[:, 2:5, 2:5] = True
    valid = torch.ones_like(y)
    instance, center = ici_terms(p, y, valid)
    expected = soft_dice(torch.full((3, 3), 0.8), torch.ones((3, 3)))
    assert instance.item() == pytest.approx(expected.item())
    instance.backward(retain_graph=True)
    assert p.grad[:, 2:5, 2:5].abs().sum() > 0
    assert p.grad[:, 8:10, 8:10].abs().sum() == 0
    p.grad.zero_()
    center.backward()
    assert p.grad.abs().sum() > 0  # Our documented component-confidence surrogate.


def test_center_square_clipping():
    labels = np.zeros((10, 10), np.int32)
    labels[0, 0] = 1
    result = center_map(labels, size=7)
    assert result.sum() == 16
    assert not result[-1, -1]


def test_geometry_and_boundary_have_gradients():
    batch = sample_batch()
    target = batch["mask"] == 2
    for fn in ("geometry", "boundary"):
        p = torch.full(target.shape, 0.3, requires_grad=True)
        loss = geometry_loss(p, target, batch["instances"], batch["valid"]) if fn == "geometry" else boundary_loss(p, target, batch["instances"], batch["valid"], DEFAULTS["loss"])
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


def test_boundary_control_and_clamp():
    mask = np.zeros((20, 20), bool)
    mask[2:18, 2:18] = True
    fixed = deepcopy(DEFAULTS["loss"])
    fixed["boundary_mode"] = "fixed"
    adaptive = deepcopy(DEFAULTS["loss"])
    adaptive["tau_min"] = adaptive["tau_max"] = fixed["fixed_tau"]
    assert np.allclose(boundary_weights(mask, fixed), boundary_weights(mask, adaptive))


def test_consistency_identity_and_gradients():
    first = torch.randn((2, 3, 8, 8), requires_grad=True)
    second = torch.randn_like(first, requires_grad=True)
    valid = torch.ones((2, 8, 8), dtype=torch.bool)
    assert js_divergence(first, first, valid).item() == pytest.approx(0, abs=1e-7)
    loss = js_divergence(first, second, valid)
    loss.backward()
    assert loss > 0 and first.grad.abs().sum() > 0 and second.grad.abs().sum() > 0


def test_negative_batch_and_ignore_padding():
    config = deepcopy(DEFAULTS)
    config["loss"].update(instance=0.5, center=0.25, geometry=0.25, boundary=0.25)
    batch = sample_batch()
    batch["mask"].zero_()
    batch["instances"] = [torch.zeros((0, 32, 32), dtype=torch.bool)] * 2
    batch["valid"][:, :4] = False
    batch["mask"][:, :4] = 255
    logits = torch.randn((2, 3, 32, 32), requires_grad=True)
    loss, _ = objective(logits, batch, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[:, :, :4].abs().sum() == 0


def test_all_ignore_rejected():
    config = deepcopy(DEFAULTS)
    batch = sample_batch()
    batch["valid"].zero_()
    batch["mask"].fill_(255)
    with pytest.raises(ValueError, match="No valid"):
        objective(torch.rand((2, 3, 32, 32)), batch, config)
