from copy import deepcopy

import pytest
import torch

from shapeguard.config import DEFAULTS, load_config
from shapeguard.losses.objective import objective
from shapeguard.models import build_model


def test_all_main_experiments_are_binary():
    for name in ("baseline", "ici", "geometry_square", "geometry", "boundary_fixed", "boundary_adaptive", "proposed", "contrast_control", "full", "attention"):
        config = load_config(f"configs/gpu/{name}.yaml")
        assert config["data"]["task"] == "binary"
        assert config["data"]["sampling"] == "balanced"
        assert config["train"]["cuda_memory_fraction"] == 0.075


@pytest.mark.parametrize("name", ["baseline", "ici", "geometry_square", "geometry", "boundary_fixed", "boundary_adaptive", "proposed", "full"])
def test_binary_objective_has_finite_gradients(name):
    config = load_config(f"configs/{name}.yaml")
    config["model"]["widths"] = [2, 4, 8, 16, 32]
    target = torch.zeros((2, 32, 32), dtype=torch.long)
    target[0, 14:17, 4:28] = 1
    batch = {"image": torch.rand((2, 1, 32, 32)), "mask": target,
             "valid": torch.ones_like(target, dtype=torch.bool),
             "instances": [(target[0] == 1)[None], torch.zeros((0, 32, 32), dtype=torch.bool)]}
    model = build_model(config)
    logits = model(batch["image"])
    assert logits.shape[1] == 2
    loss, _ = objective(logits, batch, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_class_weights_must_match_task(tmp_path):
    from shapeguard.config import validate
    config = deepcopy(DEFAULTS)
    config["loss"]["class_weights"] = [1, 1, 10]
    with pytest.raises(ValueError, match="one positive weight"):
        validate(config)


def test_cuda_selection_has_explicit_index(monkeypatch):
    from shapeguard.reproducibility import choose_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert choose_device("cuda") == torch.device("cuda", 0)
    with pytest.raises(ValueError, match="visible allocation"):
        choose_device("cuda:1")
