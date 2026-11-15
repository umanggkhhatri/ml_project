from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import torch

from shapeguard.config import DEFAULTS, load_config
from shapeguard.data.dataset import FractureDataset, collate
from shapeguard.engine.checkpoint import load_checkpoint
from shapeguard.engine.train import train
from shapeguard.losses.objective import objective
from shapeguard.models import build_model
from shapeguard.smoke import synthetic_dataset


def tiny_config(tmp_path):
    manifest = synthetic_dataset(tmp_path / "data")
    config = deepcopy(DEFAULTS)
    config["device"] = "cpu"
    config["model"]["widths"] = [2, 4, 8, 16, 32]
    config["data"].update(manifest=str(manifest), task="multiclass", size=32, augment=False)
    config["train"].update(output=str(tmp_path / "runs"), epochs=1, patience=5, threads=2)
    return config


def test_configs_load():
    for path in Path("configs").rglob("*.yaml"):
        load_config(path)
    proposed = load_config("configs/proposed.yaml")
    assert proposed["loss"]["geometry"] > 0 and proposed["loss"]["center"] == 0
    assert load_config("configs/full.yaml")["data"]["dual_view"]


def test_unknown_and_circular_config(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: 1\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
    path.write_text("extends: bad.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(path)


def test_small_sample_overfit(tmp_path):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = tiny_config(tmp_path)
    data = FractureDataset(config["data"]["manifest"], "train", size=32)
    batch = collate([data[0], data[1]])
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = float(objective(model(batch["image"]), batch, config)[0].detach())
    for _ in range(50):
        optimizer.zero_grad()
        loss, _ = objective(model(batch["image"]), batch, config)
        loss.backward()
        optimizer.step()
    final = float(objective(model(batch["image"]), batch, config)[0].detach())
    assert final < initial * 0.6


def test_checkpoint_resume_and_data_guard(tmp_path):
    config = tiny_config(tmp_path)
    output = train(config)
    first = load_checkpoint(output / "last.pt")
    assert first["epoch"] == 0 and first["optimizer"]["state"]
    with pytest.raises(FileExistsError):
        train(config)
    config["train"]["epochs"] = 2
    train(config, resume=True)
    last = load_checkpoint(output / "last.pt")
    assert last["epoch"] == 1 and len(pd.read_csv(output / "history.csv")) == 2
    altered = deepcopy(config)
    altered["train"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="unchanged"):
        train(altered, resume=True)


def test_dual_view_training_and_resume_matches_continuous(tmp_path):
    config = tiny_config(tmp_path)
    config["data"]["dual_view"] = True
    config["data"]["sampling"] = "balanced"
    config["loss"]["consistency"] = 0.1
    config["name"] = "resumed"
    output = train(config)
    config["train"]["epochs"] = 2
    train(config, resume=True)
    resumed = load_checkpoint(output / "last.pt")["model"]
    continuous = deepcopy(config)
    continuous["name"] = "continuous"
    other = train(continuous)
    uninterrupted = load_checkpoint(other / "last.pt")["model"]
    for key in resumed:
        assert torch.equal(resumed[key], uninterrupted[key]), key
