import random
from pathlib import Path

import numpy as np
import torch


def rng_state(generator, device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
             "loader": generator.get_state()}
    if device.type == "cuda":
        # Do not initialize contexts on every visible GPU of a shared server.
        state["cuda"] = torch.cuda.get_rng_state(device)
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, generator, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if "cuda" in state and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    if "mps" in state and device.type == "mps":
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # Only load your own trusted checkpoints: optimizer/RNG payloads use pickle.
    return torch.load(path, map_location="cpu", weights_only=False)
