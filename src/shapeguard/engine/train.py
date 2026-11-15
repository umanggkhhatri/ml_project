"""Resumable experiments with fixed data provenance and validation-only model selection."""

import json
import os
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from shapeguard.data.dataset import FractureDataset, collate
from shapeguard.data.manifest import fingerprint, freeze_annotations
from shapeguard.engine.checkpoint import load_checkpoint, restore_rng, rng_state, save_checkpoint
from shapeguard.evaluation.metrics import binary_metrics
from shapeguard.losses.consistency import contrast_view, js_divergence
from shapeguard.losses.objective import objective
from shapeguard.models import build_model
from shapeguard.reproducibility import (
    choose_device,
    environment,
    save_json,
    seed_everything,
    synchronize,
)


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def optimizer_step(model, optimizer, scaler, amp=False):
    """Let GradScaler skip ordinary FP16 overflows; keep the strict float32 guard.

    unscale_ records nonfinite gradients before clipping. scaler.step then skips
    the optimizer update on overflow and scaler.update reduces the scale.
    """
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=not amp)
    skipped = amp and not bool(torch.isfinite(norm))
    scaler.step(optimizer)
    scaler.update()
    return norm, skipped


def validate_model(model, loader, device):
    model.eval()
    positive_dice, foreground_dice, negative_fp = [], [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device))
            predictions = logits.argmax(1).cpu().numpy()
            fracture_class = logits.shape[1] - 1
            for pred, target, valid in zip(predictions, batch["mask"].numpy(), batch["valid"].numpy()):
                for cls in range(1, logits.shape[1]):
                    metrics = binary_metrics(pred == cls, target == cls, valid, distances=False)
                    foreground_dice.append(metrics["dice"])
                    if cls == fracture_class:
                        if metrics["target_pixels"]:
                            positive_dice.append(metrics["dice"])
                        else:
                            negative_fp.append(float(metrics["prediction_pixels"] > 0))
    return {
        "fracture_positive_dice": float(np.mean(positive_dice)) if positive_dice else np.nan,
        "foreground_dice": float(np.mean(foreground_dice)),
        "negative_image_fp_rate": float(np.mean(negative_fp)) if negative_fp else np.nan,
    }


def train(config, resume=False):
    seed_everything(config["seed"])
    torch.set_num_threads(config["train"]["threads"])
    device = choose_device(config["device"])
    if device.type == "cuda" and config["train"]["cuda_memory_fraction"] is not None:
        torch.cuda.set_per_process_memory_fraction(config["train"]["cuda_memory_fraction"], device)
    if config["train"]["amp"] and device.type != "cuda":
        raise ValueError("AMP is supported here only on CUDA; use float32 on Mac")
    output = Path(config["train"]["output"]) / config["name"] / f"seed_{config['seed']}"
    if output.exists() and not resume:
        raise FileExistsError(f"Run exists: {output}. Use --resume or a new experiment name.")
    if resume and not (output / "last.pt").exists():
        raise FileNotFoundError("No last.pt to resume")
    settings = config["data"]
    train_data = FractureDataset(settings["manifest"], "train", settings["task"], settings["size"], settings["augment"], settings["contrast_augment"])
    val_data = FractureDataset(settings["manifest"], "val", settings["task"], settings["size"])
    # Eligibility checks happen before freezing so a missing pilot does not lock an empty cohort.
    if settings["task"] == "multiclass":
        freeze_annotations(settings["manifest"])
    provenance = fingerprint(settings["manifest"], settings["task"])
    generator = torch.Generator().manual_seed(config["seed"])
    sampler = None
    if settings["sampling"] == "balanced":
        counts = train_data.frame.fractured.value_counts()
        if len(counts) != 2:
            raise ValueError("Balanced sampling requires positive and negative training images")
        weights = [1 / counts[int(label)] for label in train_data.frame.fractured]
        sampler = WeightedRandomSampler(weights, len(train_data), replacement=True, generator=generator)
    loader = DataLoader(train_data, batch_size=config["train"]["batch_size"], shuffle=sampler is None, sampler=sampler,
                        num_workers=settings["workers"], collate_fn=collate, generator=generator,
                        persistent_workers=settings["workers"] > 0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=config["train"]["batch_size"], shuffle=False,
                            num_workers=settings["workers"], collate_fn=collate,
                            persistent_workers=settings["workers"] > 0, pin_memory=device.type == "cuda")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["train"]["amp"])
    start_epoch, best, stale, history = 0, -float("inf"), 0, []
    if resume:
        state = load_checkpoint(output / "last.pt")
        old, new = deepcopy(state["config"]), deepcopy(config)
        old["train"].pop("epochs")
        new["train"].pop("epochs")
        if old != new or state["provenance"] != provenance:
            raise ValueError("Resume requires unchanged configuration/data; only epoch limit may change")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # Move Adam state tensors to the chosen accelerator after CPU checkpoint loading.
        for item in optimizer.state.values():
            for key, value in item.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    item[key] = value.to(device)
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"], generator, device)
        start_epoch, best, stale, history = state["epoch"] + 1, state["best"], state["stale"], state["history"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    hardware = environment(device)
    hardware["cuda_memory_fraction"] = config["train"]["cuda_memory_fraction"]
    hardware["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    save_json(output / "environment.json", hardware)
    save_json(output / "data_provenance.json", provenance)
    (output / "manifest_snapshot.csv").write_bytes(Path(settings["manifest"]).read_bytes())
    save_json(output / "eligible_images.json", {
        "train": train_data.frame.image_id.tolist(), "val": val_data.frame.image_id.tolist(),
        "test": [r["image_id"] for r in provenance["records"] if r["split"] == "test"],
    })
    for epoch in range(start_epoch, config["train"]["epochs"]):
        if stale >= config["train"]["patience"]:
            break
        model.train()
        totals = defaultdict(float)
        examples = 0
        gradient_examples = 0
        skipped_steps = 0
        synchronize(device)
        start = time.perf_counter()
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config["train"]["amp"]):
                logits = model(batch["image"])
                # Losses use float32, including during CUDA autocast.
                loss, terms = objective(logits.float(), batch, config)
                if settings["dual_view"] or config["loss"]["consistency"]:
                    alternate = model(contrast_view(batch["image"], batch["valid"]))
                    alternate_loss, alternate_terms = objective(alternate.float(), batch, config)
                    js = js_divergence(logits.float(), alternate.float(), batch["valid"])
                    loss = (loss + alternate_loss) / 2 + config["loss"]["consistency"] * js
                    terms = {key: (value + alternate_terms[key]) / 2 for key, value in terms.items()}
                    terms["consistency"] = float(js.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; stop and inspect masks/configuration")
            scaler.scale(loss).backward()
            norm, skipped = optimizer_step(model, optimizer, scaler, config["train"]["amp"])
            skipped_steps += int(skipped)
            n = len(batch["image"])
            examples += n
            totals["loss"] += float(loss.detach()) * n
            if torch.isfinite(norm):
                totals["gradient_norm"] += float(norm) * n
                gradient_examples += n
            for key, value in terms.items():
                totals[key] += value * n
        synchronize(device)
        elapsed = time.perf_counter() - start
        metrics = validate_model(model, val_loader, device)
        score = metrics[config["train"]["selection"]]
        if not np.isfinite(score):
            raise ValueError("Checkpoint-selection metric undefined: validation requires positive fractures")
        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        record = {"epoch": epoch, **{k: v / examples for k, v in totals.items()},
                  **metrics, "train_seconds": elapsed, "best_score": best}
        record["gradient_norm"] = totals["gradient_norm"] / gradient_examples if gradient_examples else np.nan
        record["amp_skipped_steps"] = skipped_steps
        record["successful_optimizer_steps"] = len(loader) - skipped_steps
        history.append(record)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                   "config": config, "provenance": provenance, "epoch": epoch, "best": best, "stale": stale,
                   "rng": rng_state(generator, device), "history": history, "device": str(device)}
        save_checkpoint(output / "last.pt", payload)
        if improved:
            save_checkpoint(output / "best.pt", payload)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(json.dumps({"epoch": epoch, "loss": record["loss"], "val_fracture_dice": score, "seconds": elapsed}), flush=True)
    save_json(output / "training_summary.json", {"epochs_completed": len(history), "best_validation_score": best,
                                               "early_stopped": stale >= config["train"]["patience"]})
    return output
