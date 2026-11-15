"""Evaluate only the frozen experiment cohort. Distances are at network resolution."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from shapeguard.data.dataset import FractureDataset
from shapeguard.data.manifest import fingerprint
from shapeguard.engine.checkpoint import load_checkpoint
from shapeguard.evaluation.efficiency import benchmark
from shapeguard.evaluation.instances import instance_metrics
from shapeguard.evaluation.metrics import binary_metrics
from shapeguard.evaluation.robustness import corrupt
from shapeguard.models import build_model
from shapeguard.reproducibility import choose_device, save_json, sha256, synchronize


def training_size_thresholds(dataset):
    fractions = []
    for index in range(len(dataset)):
        sample = dataset[index]
        denominator = sample["valid"].sum().item()
        fractions.extend(float(m.sum()) / denominator for m in sample["instances"] if m.any())
    if not fractions:
        raise ValueError("No positive fracture instances in training cohort")
    return np.quantile(fractions, [1 / 3, 2 / 3]).tolist()


def evaluate(checkpoint, device_request="auto", split="test", robustness=False):
    checkpoint = Path(checkpoint)
    state = load_checkpoint(checkpoint)
    config = state["config"]
    torch.set_num_threads(config["train"]["threads"])
    device = choose_device(device_request)
    settings = config["data"]
    current = fingerprint(settings["manifest"], settings["task"])
    if current != state["provenance"]:
        raise ValueError("Dataset/annotations changed since training; evaluation refused")
    dataset = FractureDataset(settings["manifest"], split, settings["task"], settings["size"])
    train_dataset = FractureDataset(settings["manifest"], "train", settings["task"], settings["size"])
    thresholds = training_size_thresholds(train_dataset)
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    directory = checkpoint.parent / f"evaluation_{split}"
    if directory.exists():
        raise FileExistsError(f"Evaluation already exists: {directory}; use another run/checkpoint directory")
    directory.mkdir(parents=True)
    (directory / "predictions").mkdir()
    efficiency = benchmark(model, device, settings["size"])
    efficiency["checkpoint_bytes"] = checkpoint.stat().st_size
    records, instance_records, detection_records, timings, resizing = [], [], [], [], []
    for index in range(len(dataset)):
        synchronize(device)
        start = time.perf_counter()
        sample = dataset[index]
        with torch.inference_mode():
            logits = model(sample["image"][None].to(device))
            prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
        np.save(directory / "predictions" / f"{sample['image_id']}.npy", prediction)
        target, valid = sample["mask"].numpy(), sample["valid"].numpy()
        resizing.append({"image_id": sample["image_id"], **sample["meta"]})
        names = ["background", "bone", "fracture"] if settings["task"] == "multiclass" else ["background", "fracture"]
        for cls, name in enumerate(names):
            metrics = binary_metrics(prediction == cls, target == cls, valid)
            records.append({"image_id": sample["image_id"], "group_id": sample["group_id"], "class": name, **metrics})
        fracture_class = len(names) - 1
        local, detection = instance_metrics(prediction == fracture_class, sample["instances"].numpy(), thresholds, valid)
        instance_records.extend({"image_id": sample["image_id"], **r} for r in local)
        detection_records.append({"image_id": sample["image_id"], **detection})
    frame = pd.DataFrame(records)
    frame.to_csv(directory / "per_class_metrics.csv", index=False)
    fracture = frame[frame["class"] == "fracture"].copy()
    fracture.to_csv(directory / "fracture_metrics.csv", index=False)
    pd.DataFrame(instance_records).to_csv(directory / "instances.csv", index=False)
    pd.DataFrame(detection_records).to_csv(directory / "detections.csv", index=False)
    pd.DataFrame(resizing).to_csv(directory / "resize_audit.csv", index=False)
    class_summary = []
    for name, group in frame.groupby("class"):
        record = {"class": name, "images": len(group)}
        for metric in ("dice", "iou", "precision", "recall", "pixel_accuracy"):
            record[metric] = float(group[metric].mean())
        positive = group[group.target_pixels > 0]
        record["positive_dice"] = float(positive.dice.mean())
        tp, fp, fn = group.tp.sum(), group.fp.sum(), group.fn.sum()
        record["pooled_precision"] = float(tp / (tp + fp)) if tp + fp else np.nan
        record["pooled_recall"] = float(tp / (tp + fn)) if tp + fn else np.nan
        record["pooled_dice"] = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0
        for metric in ("hausdorff", "hd95"):
            finite = group[np.isfinite(group[metric])]
            record[metric + "_finite_mean"] = float(finite[metric].mean())
            record[metric + "_infinite_cases"] = int(np.isinf(group[metric]).sum())
            positive_finite = positive[np.isfinite(positive[metric])]
            record[metric + "_positive_finite_mean"] = float(positive_finite[metric].mean())
            record[metric + "_positive_infinite_cases"] = int(np.isinf(positive[metric]).sum())
        class_summary.append(record)
    pd.DataFrame(class_summary).to_csv(directory / "class_summary.csv", index=False)
    negatives = fracture[fracture.target_pixels == 0]
    positives = fracture[fracture.target_pixels > 0]
    efficiency["preprocessing_model_transfer_median_ms"] = float(np.median(timings))
    efficiency["inclusive_timing_note"] = "Includes image, semantic/instance-mask loading, preprocessing, forward pass and prediction transfer; excludes writing predictions and metrics."
    summary = {
        "experiment": config["name"], "seed": config["seed"], "task": settings["task"], "split": split,
        "checkpoint_sha256": sha256(checkpoint), "manifest_sha256": current["manifest_sha256"],
        "cohort_sha256": hashlib.sha256(json.dumps(current["records"], sort_keys=True).encode()).hexdigest(),
        "fracture_positive_dice": float(positives.dice.mean()), "fracture_dice_all_images": float(fracture.dice.mean()),
        "fracture_positive_iou": float(positives.iou.mean()),
        "negative_image_fp_rate": float((negatives.prediction_pixels > 0).mean()),
        "size_thresholds_training_only": thresholds, "efficiency": efficiency,
        "empty_policy": "Both empty: Dice/IoU=1, HD=0. One empty: HD=inf; report infinite counts separately. Undefined precision/recall are NaN, not silently zero.",
        "distance_units": f"pixels at {settings['size']}x{settings['size']} letterboxed network resolution; not millimetres",
        "bone_annotation_note": "Bone masks are accepted student/AI-assisted annotations, not clinically validated ground truth.",
    }
    save_json(directory / "summary.json", summary)
    if robustness:
        robustness_rows = []
        for kind in ("contrast", "gamma", "noise", "blur"):
            for severity in (1, 2, 3):
                for index in range(len(dataset)):
                    sample = dataset[index]
                    # Stable image-based seed ensures the same corruption across experiments.
                    seed = int(hashlib.sha256(f"{config['seed']}:{sample['image_id']}:{kind}:{severity}".encode()).hexdigest()[:8], 16)
                    image = corrupt(sample["image"][0].numpy(), kind, severity, np.random.default_rng(seed))
                    image[~sample["valid"].numpy()] = 0
                    with torch.inference_mode():
                        pred = model(torch.from_numpy(image.copy())[None, None].to(device)).argmax(1)[0].cpu().numpy()
                    target = sample["mask"].numpy() == fracture_class
                    metrics = binary_metrics(pred == fracture_class, target, sample["valid"].numpy(), distances=False)
                    robustness_rows.append({"image_id": sample["image_id"], "kind": kind, "severity": severity, **metrics})
        pd.DataFrame(robustness_rows).to_csv(directory / "robustness.csv", index=False)
    return directory
