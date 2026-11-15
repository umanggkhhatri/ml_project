from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapeguard.data.dataset import FractureDataset
from shapeguard.engine.checkpoint import load_checkpoint


def overlay(image, mask, task="multiclass"):
    base = image.astype(np.float32)
    if base.max() <= 1:
        base *= 255
    rgb = np.repeat(base[:, :, None], 3, axis=2)
    colors = {1: [40, 220, 100], 2: [255, 40, 60]} if task == "multiclass" else {1: [255, 40, 60]}
    for cls, color in colors.items():
        selected = mask == cls
        rgb[selected] = rgb[selected] * 0.55 + np.asarray(color) * 0.45
    return np.clip(rgb, 0, 255).astype(np.uint8)


def figures(checkpoint, evaluation):
    checkpoint, evaluation = Path(checkpoint), Path(evaluation)
    state = load_checkpoint(checkpoint)
    config = state["config"]
    output = evaluation / "figures"
    output.mkdir(exist_ok=True)
    history = pd.read_csv(checkpoint.parent / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.epoch, history.loss, label="Training objective")
    axes[1].plot(history.epoch, history.fracture_positive_dice, label="Validation fracture-positive Dice")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)
    metrics = pd.read_csv(evaluation / "fracture_metrics.csv")
    positives = metrics[metrics.target_pixels > 0].sort_values("dice")
    # Deterministic best/worst examples, clearly labeled; not cherry-picked claims.
    chosen = pd.concat([positives.head(2), positives.tail(2)]).drop_duplicates("image_id")
    dataset = FractureDataset(config["data"]["manifest"], evaluation.name.removeprefix("evaluation_"), config["data"]["task"], config["data"]["size"])
    by_id = {row.image_id: i for i, row in enumerate(dataset.frame.itertuples())}
    if len(chosen):
        fig, axes = plt.subplots(len(chosen), 3, figsize=(10, len(chosen) * 3), squeeze=False)
        for r, row in enumerate(chosen.itertuples()):
            sample = dataset[by_id[row.image_id]]
            image = sample["image"][0].numpy()
            pred = np.load(evaluation / "predictions" / f"{row.image_id}.npy")
            target = sample["mask"].numpy()
            for axis, shown, title in zip(axes[r], [image, overlay(image, target, config["data"]["task"]), overlay(image, pred, config["data"]["task"])], ["Input", "Annotation", f"Prediction: fracture Dice {row.dice:.3f}"]):
                axis.imshow(shown, cmap="gray")
                axis.set_title(f"{row.image_id}\n{title}", fontsize=9)
                axis.axis("off")
        fig.tight_layout()
        fig.savefig(output / "best_worst_examples.png", dpi=180)
        plt.close(fig)
    instances = pd.read_csv(evaluation / "instances.csv")
    if len(instances):
        grouped = instances.groupby("size_group").matched_dice.mean().reindex(["small", "medium", "large"])
        fig, axis = plt.subplots(figsize=(6, 4))
        grouped.plot.bar(ax=axis)
        axis.set_ylabel("One-to-one matched Dice (misses = 0)")
        fig.tight_layout()
        fig.savefig(output / "size_analysis.png", dpi=180)
        plt.close(fig)
    robustness = evaluation / "robustness.csv"
    if robustness.exists():
        table = pd.read_csv(robustness)
        table = table[table.target_pixels > 0]
        fig, axis = plt.subplots(figsize=(7, 4))
        for kind, group in table.groupby("kind"):
            curve = group.groupby("severity").dice.mean()
            axis.plot(curve.index, curve.values, marker="o", label=kind)
        axis.set_xlabel("Synthetic corruption severity")
        axis.set_ylabel("Fracture-positive Dice")
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / "robustness.png", dpi=180)
        plt.close(fig)
    return output
