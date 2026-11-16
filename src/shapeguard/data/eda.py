"""Dataset EDA and proposal contact sheets with explicit annotation status."""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from shapeguard.data.manifest import read_manifest
from shapeguard.reporting.figures import overlay
from shapeguard.reproducibility import save_json


def eda(manifest, selection=None):
    manifest = Path(manifest).resolve()
    frame = read_manifest(manifest)
    output = manifest.parent / "eda"
    output.mkdir(exist_ok=True)
    split_counts = frame.groupby(["split", "fractured"]).size().unstack(fill_value=0)
    split_counts.to_csv(output / "split_counts.csv")
    anatomy = frame.groupby("split")[["hand", "leg", "hip", "shoulder", "hardware", "multiscan"]].sum()
    anatomy.to_csv(output / "anatomy_counts.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    frame.fractured.value_counts().reindex([0, 1]).plot.bar(ax=axes[0, 0])
    axes[0, 0].set_title("Usable images: fracture presence")
    axes[0, 0].set_xticklabels(["Negative", "Positive"], rotation=0)
    axes[0, 1].hist(frame.width, bins=30, alpha=0.6, label="Width")
    axes[0, 1].hist(frame.height, bins=30, alpha=0.6, label="Height")
    axes[0, 1].set_title("Native dimensions (pixels)")
    axes[0, 1].legend()
    positives = frame[frame.fractured == 1]
    axes[1, 0].hist(np.log10(positives.fracture_fraction.clip(lower=1e-8)), bins=30)
    axes[1, 0].set_title("Positive-image fracture fraction (log10)")
    anatomy.T.plot.bar(ax=axes[1, 1])
    axes[1, 1].set_title("Attributes by split (not mutually exclusive)")
    fig.tight_layout()
    fig.savefig(output / "dataset_eda.png", dpi=160)
    plt.close(fig)
    source_audit = manifest.parent / "audit.json"
    reasons = Counter()
    if source_audit.exists():
        audit = json.loads(source_audit.read_text())
        reasons = Counter(record["reason"].split(":")[0] for record in audit["quarantined"])
    save_json(output / "summary.json", {
        "usable_images": len(frame), "positive_images": int(frame.fractured.sum()),
        "original_polygon_instances": int(frame.instance_count.sum()),
        "accepted_bone_annotations": int((frame.bone_status == "accepted").sum()),
        "quarantine_reasons": dict(reasons), "patient_level_split": False,
        "native_median_width": float(frame.width.median()), "native_median_height": float(frame.height.median()),
    })
    if selection:
        ids = json.loads(Path(selection).read_text())["image_ids"]
        selected = frame[frame.image_id.isin(ids)].copy()
        fig, axes = plt.subplots(len(selected), 2, figsize=(9, max(1, len(selected)) * 3), squeeze=False)
        for i, row in enumerate(selected.itertuples()):
            image = np.asarray(Image.open(row.image_path).convert("L"))
            fracture = np.asarray(Image.open(row.fracture_path)) > 0
            proposal_path = manifest.parent / "proposals" / f"{row.image_id}.png"
            mask = np.zeros(image.shape, np.uint8)
            if proposal_path.exists():
                mask[np.asarray(Image.open(proposal_path)) > 0] = 1
            mask[fracture] = 2
            shown = overlay(image, mask)
            axes[i, 0].imshow(image, cmap="gray")
            axes[i, 1].imshow(shown)
            axes[i, 0].set_title(f"{row.image_id} | {row.split} | fractured={row.fractured}", fontsize=9)
            axes[i, 1].set_title("UNACCEPTED bone proposal + original fracture", fontsize=9)
            for axis in axes[i]:
                axis.axis("off")
        fig.tight_layout()
        fig.savefig(output / "pilot_proposals.png", dpi=100)
        plt.close(fig)
    return output
