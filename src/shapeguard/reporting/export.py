"""Publishable numerical artifacts, excluding images, checkpoints and private paths."""

import json
import shutil
from pathlib import Path

from shapeguard.reproducibility import save_json, sha256


def export_metrics(study="results/binary_study"):
    study = Path(study)
    status = json.loads((study / "study_status.json").read_text())
    if not status.get("completed"):
        raise ValueError("Only completed studies can be exported")
    records = []
    for entry in status["runs"]:
        evaluation = Path(entry["evaluation"])
        run = evaluation.parent
        destination = study / "experiments" / entry["experiment"] / f"seed_{entry['seed']}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("history.csv", "config.yaml", "training_summary.json"):
            source = run / name
            shutil.copy2(source, destination / name)
            records.append({"source": str(source), "export": str(destination / name), "sha256": sha256(source)})
        for name in ("class_summary.csv", "per_class_metrics.csv", "fracture_metrics.csv", "instances.csv", "detections.csv", "resize_audit.csv", "robustness.csv"):
            source = evaluation / name
            if source.exists():
                shutil.copy2(source, destination / name)
                records.append({"source": str(source), "export": str(destination / name), "sha256": sha256(source)})
        summary = json.loads((evaluation / "summary.json").read_text())
        # Correct a legacy descriptive field only; never change any numerical value.
        summary.pop("bone_annotation_note", None)
        summary["annotation_provenance"] = "Original FracAtlas fracture polygons; no student bone annotations used in the binary study."
        save_json(destination / "summary.json", summary)
    save_json(study / "export_manifest.json", {"files": records, "summary_metadata_correction": "Removed legacy bone_annotation_note and clarified binary annotation provenance. Numerical results unchanged.", "excluded": ["raw dataset images", "predicted mask arrays", "checkpoints", "SSH credentials and commands"]})
    return study / "experiments"
