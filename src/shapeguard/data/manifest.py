"""Portable paths and immutable hashes for experiment provenance."""

from pathlib import Path

import pandas as pd

from shapeguard.reproducibility import save_json, sha256

PATH_COLUMNS = ("image_path", "fracture_path", "instance_path", "bone_path", "mask_path")


def read_manifest(path):
    path = Path(path).resolve()
    frame = pd.read_csv(path).fillna("")
    if frame.image_id.duplicated().any():
        raise ValueError("Manifest contains duplicate image IDs")
    for column in PATH_COLUMNS:
        frame[column] = [str((path.parent / p).resolve()) if p else "" for p in frame[column]]
    return frame


def fingerprint(manifest, task):
    frame = read_manifest(manifest)
    if not set(frame.split).issubset({"train", "val", "test"}):
        raise ValueError("Every record needs an assigned split")
    if frame.groupby("group_id").split.nunique().max() > 1:
        raise ValueError("A duplicate group appears in multiple splits")
    if task == "multiclass":
        frame = frame[frame.bone_status == "accepted"]
    records = []
    for row in frame.itertuples():
        record = {"image_id": row.image_id, "split": row.split, "group_id": row.group_id}
        if task == "multiclass":
            record["bone_status"] = row.bone_status
        columns = ("image_path", "fracture_path", "instance_path", "mask_path") if task == "multiclass" else ("image_path", "fracture_path", "instance_path")
        for column in columns:
            value = getattr(row, column)
            if value:
                record[column + "_sha256"] = sha256(value)
        records.append(record)
    if not records:
        raise ValueError("No eligible images")
    return {"manifest_sha256": sha256(manifest), "task": task, "records": records}


def freeze_annotations(manifest):
    path = Path(manifest).resolve()
    marker = path.parent / "annotations_frozen.json"
    frame = read_manifest(path)
    ids = sorted(frame.loc[frame.bone_status == "accepted", "image_id"].tolist())
    if marker.exists():
        import json
        frozen = json.loads(marker.read_text())
        if frozen["image_ids"] != ids:
            raise ValueError("Accepted annotation cohort changed after freezing")
        for record in frozen["hashes"]:
            row = frame[frame.image_id == record["image_id"]].iloc[0]
            if sha256(row.bone_path) != record["bone_sha256"] or sha256(row.mask_path) != record["mask_sha256"]:
                raise ValueError(f"Frozen annotation changed: {row.image_id}")
        return
    save_json(marker, {"image_ids": ids, "hashes": [
        {"image_id": row.image_id, "bone_sha256": sha256(row.bone_path), "mask_sha256": sha256(row.mask_path)}
        for row in frame[frame.bone_status == "accepted"].itertuples()
    ], "policy": "Accepted cohort and masks frozen before model comparisons."})
