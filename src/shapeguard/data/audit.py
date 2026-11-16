"""Join source labels and polygons; quarantine inconsistent records rather than guessing."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from shapeguard.data.annotations import rasterize_polygons
from shapeguard.reproducibility import save_json, sha256


def image_hash(image):
    """64-bit perceptual difference hash used only to flag candidates, not diagnose duplicates."""
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    return sum(int(bit) << i for i, bit in enumerate(bits.ravel()))


def prepare(source, output="data/prepared"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (output / "manifest.csv").exists():
        raise FileExistsError("Prepared manifest exists; use a new output directory to preserve annotations")
    csv_files = list(source.rglob("dataset.csv"))
    if len(csv_files) != 1:
        raise ValueError("Expected exactly one source dataset.csv")
    table = pd.read_csv(csv_files[0])
    candidates = []
    for path in source.rglob("*.json"):
        if "coco" in str(path).lower():
            content = json.loads(path.read_text())
            if isinstance(content, dict) and "images" in content and "annotations" in content:
                candidates.append((path, content))
    if len(candidates) != 1:
        raise ValueError(f"Expected one COCO polygon file, found {len(candidates)}")
    coco_path, coco = candidates[0]
    images_by_id = {image["id"]: image for image in coco["images"]}
    ann_by_name = {}
    orphan_annotations = []
    for ann in coco["annotations"]:
        image = images_by_id.get(ann["image_id"])
        if image is None:
            orphan_annotations.append(ann["id"])
            continue
        name = Path(image["file_name"]).name
        ann_by_name.setdefault(name, []).append((ann, image))
    image_paths, duplicate_files, ambiguous_names = {}, [], set()
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if path.name in image_paths:
                existing = image_paths[path.name]
                identical = sha256(existing) == sha256(path)
                duplicate_files.append({"filename": path.name, "paths": [str(existing), str(path)], "identical": identical})
                if not identical:
                    ambiguous_names.add(path.name)
                continue
            image_paths[path.name] = path
    name_column = next((c for c in ("image_id", "image", "filename", "file_name") if c in table), None)
    if name_column is None or "fractured" not in table:
        raise ValueError(f"Unsupported source CSV columns: {list(table.columns)}")
    if table[name_column].duplicated().any():
        raise ValueError("Duplicate source CSV image IDs require resolution before preparation")
    source_metadata = source.parent / f"{source.name}.json"
    version = json.loads(source_metadata.read_text()).get("metadata", {}).get("version") if source_metadata.exists() else None
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("fracture_masks", "instances", "bone_masks", "multiclass_masks", "proposals"):
        (output / folder).mkdir(exist_ok=True)
    rows, issues, hashes = [], [], []
    groups = {}
    for _, record in table.iterrows():
        name = Path(str(record[name_column])).name
        if not Path(name).suffix:
            name += ".jpg"
        if name in ambiguous_names:
            issues.append({"image": name, "reason": "same_filename_different_image_content"})
            continue
        path = image_paths.get(name)
        if path is None:
            issues.append({"image": name, "reason": "missing_image"})
            continue
        try:
            with Image.open(path) as im:
                width, height = im.size
                array = np.asarray(im.convert("L"))
                orientation = im.getexif().get(274, 1)
        except (OSError, ValueError) as error:
            issues.append({"image": name, "reason": f"unreadable_image: {error}"})
            continue
        if orientation != 1:
            issues.append({"image": name, "reason": "nontrivial_EXIF_orientation"})
            continue
        positive = bool(int(record["fractured"]))
        anns = ann_by_name.get(name, [])
        if positive != bool(anns):
            issues.append({"image": name, "reason": "classification_polygon_disagreement"})
            continue
        instance_masks = []
        try:
            for ann, meta in anns:
                if width != meta["width"] or height != meta["height"]:
                    raise ValueError("COCO/native dimension mismatch")
                mask = rasterize_polygons(ann["segmentation"], width, height)
                if not mask.any():
                    raise ValueError("empty positive polygon")
                instance_masks.append(mask)
        except ValueError as error:
            issues.append({"image": name, "reason": str(error)})
            continue
        stem = Path(name).stem
        union = np.maximum.reduce(instance_masks) if instance_masks else np.zeros((height, width), np.uint8)
        fracture_path = output / "fracture_masks" / f"{stem}.png"
        instance_path = output / "instances" / f"{stem}.npz"
        Image.fromarray(union).save(fracture_path)
        stack = np.stack(instance_masks) if instance_masks else np.zeros((0, height, width), np.uint8)
        np.savez_compressed(instance_path, masks=stack)
        digest = sha256(path)
        group = groups.setdefault(digest, stem)
        dhash = image_hash(array)
        hashes.append((stem, dhash, digest))
        row = {
            "image_id": stem, "image_path": os.path.relpath(path, output), "fracture_path": fracture_path.relative_to(output).as_posix(),
            "instance_path": instance_path.relative_to(output).as_posix(), "bone_path": "", "mask_path": "",
            "bone_status": "missing", "split": "", "group_id": group,
            "sha256": digest, "width": width, "height": height, "fractured": int(positive),
            "instance_count": len(instance_masks), "fracture_fraction": float(union.mean()),
            "source_version": version,
        }
        for column in ("hand", "leg", "hip", "shoulder", "hardware", "multiscan", "frontal", "lateral", "oblique"):
            row[column] = int(record.get(column, 0))
        rows.append(row)
    if not rows:
        raise ValueError("No usable source images; inspect source schema")
    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False)
    near_duplicates = []
    for i, (name, dhash, digest) in enumerate(hashes):
        for other, other_hash, other_digest in hashes[:i]:
            distance = (dhash ^ other_hash).bit_count()
            if distance <= 3 and digest != other_digest:
                near_duplicates.append({"image_a": name, "image_b": other, "hash_distance": distance})
    save_json(output / "audit.json", {
        "usable_images": len(rows), "positive_images": sum(r["fractured"] for r in rows),
        "instances": sum(r["instance_count"] for r in rows), "quarantined": issues,
        "orphan_annotations": orphan_annotations,
        "duplicate_source_files": duplicate_files,
        "unlisted_images": sorted(set(image_paths) - {Path(str(x)).name for x in table[name_column]}),
        "coco_sha256": sha256(coco_path), "source_csv_sha256": sha256(csv_files[0]),
        "near_duplicate_candidates": len(near_duplicates),
        "note": "Patient identifiers unavailable. Perceptual hashes flag candidates, not confirmed duplicates.",
    })
    pd.DataFrame(near_duplicates, columns=["image_a", "image_b", "hash_distance"]).to_csv(
        output / "near_duplicates.csv", index=False
    )
    return output / "manifest.csv"
