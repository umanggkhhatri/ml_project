import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from shapeguard.annotation.server import save_annotation
from shapeguard.data.annotations import combine_masks, expanded_box, rasterize_polygons
from shapeguard.data.audit import prepare
from shapeguard.data.dataset import FractureDataset, collate
from shapeguard.data.download import extract_zip
from shapeguard.data.manifest import fingerprint, freeze_annotations
from shapeguard.data.transforms import letterbox
from shapeguard.smoke import synthetic_dataset


def test_polygon_and_precedence():
    mask = rasterize_polygons([[1, 1, 4, 1, 4, 4, 1, 4]], 8, 8)
    assert mask[2, 2] == 1 and mask[0, 0] == 0
    bone = np.zeros((8, 8), np.uint8)
    bone[5:7, 5:7] = 1
    combined = combine_masks(bone, mask)
    assert combined[2, 2] == 2  # Do not trim fracture polygons to bone.
    assert combined[5, 5] == 1


@pytest.mark.parametrize("polygon", [[1, 2, 3], [0, 0, 1, 1, float("nan"), 2], [-1, 0, 1, 0, 1, 1]])
def test_bad_polygons_rejected(polygon):
    with pytest.raises(ValueError):
        rasterize_polygons([polygon], 8, 8)


def test_zip_slip_blocked(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError):
        extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_letterbox_alignment_and_ignore_padding():
    image = np.zeros((10, 20), np.float32)
    image[:, 5:10] = 1
    mask = image.astype(np.uint8)
    out, target, instances, valid, meta = letterbox(image, mask, mask[None], 20)
    assert out.shape == target.shape == valid.shape == (20, 20)
    assert np.all(target[~valid] == 255)
    assert np.array_equal(target[valid], instances[0][valid])
    assert meta["top"] == 5 and meta["left"] == 0


def test_geometry_preserves_aspect_ratio():
    mask = np.zeros((100, 100), bool)
    mask[40:45, 10:90] = True
    y0, y1, x0, x1 = expanded_box(mask)
    assert x1 - x0 > 5 * (y1 - y0)
    assert y0 < 40 and y1 > 45


def test_dataset_excludes_draft_bone_masks(tmp_path):
    path = synthetic_dataset(tmp_path)
    frame = pd.read_csv(path).fillna("")
    frame.loc[0, "bone_status"] = "draft"
    frame.to_csv(path, index=False)
    multi = FractureDataset(path, "train", size=48)
    binary = FractureDataset(path, "train", task="binary", size=48)
    assert len(multi) == 7 and len(binary) == 8
    batch = collate([multi[0], multi[1]])
    assert batch["image"].shape == (2, 1, 48, 48)
    assert isinstance(batch["instances"], list)


def test_freeze_rejects_changed_masks(tmp_path):
    path = synthetic_dataset(tmp_path)
    freeze_annotations(path)
    freeze_annotations(path)
    mask = tmp_path / "SYNTHETIC_000_bone.png"
    Image.fromarray(np.zeros((48, 64), np.uint8)).save(mask)
    with pytest.raises(ValueError, match="changed"):
        freeze_annotations(path)


def test_fingerprint_detects_changes(tmp_path):
    path = synthetic_dataset(tmp_path)
    first = fingerprint(path, "multiclass")
    mask = tmp_path / "SYNTHETIC_000_mask.png"
    Image.fromarray(np.zeros((48, 64), np.uint8)).save(mask)
    assert first != fingerprint(path, "multiclass")


def test_annotation_acceptance_and_freezing(tmp_path):
    path = synthetic_dataset(tmp_path)
    frame = pd.read_csv(path).fillna("")
    frame.loc[0, "bone_status"] = "missing"
    frame.to_csv(path, index=False)
    bone = np.zeros((48, 64), np.uint8)
    bone[8:40, 12:52] = 255
    buffer = io.BytesIO()
    Image.fromarray(bone).save(buffer, format="PNG")
    record = save_annotation(path, "SYNTHETIC_000", buffer.getvalue(), accepted=True)
    assert record["status"] == "accepted"
    with pytest.raises(ValueError, match="already accepted"):
        save_annotation(path, "SYNTHETIC_000", buffer.getvalue())
    freeze_annotations(path)
    with pytest.raises(ValueError, match="frozen"):
        save_annotation(path, "SYNTHETIC_001", buffer.getvalue())
    assert json.loads((tmp_path / "annotations_frozen.json").read_text())["image_ids"]


def test_source_audit_quarantines_corruption_and_deduplicates_files(tmp_path):
    source = tmp_path / "source"
    for folder in ("images/Fractured", "images/Non_fractured", "Annotations/COCO JSON"):
        (source / folder).mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(source / "images/Fractured/positive.jpg")
    # Same filename and identical contents in both source folders must not abort the audit.
    (source / "images/Non_fractured/positive.jpg").write_bytes((source / "images/Fractured/positive.jpg").read_bytes())
    Image.fromarray(np.ones((8, 8), np.uint8)).save(source / "images/Non_fractured/negative.jpg")
    (source / "images/Non_fractured/corrupt.jpg").write_bytes(b"not a jpeg")
    pd.DataFrame({"image_id": ["positive.jpg", "negative.jpg", "corrupt.jpg"], "fractured": [1, 0, 0]}).to_csv(source / "dataset.csv", index=False)
    (source / "Annotations/COCO JSON/masks.json").write_text(json.dumps({
        "images": [{"id": 0, "file_name": "positive.jpg", "width": 8, "height": 8}],
        "annotations": [{"id": 0, "image_id": 0, "segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]]}],
    }))
    output = tmp_path / "prepared"
    path = prepare(source, output)
    assert len(pd.read_csv(path)) == 2
    audit = json.loads((output / "audit.json").read_text())
    assert audit["duplicate_source_files"][0]["identical"]
    assert "unreadable_image" in audit["quarantined"][0]["reason"]
