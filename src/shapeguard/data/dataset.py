"""Only explicitly accepted bone annotations enter multiclass training."""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from shapeguard.data.annotations import validate_mask
from shapeguard.data.manifest import read_manifest
from shapeguard.data.transforms import augment, letterbox, normalize


class FractureDataset(Dataset):
    def __init__(self, manifest, split, task="multiclass", size=384, augment_data=False, contrast=False):
        frame = read_manifest(manifest)
        frame = frame[frame["split"] == split].copy()
        if task == "multiclass":
            frame = frame[(frame["bone_status"] == "accepted") & (frame["mask_path"] != "")]
        self.frame = frame.reset_index(drop=True)
        self.task, self.size, self.augment_data, self.contrast = task, size, augment_data, contrast
        if self.frame.empty:
            raise ValueError(f"No usable {split} images for {task}. Assign splits and accept bone annotations first.")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as handle:
            image = np.asarray(handle.convert("L"))
        path = row.mask_path if self.task == "multiclass" else row.fracture_path
        mask = np.asarray(Image.open(path)).copy()
        if self.task == "binary":
            mask = (mask > 0).astype(np.uint8)
        validate_mask(mask, 3 if self.task == "multiclass" else 2)
        if mask.shape != image.shape:
            raise ValueError(f"Native mask/image mismatch: {row.image_id}")
        with np.load(row.instance_path) as handle:
            instances = handle["masks"]
        if instances.shape[1:] != image.shape:
            raise ValueError(f"Native instance/image mismatch: {row.image_id}")
        image, mask, instances, valid, meta = letterbox(normalize(image), mask, instances, self.size)
        if self.augment_data:
            image, mask, instances, valid = augment(image, mask, instances, valid, self.contrast)
        return {
            "image": torch.from_numpy(image.copy()).unsqueeze(0),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "instances": torch.from_numpy(instances.copy()).bool(),
            "valid": torch.from_numpy(valid.copy()),
            "image_id": str(row.image_id), "group_id": str(row.group_id), "meta": meta,
        }


def collate(samples):
    return {
        "image": torch.stack([s["image"] for s in samples]),
        "mask": torch.stack([s["mask"] for s in samples]),
        "valid": torch.stack([s["valid"] for s in samples]),
        "instances": [s["instances"] for s in samples],
        "image_id": [s["image_id"] for s in samples],
        "group_id": [s["group_id"] for s in samples],
        "meta": [s["meta"] for s in samples],
    }
