import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from shapeguard.data.manifest import read_manifest
from shapeguard.reproducibility import choose_device, save_json, sha256


def threshold_proposal(image):
    """Cheap brightness proposal ONLY. May include soft tissue/text or omit bones."""
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (mask > 0).astype(np.uint8)


def generate(manifest, selection, method="threshold", checkpoint=None, prompts=None, model_type="vit_b", device="auto"):
    manifest = Path(manifest).resolve()
    frame = read_manifest(manifest)
    ids = json.loads(Path(selection).read_text())["image_ids"]
    selected = frame[frame.image_id.isin(ids)]
    if len(selected) != len(set(ids)):
        raise ValueError("Selection contains unknown or repeated IDs")
    directory = manifest.parent / "proposals"
    directory.mkdir(exist_ok=True)
    predictor = None
    prompt_data = {}
    if method == "sam":
        if not checkpoint or not prompts:
            raise ValueError("SAM requires an explicit checkpoint and image-specific prompt JSON")
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as error:
            raise RuntimeError("Install optional SAM dependencies: uv sync --extra sam") from error
        model = sam_model_registry[model_type](checkpoint=checkpoint).to(choose_device(device))
        predictor = SamPredictor(model)
        prompt_data = json.loads(Path(prompts).read_text())
    elif method != "threshold":
        raise ValueError("Proposal method must be threshold or sam")
    generated = []
    for row in selected.itertuples():
        path = directory / f"{row.image_id}.png"
        if path.exists():
            raise FileExistsError(f"Proposal exists: {path}. Use the editor rather than overwriting it.")
        image = np.asarray(Image.open(row.image_path).convert("L"))
        if method == "threshold":
            mask = threshold_proposal(image)
        else:
            prompt = prompt_data.get(row.image_id)
            if not prompt:
                raise ValueError(f"Missing SAM prompts for {row.image_id}")
            predictor.set_image(np.repeat(image[:, :, None], 3, axis=2))
            masks = []
            for box in prompt.get("boxes", []):
                box = np.asarray(box, np.float32)
                if box.shape != (4,) or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= image.shape[1] and 0 <= box[1] < box[3] <= image.shape[0]):
                    raise ValueError(f"Invalid native-coordinate box for {row.image_id}")
                with torch.inference_mode():
                    result, _, _ = predictor.predict(box=box, multimask_output=False)
                masks.append(result[0])
            if not masks:
                raise ValueError("Supply one or more native-coordinate bone boxes per image")
            mask = np.maximum.reduce(masks).astype(np.uint8)
        Image.fromarray(mask * 255).save(path)
        save_json(directory / f"{row.image_id}.json", {
            "image_id": row.image_id, "method": method, "status": "proposal_only",
            "source_image_sha256": row.sha256,
            "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
            "prompts": prompt_data.get(row.image_id),
            "warning": "Not an accepted bone annotation. Inspect and correct before training.",
        })
        generated.append(row.image_id)
    return generated
