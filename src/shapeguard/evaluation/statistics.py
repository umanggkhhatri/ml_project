import numpy as np
import pandas as pd


def paired_bootstrap(first, second, metric="dice", samples=2000, seed=42, positive_only=False):
    first, second = pd.read_csv(first), pd.read_csv(second)
    for frame in (first, second):
        if frame["image_id"].duplicated().any():
            raise ValueError("Expected one fracture record per image; filter the per-class CSV first")
    joined = first.merge(second, on="image_id", suffixes=("_a", "_b"), validate="one_to_one")
    if len(joined) != len(first) or len(joined) != len(second):
        raise ValueError("Paired runs must evaluate exactly the same images")
    if not (joined.group_id_a == joined.group_id_b).all():
        raise ValueError("Duplicate-group definitions differ between runs")
    if "target_pixels_a" in joined and not (joined.target_pixels_a == joined.target_pixels_b).all():
        raise ValueError("Paired annotation targets differ")
    if positive_only:
        joined = joined[joined.target_pixels_a > 0].reset_index(drop=True)
    if joined.empty or samples < 1:
        raise ValueError("Need paired images and a positive bootstrap sample count")
    difference = joined[f"{metric}_b"] - joined[f"{metric}_a"]
    if not np.isfinite(difference).all():
        raise ValueError("Bootstrap metric must be finite on every paired image")
    groups = list(joined.groupby("group_id_a").indices.values())
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in selected])
        estimates.append(float(difference.iloc[indices].mean()))
    return {
        "metric": metric, "difference_b_minus_a": float(difference.mean()),
        "ci95": np.percentile(estimates, [2.5, 97.5]).tolist(),
        "images": len(joined), "resampling_units": len(groups), "samples": samples, "seed": seed,
        "unit": "duplicate group (not patient; patient identifiers unavailable)",
        "positive_only": positive_only,
    }
