"""Conservative duplicate grouping and fixed, group-disjoint splits."""

from pathlib import Path

import numpy as np
import pandas as pd

from shapeguard.reproducibility import save_json


def make_splits(manifest, seed=42, pilot=20):
    path = Path(manifest)
    frame = pd.read_csv(path).fillna("")
    if frame["split"].astype(str).str.len().any():
        raise ValueError("Splits already assigned; do not regenerate an experiment's split")
    parent = {str(x): str(x) for x in frame["image_id"]}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        if a in parent and b in parent:
            parent[root(b)] = root(a)

    for _, group in frame.groupby("sha256"):
        ids = list(group["image_id"].astype(str))
        for other in ids[1:]:
            join(ids[0], other)
    near_path = path.parent / "near_duplicates.csv"
    if near_path.exists():
        for row in pd.read_csv(near_path).itertuples():
            join(str(row.image_a), str(row.image_b))
    frame["group_id"] = [root(str(x)) for x in frame["image_id"]]
    rng = np.random.default_rng(seed)
    assignments = {}
    strata = {}
    for group_id, records in frame.groupby("group_id"):
        # Fracture presence stratification; anatomical balance is reported, not guaranteed.
        strata.setdefault(int(records["fractured"].max()), []).append(group_id)
    for groups in strata.values():
        rng.shuffle(groups)
        n = len(groups)
        if n < 3:
            raise ValueError("Need at least three duplicate groups in each fracture-presence stratum")
        nval = max(1, round(n * 0.15))
        ntest = max(1, round(n * 0.15))
        for i, group in enumerate(groups):
            assignments[group] = "val" if i < nval else "test" if i < nval + ntest else "train"
    frame["split"] = frame["group_id"].map(assignments)
    frame.to_csv(path, index=False)
    summary = frame.groupby("split").agg(images=("image_id", "count"), positive=("fractured", "sum"))
    summary.to_csv(path.parent / "split_summary.csv")
    save_json(path.parent / "split_policy.json", {
        "seed": seed, "ratios": [0.7, 0.15, 0.15],
        "grouping": "exact and conservative near-duplicate groups; NOT patient-level",
        "stratification": "fracture presence", "assignments": assignments,
    })
    # Fixed pilot from simpler single-view limb images, with both fracture classes where available.
    pool = frame[(frame["multiscan"] == 0) & (frame["hardware"] == 0)]
    limb = pool[(pool["leg"] == 1) | (pool["hand"] == 1)]
    if len(limb) >= pilot:
        pool = limb
    selections = []
    for positive, group in pool.groupby("fractured"):
        ids = group["image_id"].to_numpy().copy()
        rng.shuffle(ids)
        selections.extend(ids[: max(1, pilot // 2)].tolist())
    save_json(path.parent / "pilot.json", {"image_ids": selections[:pilot], "seed": seed})
    return summary
