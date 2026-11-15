import numpy as np
import pandas as pd
import pytest

from shapeguard.evaluation.instances import instance_metrics
from shapeguard.evaluation.metrics import binary_metrics
from shapeguard.evaluation.statistics import paired_bootstrap


def test_identical_and_disjoint():
    mask = np.zeros((8, 8), bool)
    mask[1:3, 1:3] = True
    same = binary_metrics(mask, mask)
    assert same["dice"] == same["iou"] == 1
    assert same["hausdorff"] == same["hd95"] == 0
    other = np.zeros_like(mask)
    other[5:7, 5:7] = True
    different = binary_metrics(other, mask)
    assert different["dice"] == different["iou"] == 0
    assert different["hausdorff"] > 0


def test_empty_policies():
    empty = np.zeros((5, 5), bool)
    same = binary_metrics(empty, empty)
    assert same["dice"] == 1 and same["hausdorff"] == 0
    assert np.isnan(same["precision"]) and np.isnan(same["recall"])
    target = empty.copy()
    target[2, 2] = True
    missed = binary_metrics(empty, target)
    assert missed["dice"] == missed["recall"] == 0
    assert np.isinf(missed["hausdorff"]) and missed["one_empty"]


def test_padding_excluded():
    target = np.zeros((5, 5), bool)
    prediction = target.copy()
    prediction[0] = True
    valid = np.ones_like(target)
    valid[0] = False
    assert binary_metrics(prediction, target, valid)["fp"] == 0


def test_merged_prediction_matches_once():
    a, b = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
    a[2:4, 1:3] = True
    b[2:4, 6:8] = True
    pred = a | b
    pred[2, 3:6] = True
    records, counts = instance_metrics(pred, [a, b], [0.05, 0.15])
    assert counts["matched_instances"] == 1 and counts["missed_instances"] == 1
    assert sum(r["matched"] for r in records) == 1


def test_bootstrap_pairing(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    frame = pd.DataFrame({"image_id": ["a", "b", "c"], "group_id": ["x", "x", "y"], "dice": [0.2, 0.3, 0.4], "target_pixels": [1, 2, 0]})
    frame.to_csv(first, index=False)
    frame["dice"] += 0.1
    frame.to_csv(second, index=False)
    result = paired_bootstrap(first, second, samples=100)
    assert result["difference_b_minus_a"] == pytest.approx(0.1)
    assert result["resampling_units"] == 2
    result = paired_bootstrap(first, second, samples=100, positive_only=True)
    assert result["images"] == 2
    frame = frame.iloc[:2]
    frame.to_csv(second, index=False)
    with pytest.raises(ValueError, match="same images"):
        paired_bootstrap(first, second)
