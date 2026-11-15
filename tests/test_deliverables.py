import json
import re

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from shapeguard.reporting.deliverables import analysis, build_report, build_slides
from shapeguard.smoke import synthetic_dataset


def test_unfinished_study_cannot_generate_final_results(tmp_path):
    (tmp_path / "study_status.json").write_text(json.dumps({"completed": False}))
    with pytest.raises(ValueError, match="completed study"):
        analysis(tmp_path)


def test_document_limits_with_explicit_synthetic_fixture(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("pptx")
    from pptx import Presentation

    from shapeguard.data.manifest import read_manifest

    # Template verification only; no synthetic result is written into final project artifacts.
    frame = read_manifest(synthetic_dataset(tmp_path / "synthetic_data"))
    figures = tmp_path / "synthetic_figures"
    figures.mkdir()
    placeholder = np.full((80, 160, 3), 240, np.uint8)
    for name in ("architecture", "comparison", "size_robustness"):
        Image.fromarray(placeholder).save(figures / f"{name}.png")
    evaluation = tmp_path / "synthetic_evaluation"
    (evaluation / "figures").mkdir(parents=True)
    Image.fromarray(placeholder).save(evaluation / "figures/best_worst_examples.png")
    main = pd.DataFrame([{
        "experiment": name, "fracture_positive_dice": 0.1, "fracture_positive_iou": 0.05,
        "fracture_pooled_precision": 0.1, "fracture_pooled_recall": 0.1,
        "negative_image_fp_rate": 0.5, "parameters": 100, "conv_flops": 1000, "latency_median_ms": 1,
    } for name in ("baseline", "ici_2d_adaptation", "proposed")])
    entries = [{"experiment": "proposed", "seed": 42, "evaluation": str(evaluation)}]
    report, slides = tmp_path / "SYNTHETIC_TEMPLATE_TEST.pdf", tmp_path / "SYNTHETIC_TEMPLATE_TEST.pptx"
    build_report(report, main, [], main.iloc[0], frame, figures, entries, "SYNTHETIC TEMPLATE TEST", None)
    build_slides(slides, main, [], main.iloc[0], frame, figures, entries, "SYNTHETIC TEMPLATE TEST", None)
    assert len(re.findall(rb"/Type\s*/Page\b", report.read_bytes())) == 8
    assert len(Presentation(slides).slides) == 10
