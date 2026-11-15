import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from shapeguard.reporting.poster import fit_text
from shapeguard.reproducibility import sha256


def test_text_overflow_is_not_silently_hidden():
    pytest.importorskip("reportlab")
    lines, size = fit_text("Small-instance supervision preserves context.", 3, 1, 12)
    assert lines and size <= 12
    with pytest.raises(ValueError, match="overflow"):
        fit_text("Too much text. " * 1000, 1, 0.1, 12)


def test_supplied_template_produces_one_page_without_sample_sensor_content(tmp_path):
    pytest.importorskip("pptx")
    pytest.importorskip("reportlab")
    from pptx import Presentation

    from shapeguard.reporting.poster import populate_template, render_pdf

    template = Path("assets/poster_template.pptx")
    before = sha256(template)
    image = tmp_path / "SYNTHETIC_TEST_GRAPH.png"
    Image.fromarray(np.full((100, 400, 3), 240, np.uint8)).save(image)
    from shapeguard.reporting.poster import LAYOUT
    content = {name: "SYNTHETIC TEMPLATE TEST" for name in LAYOUT}
    content["Rectangle 41"] = "Synthetic template verification, not medical results."
    main = pd.DataFrame([{"experiment": "baseline", "fracture_positive_dice": 0.1, "fracture_positive_iou": 0.05}])
    editable, pdf = tmp_path / "SYNTHETIC_POSTER_TEST.pptx", tmp_path / "SYNTHETIC_POSTER_TEST.pdf"
    identity = ("SYNTHETIC TEMPLATE TEST", "TEST", "", "Synthetic fixture")
    populate_template(template, editable, content, main, image, image, *identity)
    render_pdf(template, pdf, content, main, image, image, *identity)
    poster = Presentation(editable)
    assert len(poster.slides) == 1
    assert abs(poster.slide_width / 914400 - 13.3333333) < 0.01
    texts = " ".join(s.text for s in poster.slides[0].shapes if s.has_text_frame)
    assert "photonic" not in texts.lower() and "gpr" not in texts.lower() and "10.67" not in texts
    assert len(re.findall(rb"/Type\s*/Page\b", pdf.read_bytes())) == 1
    assert sha256(template) == before


def test_project_content_fits_template_boxes_without_fabricating_deliverables():
    pytest.importorskip("reportlab")
    from shapeguard.reporting.poster import LAYOUT, poster_content
    # In-memory layout fixture only. No final PDF/PPTX or result artifact is generated.
    main = pd.DataFrame([{"experiment": name, "fracture_positive_dice": 0.1, "fracture_positive_iou": 0.05}
                         for name in ("baseline", "ici_2d_adaptation", "proposed")])
    comparisons = [{"comparison": "Combined vs ICI", "difference": 0.0, "ci95": [-0.1, 0.1]}]
    frame = pd.DataFrame({"fractured": [1, 0]})
    content = poster_content(main, comparisons, main.iloc[0], frame)
    for name, (_, _, width, height, size, bold) in LAYOUT.items():
        lines, chosen_size = fit_text(content[name], width, height, size, bold)
        assert len(lines) * chosen_size * 1.13 <= height * 72
