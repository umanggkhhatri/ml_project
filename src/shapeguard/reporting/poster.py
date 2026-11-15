"""Single-page PDF and editable poster using the supplied institutional template."""

import io
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapeguard.data.annotations import expanded_box
from shapeguard.data.dataset import FractureDataset
from shapeguard.reporting.deliverables import analysis, number
from shapeguard.reporting.figures import overlay
from shapeguard.reproducibility import save_json, sha256

# Coordinates are inches on the supplied 13 1/3 x 7 1/2 landscape poster.
LAYOUT = {
    "TextBox 9": (0.17, 1.20, 3.45, 0.30, 16, True),
    "TextBox 11": (0.17, 1.57, 3.45, 1.17, 11.3, False),
    "TextBox 12": (0.17, 2.89, 3.45, 0.30, 16, True),
    "TextBox 13": (0.17, 3.25, 3.45, 1.38, 11.3, False),
    "TextBox 14": (0.17, 4.85, 3.45, 0.30, 16, True),
    "TextBox 15": (0.17, 5.22, 3.45, 1.32, 11.3, False),
    "TextBox 43": (3.96, 1.20, 5.50, 0.30, 16, True),
    "TextBox 20": (3.96, 1.57, 5.50, 0.70, 10.7, False),
    "TextBox 19": (3.96, 2.42, 5.50, 0.30, 16, True),
    "TextBox 25": (3.96, 4.28, 5.50, 0.23, 9.1, False),
    "TextBox 31": (3.96, 4.68, 5.50, 1.85, 9.5, False),
    "TextBox 32": (9.81, 1.20, 3.30, 0.30, 16, True),
    "TextBox 37": (9.81, 1.57, 3.30, 0.23, 8.3, False),
    "TextBox 38": (9.81, 5.56, 3.30, 0.30, 16, True),
    "TextBox 39": (9.81, 5.94, 3.30, 0.61, 9.5, False),
    "TextBox 42": (0.17, 6.76, 12.95, 0.20, 10, True),
}
ARCHITECTURE_BOX = (3.96, 2.81, 5.50, 1.38)
TABLE_BOX = (9.81, 1.89, 3.30, 1.32)
RESULTS_BOX = (9.81, 3.28, 3.30, 2.16)
SHORT_LABELS = {
    "baseline": "U-Net", "ici_2d_adaptation": "ICI-adapt", "geometry_square_control": "Square control",
    "geometry": "Geometry", "boundary_fixed": "Fixed boundary", "boundary_adaptive": "Adaptive boundary",
    "proposed": "Combined", "contrast_control": "Two-view control", "full": "+ consistency",
    "attention_unet": "Attention U-Net",
}


def import_template(source, destination):
    """Keep layout/branding but remove unrelated example research from the reusable asset."""
    from PIL import Image, ImageDraw
    from pptx import Presentation

    source, destination = Path(source), Path(destination)
    presentation = Presentation(source)
    if len(presentation.slides) != 1:
        raise ValueError("Expected a one-page poster sample")
    slide = presentation.slides[0]
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text.strip():
            if shape.name == "Rectangle 3":
                shape.text = "Project title\nAuthor, roll number, email\nDepartment, institute"
            elif shape.name in LAYOUT and LAYOUT[shape.name][5]:
                continue  # Retain the sample's section headings.
            else:
                shape.text = "Project content placeholder"
        if shape.name in {"Picture 24", "Picture 34", "Picture 36"}:
            # Each image has unique bytes, avoiding accidental shared relationships.
            blank = Image.new("RGB", (700, 350), "white")
            ImageDraw.Draw(blank).text((20, 20), f"Project figure placeholder: {shape.name}", fill="gray")
            buffer = io.BytesIO()
            blank.save(buffer, format="PNG")
            buffer.seek(0)
            old = shape._element.blipFill.blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
            _, new = slide.part.get_or_add_image_part(buffer)
            shape._element.blipFill.blip.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", new)
            slide.part.drop_rel(old)
    presentation.core_properties.title = "User-provided institutional poster layout"
    presentation.core_properties.author = ""
    presentation.core_properties.last_modified_by = ""
    presentation.core_properties.subject = "Layout and branding retained; unrelated example research removed"
    destination.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(destination)
    save_json(destination.with_suffix(".provenance.json"), {"uploaded_original_sha256": sha256(source), "reusable_template_sha256": sha256(destination),
                                                         "note": "Original uploaded file unchanged. Reusable copy retains geometry and logos but removes sample sensor text and figures."})
    return destination


def poster_content(main, comparisons, best, frame):
    by_name = main.set_index("experiment")
    proposed, ici = by_name.loc["proposed"], by_name.loc["ici_2d_adaptation"]
    paired = next(r for r in comparisons if r["comparison"] == "Combined vs ICI")
    interval = paired["ci95"]
    qualification = "CI crosses zero" if interval and interval[0] <= 0 <= interval[1] else "positive paired CI" if interval and interval[0] > 0 else "negative/unavailable CI"
    findings = {r["comparison"]: r for r in comparisons}
    boundary = findings.get("C2 scale control")
    consistency = findings.get("C3 compute control")
    notes = []
    if boundary:
        notes.append("Adaptive bands improved their control." if boundary["difference"] > 0 else "Adaptive bands did not improve their control.")
    if consistency:
        notes.append("JS reduced clean Dice." if consistency["difference"] < 0 else "JS improved clean Dice.")
    conclusion = f"Combined vs ICI Dice+: {number(proposed.fracture_positive_dice)} vs {number(ici.fracture_positive_dice)}; {qualification}. " + " ".join(notes) + " False positives remain high. Not clinically ready."
    return {
        "TextBox 9": "Abstract",
        "TextBox 11": f"We study binary fracture-region segmentation using {len(frame):,} audited FracAtlas v7 X-rays. Geometry-adaptive supervision, scale-adaptive boundary weighting and contrast consistency extend an ICI-inspired 2D U-Net. Controlled ablations test each modification. Best observed seed-42 positive-image Dice: {number(best.fracture_positive_dice)} ({SHORT_LABELS.get(best.experiment, best.experiment)}). This research prototype is not clinically validated.",
        "TextBox 12": "Introduction",
        "TextBox 13": "X-rays contain low-contrast, overlapping anatomy and small irregular fracture regions [1]. Background pixels, including normal bone, dominate. Global overlap losses can miss small instances. ICI motivates instance-balanced supervision [2]; U-Net supplies compact spatial prediction [3]. We examine geometry, boundary scale and appearance sensitivity under fixed architecture and data conditions.",
        "TextBox 14": "Problem Statement",
        "TextBox 15": "Segment each valid pixel as fracture region (1) or non-fracture (0). Improve irregular/small-region delineation without adding inference modules. Use original fracture polygons, group-disjoint splits and controlled comparisons. No bone masks are required; this is segmentation, not image-level classification.",
        "TextBox 43": "Key Contributions",
        "TextBox 20": "C1: Aspect-ratio-aware local supervision; equal-loss square control.\nC2: Instance-scale boundary weighting; fixed-width control [4].\nC3: Contrast consistency; equal two-view compute control.\nCommon setting: 384 pixels, balanced image draws, 16 epochs.",
        "TextBox 19": "Methodology",
        "TextBox 25": "Figure 1. Single-pass inference; geometry is supplied only to training losses.",
        "TextBox 31": "Algorithm: binary fracture segmentation (seed 42)\n1. Audit FracAtlas v7; freeze duplicate-group splits.\n2. Normalize and letterbox image/masks to 384 pixels.\n3. Sample balanced images; use shared weighted CE + Dice.\n4. Train U-Net, ICI adaptation, controls and proposed variants.\n5. Add local geometry, adaptive bands and optional JS terms.\n6. Select checkpoints by validation positive-image Dice.\n7. Evaluate held-out masks, instances, robustness and cost.\nControls isolate support shape, band scale and added compute.\nDetached connectivity/centroids: not an exact ICI reproduction.\nNo test-driven checkpoint or hyperparameter selection.",
        "TextBox 32": "Results and Discussion",
        "TextBox 37": "Table 1. Seed-42 test Dice+/IoU+ (positive images).",
        "TextBox 38": "Conclusion",
        "TextBox 39": conclusion,
        "TextBox 42": "References",
        "Rectangle 41": "[1] Abedeen et al., FracAtlas, Scientific Data 10:521 (2023), doi:10.1038/s41597-023-02432-4; release v7.  [2] Rachmadi et al., ICI loss, PMLR 227:286-300 (2024; MIDL 2023 PDF).  [3] Ronneberger et al., U-Net, MICCAI (2015), arXiv:1505.04597.  [4] Kervadec et al., boundary loss, PMLR 102:285-296 (2019).",
    }


def poster_architecture(path):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, axis = plt.subplots(figsize=(5.5, 1.38))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3)
    axis.axis("off")
    for x, text in [(0.1, "X-ray\n384 pixels"), (2.65, "U-Net\n16 to 256"), (5.2, "Softmax\n2 classes"), (7.75, "Fracture\nmask")]:
        axis.add_patch(FancyBboxPatch((x, 1.20), 2.05, 1.45, boxstyle="round,pad=0.04", facecolor="#e0f2fe", edgecolor="#0369a1"))
        axis.text(x + 1.025, 1.925, text, ha="center", va="center", fontsize=8.7)
    for x in (2.15, 4.70, 7.25):
        axis.add_patch(FancyArrowPatch((x, 1.925), (x + 0.5, 1.925), arrowstyle="->", mutation_scale=12))
    axis.text(5, 0.58, "Training: CE + Dice, ICI, geometry, boundary (+ consistency)", ha="center", fontsize=8.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def poster_results(path, main, evaluations, manifest):
    entry = next(e for e in evaluations if e["experiment"] == "proposed" and e["seed"] == 42)
    directory = Path(entry["evaluation"])
    metrics = pd.read_csv(directory / "fracture_metrics.csv")
    positive = metrics[metrics.target_pixels > 0].sort_values("dice")
    if positive.empty:
        raise ValueError("Poster example requires a positive test image")
    row = positive.iloc[len(positive) // 2]
    dataset = FractureDataset(manifest, "test", task="binary", size=384)
    index = dataset.frame.index[dataset.frame.image_id == row.image_id][0]
    sample = dataset[int(index)]
    image, target = sample["image"][0].numpy(), sample["mask"].numpy()
    prediction = np.load(directory / "predictions" / f"{row.image_id}.npy")
    y0, y1, x0, x1 = expanded_box(target == 1, 0.5)
    # This is visualization-only cropping after full-image inference.
    y0, y1 = max(0, y0 - 12), min(384, y1 + 12)
    x0, x1 = max(0, x0 - 12), min(384, x1 + 12)
    fig = plt.figure(figsize=(6.6, 4.32))
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.8])
    axis = fig.add_subplot(grid[:2, :])
    selected = main[main.experiment.isin(["baseline", "ici_2d_adaptation", "geometry", "proposed", "full"])]
    axis.barh([SHORT_LABELS[r] for r in selected.experiment], selected.fracture_positive_dice, color="#0284c7")
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Positive-image test Dice", fontsize=11)
    axis.tick_params(labelsize=10)
    shown = [image, overlay(image, target, "binary"), overlay(image, prediction, "binary")]
    for i, (array, title) in enumerate(zip(shown, ["Input", "Annotation", "Prediction"])):
        axis = fig.add_subplot(grid[2, i])
        axis.imshow(array[y0:y1, x0:x1], cmap="gray")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.text(0.5, 0.02, f"Median-positive example: {row.image_id}, Dice {row.dice:.3f}\nVisualization crop only; not used for inference", fontsize=9, ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.10, 1, 1), h_pad=0.6)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return {"image_id": row.image_id, "dice": float(row.dice), "selection": "median positive-image Dice, deterministic", "crop": [y0, y1, x0, x1]}


def wrapped(text, font, size, width):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = word if not line else line + " " + word
            if line and stringWidth(candidate, font, size) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def fit_text(text, width, height, size, bold=False):
    font = "Times-Bold" if bold else "Times-Roman"
    while size >= 7.5:
        lines = wrapped(text, font, size, width * 72)
        if len(lines) * size * 1.13 <= height * 72:
            return lines, size
        size -= 0.25
    raise ValueError("Poster text would overflow its allocated box; shorten content instead")


def populate_template(template, path, content, main, architecture, results, author, roll, email, affiliation):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Inches, Pt

    presentation = Presentation(template)
    if len(presentation.slides) != 1 or abs(presentation.slide_width / 914400 - 13.3333333) > 0.01 or abs(presentation.slide_height / 914400 - 7.5) > 0.01:
        raise ValueError("Expected the supplied single-page 16:9 poster template")
    slide = presentation.slides[0]
    shapes = {shape.name: shape for shape in slide.shapes}
    if not (set(LAYOUT) | {"Rectangle 3", "Rectangle 41", "Picture 24", "Picture 34", "Picture 36"}).issubset(shapes):
        raise ValueError("Poster template shape schema differs; inspect before generating")
    for name, (x, y, w, h, size, bold) in LAYOUT.items():
        shape = shapes[name]
        shape.left, shape.top, shape.width, shape.height = map(Inches, (x, y, w, h))
        lines, font_size = fit_text(content[name], w, h, size, bold)
        frame = shape.text_frame
        frame.clear()
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.word_wrap = True
        frame.vertical_anchor = MSO_ANCHOR.TOP
        for i, line in enumerate(lines):
            paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            paragraph.text = line
            paragraph.font.name = "Times New Roman"
            paragraph.font.size = Pt(font_size)
            paragraph.font.bold = bold
            paragraph.font.color.rgb = RGBColor(0, 0, 0)
            paragraph.space_before = paragraph.space_after = Pt(0)
            paragraph.line_spacing = 1.13
    header = shapes["Rectangle 3"].text_frame
    header.clear()
    header.margin_left, header.margin_right = Inches(1.25), Inches(1.6)
    header.margin_top = header.margin_bottom = Inches(0.03)
    header.vertical_anchor = MSO_ANCHOR.MIDDLE
    header.word_wrap = True
    title = "ShapeGuard-Frac: Geometry-Adaptive Binary X-ray Fracture Segmentation"
    header_lines = [(title, 18, True), (f"{author} | {roll}" + (f" | {email}" if email else ""), 11, False), (affiliation, 10.5, False)]
    for i, (text, size, bold) in enumerate(header_lines):
        paragraph = header.paragraphs[0] if i == 0 else header.add_paragraph()
        paragraph.text = text
        paragraph.alignment = PP_ALIGN.CENTER
        paragraph.font.name = "Times New Roman"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = RGBColor(0, 0, 0)
        paragraph.space_before = paragraph.space_after = Pt(1)
    refs = shapes["Rectangle 41"].text_frame
    refs.clear()
    refs.margin_left, refs.margin_right = Inches(0.12), Inches(0.12)
    refs.margin_top, refs.margin_bottom = Inches(0.24), Inches(0.04)
    refs.word_wrap = True
    refs.vertical_anchor = MSO_ANCHOR.TOP
    lines, size = fit_text(content["Rectangle 41"], 12.95, 0.45, 8.6)
    for i, line in enumerate(lines):
        paragraph = refs.paragraphs[0] if i == 0 else refs.add_paragraph()
        paragraph.text = line
        paragraph.font.name = "Times New Roman"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = RGBColor(0, 0, 0)
        paragraph.space_after = Pt(0)

    def remove_picture(name):
        shape = shapes[name]
        relationship = shape._element.blipFill.blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
        shape._element.getparent().remove(shape._element)
        slide.part.drop_rel(relationship)

    for name in ("Picture 24", "Picture 34", "Picture 36"):
        remove_picture(name)
    for picture, box in ((architecture, ARCHITECTURE_BOX), (results, RESULTS_BOX)):
        x, y, w, h = box
        slide.shapes.add_picture(str(picture), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    x, y, w, h = TABLE_BOX
    table = slide.shapes.add_table(len(main) + 1, 3, Inches(x), Inches(y), Inches(w), Inches(h)).table
    for column, fraction in zip(table.columns, (0.58, 0.21, 0.21)):
        column.width = Inches(w * fraction)
    rows = [["Method", "Dice+", "IoU+"]] + [[SHORT_LABELS.get(r.experiment, r.experiment), number(r.fracture_positive_dice), number(r.fracture_positive_iou)] for r in main.itertuples()]
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = value
            cell.margin_left = cell.margin_right = Inches(0.03)
            cell.margin_top = cell.margin_bottom = Inches(0.005)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(224, 242, 254) if i == 0 else RGBColor(255, 255, 255)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = "Times New Roman"
                paragraph.font.size = Pt(7.0)
                paragraph.font.bold = i == 0
                paragraph.font.color.rgb = RGBColor(0, 0, 0)
                paragraph.space_after = Pt(0)
    presentation.core_properties.title = title
    presentation.core_properties.subject = "Binary X-ray fracture-region segmentation research poster"
    presentation.core_properties.author = author
    presentation.core_properties.last_modified_by = author
    presentation.save(path)


def theme_fill(template, shape):
    """Resolve the supplied template's solid theme fills and luminance transforms."""
    namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    node = shape.element.find("{http://schemas.openxmlformats.org/presentationml/2006/main}spPr/a:solidFill", namespace)
    if node is None:
        return (1.0, 1.0, 1.0)
    color = list(node)[0]
    if color.tag.endswith("srgbClr"):
        value = color.attrib["val"]
    else:
        name = {"bg1": "lt1", "tx1": "dk1", "bg2": "lt2", "tx2": "dk2"}.get(color.attrib["val"], color.attrib["val"])
        with zipfile.ZipFile(template) as archive:
            theme = ElementTree.fromstring(archive.read("ppt/theme/theme1.xml"))
        scheme = theme.find(f".//a:clrScheme/a:{name}", namespace)
        value = list(scheme)[0].attrib.get("lastClr", list(scheme)[0].attrib.get("val"))
    rgb = np.array([int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    for transform in color:
        factor = int(transform.attrib["val"]) / 100000
        if transform.tag.endswith("lumMod"):
            rgb *= factor
        elif transform.tag.endswith("lumOff"):
            rgb += factor
        elif transform.tag.endswith("tint"):
            rgb = rgb * factor + 1 - factor
        elif transform.tag.endswith("shade"):
            rgb *= factor
    return tuple(np.clip(rgb, 0, 1))


def render_pdf(template, path, content, main, architecture, results, author, roll, email, affiliation):
    from pptx import Presentation
    from reportlab.lib.colors import Color, HexColor
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    presentation = Presentation(template)
    shapes = {s.name: s for s in presentation.slides[0].shapes}
    width, height = presentation.slide_width / 12700, presentation.slide_height / 12700
    document = canvas.Canvas(str(path), pagesize=(width, height))
    document.setTitle("ShapeGuard-Frac research poster")
    document.setAuthor(author)
    for name in ("Rectangle 3", "Rectangle 8", "Rectangle 18", "Rectangle 26", "Rectangle 41"):
        shape = shapes[name]
        x, y, w, h = (shape.left / 12700, shape.top / 12700, shape.width / 12700, shape.height / 12700)
        document.setFillColor(Color(*theme_fill(template, shape)))
        document.setStrokeColor(HexColor("#475569"))
        document.setLineWidth(0.5)
        document.rect(x, height - y - h, w, h, fill=1, stroke=1)
    for name in ("Picture 5", "Picture 7"):
        shape = shapes[name]
        document.drawImage(ImageReader(io.BytesIO(shape.image.blob)), shape.left / 12700, height - (shape.top + shape.height) / 12700, shape.width / 12700, shape.height / 12700, mask="auto")
    title = "ShapeGuard-Frac: Geometry-Adaptive Binary X-ray Fracture Segmentation"
    document.setFillColor(HexColor("#000000"))
    center = width / 2
    for y, text, size, bold in [(0.30, title, 18, True), (0.61, f"{author} | {roll}" + (f" | {email}" if email else ""), 11, False), (0.84, affiliation, 10.5, False)]:
        document.setFont("Times-Bold" if bold else "Times-Roman", size)
        document.drawCentredString(center, height - y * 72, text)
    for name, (x, y, w, h, size, bold) in LAYOUT.items():
        lines, size = fit_text(content[name], w, h, size, bold)
        document.setFont("Times-Bold" if bold else "Times-Roman", size)
        for i, line in enumerate(lines):
            document.drawString(x * 72, height - y * 72 - size - i * size * 1.13, line)
    lines, size = fit_text(content["Rectangle 41"], 12.95, 0.45, 8.6)
    document.setFont("Times-Roman", size)
    for i, line in enumerate(lines):
        document.drawString(0.17 * 72, height - 7.00 * 72 - size - i * size * 1.13, line)
    for picture, (x, y, w, h) in ((architecture, ARCHITECTURE_BOX), (results, RESULTS_BOX)):
        document.drawImage(str(picture), x * 72, height - (y + h) * 72, w * 72, h * 72, mask="auto")
    x, y, w, h = TABLE_BOX
    rows = [["Method", "Dice+", "IoU+"]] + [[SHORT_LABELS.get(r.experiment, r.experiment), number(r.fracture_positive_dice), number(r.fracture_positive_iou)] for r in main.itertuples()]
    row_height = h * 72 / len(rows)
    column_widths = [w * 72 * fraction for fraction in (0.58, 0.21, 0.21)]
    for i, row in enumerate(rows):
        top = height - y * 72 - i * row_height
        document.setFillColor(HexColor("#e0f2fe") if i == 0 else HexColor("#ffffff"))
        document.setStrokeColor(HexColor("#94a3b8"))
        document.rect(x * 72, top - row_height, w * 72, row_height, fill=1, stroke=1)
        document.setFillColor(HexColor("#000000"))
        document.setFont("Times-Bold" if i == 0 else "Times-Roman", 7.0)
        left = x * 72 + 2
        for value, column_width in zip(row, column_widths):
            document.drawString(left, top - row_height + 2.0, value)
            left += column_width
    document.showPage()
    document.save()


def poster(study, manifest, output, template, author, roll, email="", affiliation=""):
    from shapeguard.data.manifest import read_manifest

    template = Path(template)
    _, main, comparisons, best = analysis(study)
    status = json.loads((Path(study) / "study_status.json").read_text())
    frame = read_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graphics = output / "poster_figures"
    graphics.mkdir(exist_ok=True)
    architecture, results = graphics / "architecture.png", graphics / "results.png"
    poster_architecture(architecture)
    example = poster_results(results, main, status["runs"], manifest)
    content = poster_content(main, comparisons, best, frame)
    stem = f"{roll}_MLProject_Poster"
    editable, pdf = output / f"{stem}.pptx", output / f"{stem}.pdf"
    populate_template(template, editable, content, main, architecture, results, author, roll, email, affiliation)
    render_pdf(template, pdf, content, main, architecture, results, author, roll, email, affiliation)
    save_json(output / "poster_provenance.json", {"template_sha256": sha256(template), "template_size_inches": [13.3333333333, 7.5],
                                              "pages": 1, "editable_slides": 1, "example": example,
                                              "author": author, "roll_number": roll, "email": email, "affiliation": affiliation,
                                              "figures_note": "Actual completed-study metrics; original template retained unchanged."})
    return {"poster_pdf": str(pdf), "poster_editable": str(editable), "poster_figures": str(graphics)}
