# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["python-docx>=1.2,<2", "matplotlib>=3.9,<4", "Pillow>=10.4,<13"]
# ///
"""Build an editable, cited report draft from completed study artifacts.

Run: uv run scripts/build_word_report.py
Existing PDF and poster files are never overwritten. This is an assisted draft;
the student must review it against the institution's authorship requirements.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables"
STUDY = ROOT / "results/binary_study"
FIGURES = OUT / "word_report_figures"
FONT = "Times New Roman"
METHODS = {
    "baseline": "U-Net baseline",
    "ici_2d_adaptation": "ICI-inspired adaptation",
    "geometry_square_control": "Square-local control",
    "geometry": "Geometry-adaptive",
    "boundary_fixed": "Fixed boundary",
    "boundary_adaptive": "Adaptive boundary",
    "proposed": "Combined proposed",
    "contrast_control": "Two-view control",
    "full": "With consistency",
    "attention_unet": "Attention U-Net",
}


def records(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    return float(row[key])


def setup_plot_font():
    for path in Path("/System/Library/Fonts/Supplemental").glob("Times New Roman*.ttf"):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({"font.family": FONT, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})


def save_figure(fig, name):
    path = FIGURES / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(main, manifest):
    setup_plot_font()
    fig, ax = plt.subplots(figsize=(8, 2.9))
    counts = [sum(int(r["fractured"]) == label for r in manifest) for label in (0, 1)]
    bars = ax.bar(["Non-fracture images", "Fracture-positive images"], counts,
                  color=["#8694a2", "#284b63"], width=0.5)
    ax.bar_label(bars, padding=4)
    ax.set(ylabel="Number of usable images", ylim=(0, max(counts) * 1.16))
    fig.tight_layout()
    save_figure(fig, "eda.png")

    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    boxes = [(0.1, "X-ray\n1 x 384 x 384"), (2.7, "Encoder + decoder\nU-Net with skip links"),
             (5.3, "Two scores per pixel\nSoftmax probabilities"), (7.9, "Output mask\n0: other, 1: fracture")]
    for x, text in boxes:
        ax.add_patch(FancyBboxPatch((x, 2), 2, 1.35, boxstyle="round,pad=0.08",
                                   facecolor="#eef1f4", edgecolor="#284b63"))
        ax.text(x + 1, 2.67, text, ha="center", va="center", fontsize=11)
    for x in (2.15, 4.75, 7.35):
        ax.add_patch(FancyArrowPatch((x, 2.67), (x + 0.45, 2.67), arrowstyle="->", mutation_scale=14))
    ax.text(5, 1.15, "Encoder widths: 16, 32, 64, 128, 256; decoder restores image resolution",
            ha="center", fontsize=11)
    ax.text(5, 0.5, "Training only: global, instance, geometry and boundary feedback",
            ha="center", fontsize=11)
    ax.text(5, 0, "Prediction requires the X-ray only, not its annotation", ha="center", fontsize=11)
    save_figure(fig, "architecture.png")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    rows = list(reversed(main))
    values = [number(r, "fracture_positive_dice") for r in rows]
    bars = ax.barh([METHODS[r["experiment"]] for r in rows], values,
                   color=["#284b63" if r["experiment"] == "proposed" else "#8a9dad" for r in rows])
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    ax.set(xlabel="Mean Dice on fracture-positive test images (seed 42)", xlim=(0, 0.43))
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    for method, color in (("ici_2d_adaptation", "#8694a2"), ("proposed", "#284b63")):
        history = records(STUDY / "experiments" / method / "seed_42/history.csv")
        epochs = [int(r["epoch"]) + 1 for r in history]
        axes[0].plot(epochs, [number(r, "loss") for r in history], label=METHODS[method], color=color)
        axes[1].plot(epochs, [number(r, "fracture_positive_dice") for r in history],
                     label=METHODS[method], color=color)
    axes[0].set(xlabel="Epoch", ylabel="Training objective")
    axes[1].set(xlabel="Epoch", ylabel="Validation positive-image Dice")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "training.png")

    fig, ax = plt.subplots(figsize=(9, 2.7))
    for method, color in (("ici_2d_adaptation", "#8694a2"), ("proposed", "#284b63"),
                          ("contrast_control", "#a77d42"), ("full", "#74566d")):
        corruption = records(STUDY / "experiments" / method / "seed_42/robustness.csv")
        means = []
        for severity in (1, 2, 3):
            selected = [number(r, "dice") for r in corruption if r["kind"] == "contrast"
                        and int(r["severity"]) == severity and int(r["target_pixels"]) > 0]
            means.append(float(np.mean(selected)))
        ax.plot([1, 2, 3], means, marker="o", label=METHODS[method], color=color)
    ax.set(xlabel="Synthetic contrast-reduction severity", ylabel="Positive-image Dice", xticks=[1, 2, 3])
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "robustness.png")

    indexed = {r["image_id"]: r for r in manifest}
    metrics = {r["image_id"]: r for r in records(STUDY / "experiments/proposed/seed_42/fracture_metrics.csv")}
    resize = {r["image_id"]: r for r in records(STUDY / "experiments/proposed/seed_42/resize_audit.csv")}
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.7))
    ids = ("IMG0000142", "IMG0001199")
    for row_index, image_id in enumerate(ids):
        row = indexed[image_id]
        image = np.asarray(Image.open(ROOT / "data/prepared" / row["image_path"]).convert("L"))
        truth = np.asarray(Image.open(ROOT / "data/prepared" / row["fracture_path"])) > 0
        prediction = np.load(ROOT / "runs/binary_study/proposed/seed_42/evaluation_test/predictions" / f"{image_id}.npy")
        meta = resize[image_id]
        top, left, height, width = (int(meta[k]) for k in ("top", "left", "height", "width"))
        prediction = np.asarray(Image.fromarray(prediction[top:top + height, left:left + width]).resize(
            (image.shape[1], image.shape[0]), Image.Resampling.NEAREST)) > 0
        for col, (title, mask, color) in enumerate((("Input X-ray", None, None),
                                                  ("Dataset annotation", truth, "Greens"),
                                                  ("Model prediction", prediction, "Reds"))):
            ax = axes[row_index, col]
            ax.imshow(image, cmap="gray")
            if mask is not None:
                ax.imshow(np.ma.masked_where(~mask, np.ones_like(mask, dtype=float)),
                          cmap=color, vmin=0, vmax=1, alpha=0.55)
            score = f"; Dice {number(metrics[image_id], 'dice'):.3f}" if col == 2 else ""
            ax.set_title(f"{image_id}\n{title}{score}", fontsize=10)
            ax.axis("off")
    fig.tight_layout(pad=0.6)
    save_figure(fig, "qualitative.png")

    row = indexed["IMG0000144"]
    image = np.asarray(Image.open(ROOT / "data/prepared" / row["image_path"]).convert("L"))
    truth = np.asarray(Image.open(ROOT / "data/prepared" / row["fracture_path"])) > 0
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    axes[0].imshow(image, cmap="gray")
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(np.ma.masked_where(~truth, np.ones_like(truth, dtype=float)),
                   cmap="Greens", vmin=0, vmax=1, alpha=0.55)
    axes[2].imshow(Image.open(OUT / "figures/native_demo.png"))
    for ax, title in zip(axes, ("Input X-ray", "Dataset annotation (green)", "Native-size prediction (red)")):
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    save_figure(fig, "native_output.png")


def font(run, size=None, bold=None):
    run.font.name = FONT
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    properties = run._element.get_or_add_rPr()
    fonts = properties.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        properties.insert(0, fonts)
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{key}"), FONT)


def field(paragraph, instruction, cached=""):
    node = OxmlElement("w:fldSimple")
    node.set(qn("w:instr"), instruction)
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = cached
    run.append(text)
    node.append(run)
    paragraph._p.append(node)


def paragraph(doc, text, style=None, size=None):
    p = doc.add_paragraph(style=style)
    font(p.add_run(text), size)
    p.paragraph_format.widow_control = True
    return p


def heading(doc, text, level=1):
    return doc.add_heading(text, level)


def page(doc, title):
    doc.add_page_break()
    return heading(doc, title)


def caption(doc, text):
    p = paragraph(doc, text, "Caption")
    p.paragraph_format.keep_with_next = True
    return p


def picture(doc, name, height):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    with Image.open(FIGURES / name) as im:
        width = min(6.6, height * im.width / im.height)
    p.add_run().add_picture(str(FIGURES / name), width=Inches(width))


def table(doc, title, headers, rows, widths=None):
    caption(doc, title)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for cell, text in zip(t.rows[0].cells, headers):
        cell.text = text
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "E9EDF1")
        cell._tc.get_or_add_tcPr().append(shade)
    repeat = OxmlElement("w:tblHeader")
    t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for values in rows:
        for cell, value in zip(t.add_row().cells, values):
            cell.text = str(value)
    for i, row in enumerate(t.rows):
        no_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(no_split)
        for j, cell in enumerate(row.cells):
            if widths:
                cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1
                for run in p.runs:
                    font(run, 10, i == 0)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def equation(doc, text, label):
    p = paragraph(doc, f"{text}    ({label})", size=10.5)
    p.paragraph_format.left_indent = Inches(0.12)
    p.paragraph_format.keep_with_next = True


def document_styles(doc, author):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    sec.left_margin = sec.right_margin = Inches(0.8)
    sec.header_distance = sec.footer_distance = Inches(0.35)
    for style in doc.styles:
        if hasattr(style, "font"):
            style.font.name = FONT
            style.font.color.rgb = RGBColor(0, 0, 0)
            props = style.element.get_or_add_rPr()
            fonts = OxmlElement("w:rFonts")
            for key in ("ascii", "hAnsi", "eastAsia", "cs"):
                fonts.set(qn(f"w:{key}"), FONT)
            props.append(fonts)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.05
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name, size in (("Title", 22), ("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 12)):
        style = doc.styles[name]
        style.font.size = Pt(size)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(8)
        style.paragraph_format.space_after = Pt(7)
        style.paragraph_format.keep_with_next = True
    doc.styles["Caption"].font.size = Pt(10)
    doc.styles["Caption"].font.italic = True
    doc.styles["Caption"].paragraph_format.space_after = Pt(6)
    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    font(header.add_run("ShapeGuard-Frac | Machine Learning Project"), 9)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    font(footer.add_run(f"{author['roll_number']}  |  "), 9)
    field(footer, "PAGE")
    doc.core_properties.author = author["author"]
    doc.core_properties.title = "ShapeGuard-Frac: Instance-Aware Fracture Segmentation in X-ray Images"
    doc.core_properties.subject = "Editable project report draft based on completed experiments"
    doc.core_properties.keywords = "fracture segmentation, U-Net, ICI loss, FracAtlas"


def build(overwrite=False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    status = json.loads((STUDY / "study_status.json").read_text())
    if not status["completed"] or len(status["runs"]) != 14:
        raise ValueError("This report requires the completed 14-run study")
    all_results = records(STUDY / "comparison.csv")
    main = [r for r in all_results if int(r["seed"]) == 42]
    by_name = {r["experiment"]: r for r in main}
    manifest = records(ROOT / "data/prepared/manifest.csv")
    author = json.loads((ROOT / "submission.json").read_text())
    audit = json.loads((ROOT / "data/prepared/audit.json").read_text())
    analysis = json.loads((OUT / "contribution_analysis.json").read_text())
    make_figures(main, manifest)
    doc = Document()
    document_styles(doc, author)

    # Preliminary pages, following the updated report guidelines.
    p = paragraph(doc, "MACHINE LEARNING PROJECT REPORT", size=14)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(80)
    p = paragraph(doc, "ShapeGuard-Frac", "Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = paragraph(doc, "Instance-Aware Fracture Segmentation\nin X-ray Images", size=18)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(45)
    for text in (f"Submitted by: {author['author']}", f"Roll Number: {author['roll_number']}",
                 "Department of Computer Science and Engineering", "Indian Institute of Technology Patna",
                 "Semester V | Academic Year 2026", ""):
        p = paragraph(doc, text)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = paragraph(doc, "A study of local supervision, adaptive boundary weighting\nand appearance consistency using a compact U-Net.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    page(doc, "Abstract")
    paragraph(doc, "Fracture segmentation asks a model to mark the affected region in an X-ray, rather than simply decide whether the image contains a fracture. This is difficult because fracture annotations can be small and irregular, while most image pixels belong to other structures. This project investigates whether more targeted training feedback can help a compact U-Net handle these regions more effectively.")
    paragraph(doc, f"The study uses FracAtlas version 7. After data checks, {audit['usable_images']:,} images are retained, including {audit['positive_images']} fracture-positive images and {audit['instances']} annotated instances. Exact and conservative near-duplicate groups are kept within one data split. Images are normalized and resized with their proportions preserved. All main methods use the same global loss, image-sampling policy and training budget.")
    paragraph(doc, "The reference is an explicitly adapted two-dimensional version of the Instance-wise and Center-of-Instance (ICI) idea, originally studied for stroke lesions in MRI. Three modifications are examined: local supervision fitted to fracture geometry, boundary weighting with an adjustable band width, and prediction consistency under mild gamma changes. Matched controls are included so that the effect of each modification can be examined separately. Fourteen training runs are completed, including additional seeds for the reference and combined methods.")
    paragraph(doc, "The results are mixed. The square-local control obtains the highest seed-42 positive-image Dice, 0.376. Adaptive boundary weighting performs better than its fixed-width control, whereas shape-adaptive support does not establish an advantage over square support. Consistency reduces clean-image Dice relative to its two-view control. Across three seeds, the combined method averages 0.337 Dice compared with 0.280 for the ICI-inspired reference; however, the seed-42 paired interval includes zero. Low precision and frequent false alarms remain important weaknesses. The outcome is a reproducible research study, not a clinically validated diagnostic system.")
    paragraph(doc, "Keywords: X-ray segmentation; fracture regions; U-Net; instance-aware supervision; boundary weighting; FracAtlas.", size=11)

    page(doc, "Table of Contents")
    paragraph(doc, "The report follows five numbered chapters. Mathematical expressions are presented as readable text so they remain editable in Word.")
    for text in ("Preliminary pages: Title page; Abstract; Table of Contents; List of Figures and Tables",
                 "Chapter 1. Introduction", "    1.1 Background and motivation", "    1.2 Problem statement and objectives", "    1.3 Scope and contributions",
                 "Chapter 2. Literature Review", "    2.1 Existing approaches", "    2.2 Reference paper and research gap",
                 "Chapter 3. Proposed Methodology", "    3.1 Dataset description and preprocessing", "    3.2 Exploratory data analysis", "    3.3 Model architecture and algorithm", "    3.4 Training objectives and experimental setup",
                 "Chapter 4. Results and Discussion", "    4.1 Evaluation metrics and experimental results", "    4.2 Graphical analysis", "    4.3 Ablations, repeated seeds and efficiency", "    4.4 Generated outputs and discussion",
                 "Chapter 5. Conclusion and Future Scope", "    5.1 Key findings", "    5.2 Limitations and future research", "References"):
        p = paragraph(doc, text, size=11)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    page(doc, "List of Figures and Tables")
    heading(doc, "Figures", 2)
    for text in ("Figure 3.1. Distribution of fracture-positive and non-fracture images.",
                 "Figure 3.2. Prediction pipeline and the role of training-only supervision.",
                 "Figure 4.1. Positive-image Dice for the ten seed-42 methods.",
                 "Figure 4.2. Training objectives and validation overlap across epochs.",
                 "Figure 4.3. Sensitivity to synthetic contrast reduction.",
                 "Figure 4.4. A failed and a successful held-out prediction.",
                 "Figure 4.5. Native-resolution output compared with the dataset annotation."):
        paragraph(doc, text, size=11)
    heading(doc, "Tables", 2)
    for text in ("Table 3.1. Prepared dataset splits.", "Table 3.2. Common experimental settings.",
                 "Table 4.1. Main seed-42 test results.", "Table 4.2. Paired controlled comparisons.",
                 "Table 4.3. Repeated-seed performance.", "Table 4.4. Model size, boundary errors and timing."):
        paragraph(doc, text, size=11)
    heading(doc, "Algorithm and equations", 2)
    paragraph(doc, "Algorithm 3.1 describes the training and evaluation procedure. Equations (3.1)–(3.7) define the objectives; Equations (4.1)–(4.3) define the main evaluation scores.", size=11)

    page(doc, "Chapter 1. Introduction")
    heading(doc, "1.1 Background and motivation", 2)
    paragraph(doc, "An X-ray may show a fracture in a small part of a much larger anatomical image. A segmentation model must identify that region without confusing normal bone, joints or image artefacts with the annotation. Compared with whole-image classification, this task requires a more precise answer: every pixel must be assigned to fracture or non-fracture.")
    paragraph(doc, "The practical difficulty is that most pixels are not fracture pixels. In addition, a global overlap score can give more influence to large regions than small ones. These observations motivate a study of training supervision rather than a larger network. A compact model also makes it easier to compare loss changes under a limited shared-GPU budget.")
    heading(doc, "1.2 Problem statement and objectives", 2)
    paragraph(doc, "The problem is to segment the polygon-defined fracture regions in musculoskeletal X-rays and examine whether instance-aware local and boundary feedback improves a common U-Net baseline. The objectives are to prepare trustworthy data, establish comparable reference methods, test each modification with a suitable control, and report both improvements and failures on held-out images.")
    heading(doc, "1.3 Scope and contributions", 2)
    paragraph(doc, "The completed study is binary: label 1 means an annotated fracture region; label 0 means everything else, including normal bone. The original polygons are used directly. Bone annotation tools are available as an optional extension, but multiclass results are not claimed here.")
    for text in ("1. Geometry-aware local supervision replaces the fixed-centre feedback with local Dice and binary cross-entropy around each annotated instance. An identical-objective square control tests whether support shape matters.",
                 "2. Scale-adaptive boundary weighting adjusts its attention band using instance area and boundary size. A fixed-width control tests the scale rule.",
                 "3. Appearance consistency compares predictions from two gamma views. Both views are supervised, and a two-view control isolates the consistency penalty.",
                 "4. A reproducible comparison combines data hashes, saved configurations, 14 completed runs, repeated seeds, and paired group-level uncertainty estimates."):
        paragraph(doc, text, size=11)
    paragraph(doc, "These are implemented and evaluated contributions, not a claim that all modifications improve accuracy or establish global novelty.", size=11)

    page(doc, "Chapter 2. Literature Review")
    heading(doc, "2.1 Existing approaches", 2)
    paragraph(doc, "U-Net uses an encoder to collect context and a decoder to recover spatial detail through skip connections [3]. Its direct image-to-mask structure makes it a useful baseline for this study. Attention U-Net adds gates to skip features [4]; it is the additional architectural comparator actually trained here.")
    paragraph(doc, "Other established choices include U-Net++, DeepLabV3+ and the self-configuring nnU-Net framework [5–7]. Transformer-based models such as TransUNet and promptable medical foundation models such as MedSAM offer alternative ways to use context and prior knowledge [8, 9]. They are relevant literature, but they are not implemented comparisons in this experiment. No claim is made that the proposed method outperforms them.")
    heading(doc, "2.2 Reference paper: ICI loss", 2)
    paragraph(doc, "Rachmadi, Poon and Skibbe address instance imbalance: a segmentation objective may account well for a large object while overlooking smaller ones [2]. Their Instance-wise and Center-of-Instance (ICI) loss evaluates individual objects and their centre representations. The original experiments concern stroke lesions in three-dimensional MRI, using ATLAS R2.0, not fracture X-rays.")
    paragraph(doc, "This project adapts that idea to two-dimensional U-Net segmentation. Predicted components are selected using a detached threshold and exact connected-component analysis. Selected probabilities remain differentiable, but component identities and centroid locations do not. Predicted centre squares contain component-mean confidence rather than the original hard assignment. These changes make the reference ICI-inspired; it is not an exact reproduction of the published method.")
    heading(doc, "2.3 Research gap and study design", 2)
    paragraph(doc, "The question is whether a fixed-centre representation is an appropriate training signal for elongated fracture regions, and whether one boundary-band width suits annotations of different sizes. Boundary-aware learning is established; for example, Kervadec et al. use a signed-distance formulation [10]. The present boundary objective is different: it weights binary cross-entropy by nonnegative distance from the annotation boundary.")
    paragraph(doc, "The gap is investigated with matched controls. Rectangle and square support use the same local objective. Adaptive and fixed boundaries share their other settings. Consistency is compared with equal two-view supervision. This avoids attributing a gain to geometry or regularization when it may instead come from additional local feedback or computation.")

    page(doc, "Chapter 3. Proposed Methodology")
    heading(doc, "3.1 Dataset description and preprocessing", 2)
    paragraph(doc, f"FracAtlas is a public musculoskeletal X-ray dataset with fracture annotations [1]. The pinned version-7 preparation retains {audit['usable_images']:,} images, {audit['positive_images']} positives and {audit['instances']} annotated instances. {len(audit['quarantined'])} inconsistent or unsuitable records are quarantined. Polygon masks are preserved separately and combined into one semantic fracture mask.")
    splits = records(ROOT / "data/prepared/split_summary.csv")
    order = {"train": 0, "val": 1, "test": 2}
    table(doc, "Table 3.1. Prepared dataset splits.", ["Split", "Images", "Positive", "Negative"],
          [[{"train": "Training", "val": "Validation", "test": "Test"}[r["split"]], r["images"], r["positive"],
            int(r["images"]) - int(r["positive"])] for r in sorted(splits, key=lambda r: order[r["split"]])],
          [2.2, 1.4, 1.4, 1.4])
    paragraph(doc, "Exact and conservative perceptual near-duplicate groups remain within a split. The nominal group proportions are 70/15/15, with image counts differing because groups cannot be divided. Patient IDs are unavailable, so this is not a verified patient-level split.", size=11)
    paragraph(doc, "Images are converted to grayscale, normalized using the 1st and 99th intensity percentiles, and letterboxed to 384 × 384 without stretching. Masks use nearest-neighbour resizing. Padding is excluded from learning and evaluation. Training applies paired rotations within ±10 degrees and scaling from 0.95 to 1.05; validation and test do not receive random augmentation.", size=11)
    heading(doc, "3.2 Exploratory data analysis", 2)
    picture(doc, "eda.png", 2.4)
    caption(doc, "Figure 3.1. Fracture-positive images are a minority: 715 of 3,997 usable images. Counts are taken from the prepared manifest.")
    paragraph(doc, "Figure 3.1 explains why balanced image sampling is used during training. The separate imbalance in fracture pixels is addressed by weighted cross-entropy and foreground Dice. Validation and test preserve the observed prevalence. Native dimensions, anatomical attributes and the disappearance of small instances during resizing are also recorded in the data and evaluation audits.", size=11)

    page(doc, "3.3 Model architecture and algorithm")
    paragraph(doc, "The model receives a single-channel image and produces two scores at each pixel. Encoder widths are 16, 32, 64, 128 and 256. Each block uses two 3 × 3 convolutions, group normalization and ReLU. Max pooling reduces resolution; bilinear upsampling and skip-feature concatenation restore it. Softmax converts the final scores into class probabilities. The larger probability determines the output label.")
    picture(doc, "architecture.png", 2.4)
    caption(doc, "Figure 3.2. U-Net prediction pipeline. Geometry and boundary annotations calculate training feedback; they are not inputs during prediction.")
    caption(doc, "Algorithm 3.1. Training and evaluation procedure.")
    for text in ("Input: audited X-rays, fracture masks, instance masks and a fixed configuration.",
                 "1. Freeze duplicate-group-disjoint training, validation and test splits.",
                 "2. Normalize and letterbox each image; transform its masks identically.",
                 "3. Draw training images with inverse-frequency positive/negative sampling weights.",
                 "4. Predict two-class logits with U-Net and calculate the configured supervised loss.",
                 "5. For two-view experiments, supervise the alternate gamma view and optionally add JS divergence.",
                 "6. Backpropagate, clip gradients and update parameters with AdamW.",
                 "7. Evaluate validation positive-image Dice after each epoch; save the best checkpoint.",
                 "8. Evaluate the chosen checkpoint on the unchanged test cohort and export metrics.",
                 "Output: saved model, training history, per-image predictions and comparison tables."):
        p = paragraph(doc, text, size=10.5)
        p.paragraph_format.space_after = Pt(3)
    paragraph(doc, "At inference, the image alone is needed. Padding is removed from the probability maps, which are interpolated to native image size before the final class decision. This restores output dimensions, not detail lost during downsampling.", size=11)

    page(doc, "3.4 Training objectives and experimental setup")
    paragraph(doc, "In the following expressions, p is the fracture probability, y is the binary target, and valid pixels exclude padding. CE checks pixel classification; soft Dice checks foreground overlap. Epsilon is 0.000001.", size=11)
    equation(doc, "L_global = 0.5 × weighted CE + foreground soft Dice loss", "3.1")
    equation(doc, "L_Dice = 1 − [2 × sum(p × y) + epsilon] / [sum(p) + sum(y) + epsilon]", "3.2")
    paragraph(doc, "CE class weights are [1, 10], normalized by the sum of valid target-class weights. The ICI adaptation adds 0.5 times instance loss and 0.25 times centre loss. An entirely missed selected component still needs global supervision for recovery.", size=11)
    equation(doc, "L_geometry = mean over instances [local Dice loss + local BCE]", "3.3")
    paragraph(doc, "Equation (3.3) uses rectangles expanded by 25% context per side. The local target is the union of all true fractures, avoiding false background labels for another annotation inside the region. The square control changes support geometry only. Geometry is recalculated after transforms.", size=11)
    equation(doc, "tau = clip(A / max(P, 1), 1, 8);  boundary weight = exp(−distance / tau)", "3.4")
    equation(doc, "L_boundary = mean over instances [sum(weight × BCE) / sum(weight)]", "3.5")
    paragraph(doc, "A is instance area; P is inner-boundary pixel count. A/P scales like a length before clipping, motivating the adjustable band. Fixed control tau is 3. This is not a signed-distance loss.", size=11)
    equation(doc, "L_proposed = L_global + 0.5 × L_instance + 0.25 × L_geometry + 0.25 × L_boundary", "3.6")
    equation(doc, "L_full = [L_proposed(x) + L_proposed(x^gamma)] / 2 + 0.1 × JS(p, q)", "3.7")
    paragraph(doc, "For Equation (3.7), gamma is sampled from 0.85 to 1.15, q is the alternate-view prediction, and JS is symmetric Jensen–Shannon divergence. The control uses the same two supervised passes without the JS penalty. Consistent predictions can still be wrong, so supervision remains essential.", size=11)
    table(doc, "Table 3.2. Common experimental settings.", ["Setting", "Value"],
          [["Optimizer; learning rate; weight decay", "AdamW; 0.0003; 0.0001"],
           ["Batch size; epoch budget; patience", "4; 16; 16"],
           ["Main seed; repeated seeds", "42; additional 7 and 123 for ICI/proposed"],
           ["Checkpoint selection", "Validation fracture-positive Dice"],
           ["Device and numerical precision", "Shared A100; AMP forward, float32 losses"],
           ["Allocator cap; gradient clipping", "7.5% of 40 GB; norm 5.0"]], [3.0, 3.4])

    page(doc, "Chapter 4. Results and Discussion")
    heading(doc, "4.1 Evaluation metrics and experimental results", 2)
    paragraph(doc, "TP, FP and FN count correct fracture pixels, false fracture pixels and missed fracture pixels. Scores are calculated only on valid image pixels.", size=11)
    equation(doc, "Dice = 2 × TP / (2 × TP + FP + FN);  IoU = TP / (TP + FP + FN)", "4.1")
    equation(doc, "Precision = TP / (TP + FP);  Recall = TP / (TP + FN)", "4.2")
    equation(doc, "Pixel accuracy = (TP + TN) / (TP + TN + FP + FN)", "4.3")
    paragraph(doc, "Table 4.1 reports Dice and IoU averaged over the 90 positive test images. Precision and recall pool fracture-pixel counts across all 505 test images. Negative FP is the fraction of negative images with any predicted fracture pixel; it is not clinical diagnostic specificity. Background-heavy pixel accuracy is not the headline measure.", size=11)
    table(doc, "Table 4.1. Main seed-42 test results (higher is better except negative FP).",
          ["Method", "Dice+", "IoU+", "Precision", "Recall", "Neg. FP"],
          [[METHODS[r["experiment"]], *[f"{number(r, k):.3f}" for k in
            ("fracture_positive_dice", "fracture_positive_iou", "fracture_pooled_precision", "fracture_pooled_recall", "negative_image_fp_rate")]] for r in main],
          [2.15, 0.7, 0.7, 0.95, 0.8, 0.95])
    paragraph(doc, "The square-local control has the highest seed-42 Dice (0.376). The combined method improves the observed ICI score from 0.302 to 0.327, but does not outperform every control. Its precision is only 0.057 and its negative-image FP rate is 0.993. These values show why a small overlap gain cannot be treated as evidence of safe medical use.", size=11)
    paragraph(doc, "Fourteen runs are completed: ten seed-42 configurations plus two additional seeds each for the ICI adaptation and combined method. Baseline, ICI, both local supports, fixed/adaptive boundary, combined, two-view control, consistency and Attention U-Net are the measured comparisons. Other literature models are not experimentally ranked.", size=11)

    page(doc, "4.2 Graphical analysis")
    picture(doc, "comparison.png", 2.7)
    caption(doc, "Figure 4.1. Test overlap for the ten seed-42 methods. The square-local control, not the combined model, has the highest observed Dice.")
    picture(doc, "training.png", 2.25)
    caption(doc, "Figure 4.2. Actual training objectives and validation Dice. Loss magnitudes are not directly comparable because the objectives contain different terms; validation curves support checkpoint selection.")
    picture(doc, "robustness.png", 2.1)
    caption(doc, "Figure 4.3. Actual positive-image Dice under increasing synthetic contrast reduction. This measures appearance sensitivity, not independent-hospital generalization.")
    paragraph(doc, "Figures 4.1–4.3 are regenerated from saved numerical artifacts. The clean-image comparison, training behaviour and corruption response answer different questions; none should replace the others.", size=10.5)

    page(doc, "4.3 Ablations, repeated seeds and efficiency")
    paragraph(doc, "The comparisons in Table 4.2 use paired positive test images and resample 83 duplicate groups containing 90 images. Intervals are conditional on the checkpoints and test cohort; they do not account for unavailable patient IDs or replace repeated training seeds.", size=11)
    names = ["Geometry minus ICI", "Geometry minus square", "Adaptive minus fixed boundary", "Combined minus ICI", "Consistency minus two-view control"]
    table(doc, "Table 4.2. Paired Dice differences and 95% bootstrap intervals.",
          ["Comparison", "Difference", "95% interval"],
          [[name, f"{r['difference']:+.5f}", f"[{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]"]
           for name, r in zip(names, analysis["comparisons"])], [3.25, 1.0, 2.15])
    paragraph(doc, "Local geometry supervision improves on the adapted reference, but not on the matched square support. Adaptive boundaries outperform the fixed-width control. Consistency reduces clean-image Dice. The combined seed-42 interval includes zero, so its observed improvement is uncertain under this analysis.", size=11)
    repeats = records(OUT / "repeat_seed_summary.csv")
    table(doc, "Table 4.3. Repeated-seed positive-image Dice (seeds 42, 7 and 123).",
          ["Method", "Seeds", "Mean Dice", "Sample SD"],
          [[METHODS[r["experiment"]], r["seeds"], f"{number(r, 'dice_mean'):.5f}", f"{number(r, 'dice_std'):.5f}"] for r in repeats], [2.8, 0.8, 1.4, 1.4])
    paragraph(doc, "The combined method has a positive paired difference in all three seeds. This is encouraging, but three repetitions remain a small sample and do not remove the seed-42 interval qualification.", size=11)
    table(doc, "Table 4.4. Model cost and positive-image boundary errors (seed 42).",
          ["Method", "Parameters", "GFLOPs", "ms", "HD / HD95", "Inf."],
          [[METHODS[name], f"{int(float(by_name[name]['parameters'])):,}",
            f"{number(by_name[name], 'conv_flops') / 1e9:.2f}", f"{number(by_name[name], 'latency_median_ms'):.2f}",
            f"{number(by_name[name], 'fracture_hausdorff_positive_finite_mean'):.1f} / {number(by_name[name], 'fracture_hd95_positive_finite_mean'):.1f}",
            int(float(by_name[name]["fracture_hausdorff_positive_infinite_cases"]))]
           for name in ("baseline", "ici_2d_adaptation", "proposed", "full")], [2.0, 1.05, 0.7, 0.55, 1.35, 0.55])
    paragraph(doc, "HD/HD95 are positive finite-case means in 384-resolution pixels, not millimetres. One-empty masks have infinite distance and are counted separately; both-empty masks have zero distance. FLOPs cover convolutions only. Timing is warmed, synchronized, float32, batch-one inference on a shared GPU. Identical U-Nets differ in training losses, not inference modules, so timing variation is not an architectural speed gain.", size=10.5)

    page(doc, "4.4 Generated outputs and discussion")
    picture(doc, "qualitative.png", 3.5)
    caption(doc, "Figure 4.4. Held-out examples from the combined seed-42 checkpoint. Green is the annotation and red is the prediction. The top case misses the true region (Dice 0.000); the bottom case overlaps well (Dice 0.888). Cases were selected to show failure and success, not average performance. Scores come from network-resolution evaluation.")
    picture(doc, "native_output.png", 2.5)
    caption(doc, "Figure 4.5. Image IMG0000144: input, dataset annotation and saved native-size output. Full-image inference uses the best-validation combined seed-42 checkpoint on MPS. Red areas are predictions, not medically confirmed fractures.")
    paragraph(doc, "Figure 4.4 shows why numerical averages need visual inspection: an apparently plausible prediction can miss the annotated site, while another example has good overlap. Figure 4.5 also shows extra predicted areas beyond the annotation. The image is an illustrative held-out output, not a representative claim of accuracy.", size=11)
    paragraph(doc, "Instances are matched one-to-one with a fixed IoU threshold of 0.1, preventing a merged blob from counting as multiple detections. Small/medium/large thresholds use training-only area fractions. Possible explanations for the square control's advantage include useful surrounding context; this remains a hypothesis, not a demonstrated cause.", size=11)

    page(doc, "Chapter 5. Conclusion and Future Scope")
    heading(doc, "5.1 Key findings", 2)
    paragraph(doc, "This project examined how a compact U-Net should be supervised when fracture annotations are small, irregular and imbalanced. The complete pipeline audits the data, preserves split groups, trains comparable methods and exports numerical and visual evidence. It does not rely on synthetic smoke-test results as medical performance.")
    paragraph(doc, "The clearest finding is that the modifications should be judged separately. Local supervision helped relative to the ICI-inspired reference, but shape-adaptive support did not establish an advantage over its square control. Adjustable boundary weighting improved on fixed weighting. Adding consistency reduced clean-image Dice under the chosen settings. The combined method had a higher three-seed mean than the reference, although its seed-42 paired interval crossed zero.")
    heading(doc, "5.2 Limitations", 2)
    paragraph(doc, "Overlap remains modest and false alarms are frequent. FracAtlas polygons describe annotated regions, which need not be exact physical crack boundaries. Resizing can remove small details. Duplicate grouping reduces known leakage but does not establish patient-independent evaluation. The 16-epoch budget and fixed loss coefficients limit conclusions about fully tuned performance. Detached component selection also limits the gradients available from the ICI term.")
    paragraph(doc, "Synthetic corruption tests are not a substitute for independent-site data. Shared-GPU timings are variable, and convolution-only FLOPs are incomplete cost estimates. The study neither proves global novelty nor establishes clinical readiness. Its completed binary scope should not be represented as multiclass bone segmentation.")
    heading(doc, "5.3 Future research directions", 2)
    paragraph(doc, "The next priority is reducing false positives without missing fracture regions. Useful follow-up work includes inspecting annotation quality, checking higher-resolution training, tuning on validation data only, and measuring more training seeds. A closer reproduction of the original reference would clarify which effects arise from the adaptation.")
    paragraph(doc, "Independent-patient and independent-site datasets are needed before making stronger generalization claims. If multiclass segmentation is pursued, bone masks must be explicitly validated and kept separate from the current binary results. Overall, the project supports careful experimentation: more targeted feedback can help, but more complicated feedback is not automatically better.")

    page(doc, "References")
    references = [
        "[1] Abedeen, I., et al. (2023). FracAtlas: A dataset for fracture classification, localization and segmentation of musculoskeletal radiographs. Scientific Data, 10, 521. https://doi.org/10.1038/s41597-023-02432-4. Dataset used: official Figshare release v7, article 22363012.",
        "[2] Rachmadi, F., Poon, C., and Skibbe, H. (2024). Improving Segmentation of Objects with Varying Sizes in Biomedical Images using Instance-wise and Center-of-Instance Segmentation Loss Function. Proceedings of Machine Learning Research, 227, 286–300. https://proceedings.mlr.press/v227/rachmadi24a.html. The conference PDF header refers to MIDL 2023.",
        "[3] Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. https://arxiv.org/abs/1505.04597.",
        "[4] Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. https://arxiv.org/abs/1804.03999.",
        "[5] Zhou, Z., et al. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. https://arxiv.org/abs/1807.10165.",
        "[6] Chen, L.-C., et al. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV 2018. https://arxiv.org/abs/1802.02611.",
        "[7] Isensee, F., et al. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18, 203–211. https://doi.org/10.1038/s41592-020-01008-z.",
        "[8] Chen, J., et al. (2021). TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. https://arxiv.org/abs/2102.04306.",
        "[9] Ma, J., et al. (2024). Segment anything in medical images. Nature Communications, 15, 654. https://doi.org/10.1038/s41467-024-44824-z.",
        "[10] Kervadec, H., et al. (2019). Boundary loss for highly unbalanced segmentation. Proceedings of Machine Learning Research, 102, 285–296. https://proceedings.mlr.press/v102/kervadec19a.html.",
    ]
    for text in references:
        p = paragraph(doc, text, size=10.5)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    heading(doc, "Project artifacts", 2)
    paragraph(doc, "The source code, configurations and exported results are available at https://github.com/umanggkhhatri/ml_project. Numerical sources are comparison.csv, per-run evaluation/history CSVs and the paired bootstrap files under results/binary_study/. The image-input-only native demonstration is documented in deliverables/native_demo.json. Dataset and checkpoint binaries remain outside Git.", size=10.5)

    path = OUT / f"{author['roll_number']}_MLProject.docx"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    doc.save(path)
    print(f"Created {path}")
    print("15 planned pages; confirm rendered pagination in Word before PDF submission.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace a previously generated Word draft")
    build(parser.parse_args().overwrite)
