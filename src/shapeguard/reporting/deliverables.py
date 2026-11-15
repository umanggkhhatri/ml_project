"""Eight-page report and ten-slide presentation from completed, actual study artifacts."""

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapeguard.data.manifest import read_manifest
from shapeguard.reproducibility import save_json

LABELS = {
    "baseline": "U-Net baseline", "ici_2d_adaptation": "ICI 2D adaptation",
    "geometry_square_control": "Square local control", "geometry": "Geometry adaptive",
    "boundary_fixed": "Fixed boundary", "boundary_adaptive": "Adaptive boundary",
    "proposed": "Geometry + boundary", "contrast_control": "Two-view control",
    "full": "+ contrast consistency", "attention_unet": "Attention U-Net",
}
REFERENCES = [
    "Abedeen et al. FracAtlas. Scientific Data 10, 521 (2023). doi:10.1038/s41597-023-02432-4. Official dataset release v7.",
    "Rachmadi, Poon & Skibbe. Instance-wise and Center-of-Instance loss. PMLR 227:286-300 (2024; conference PDF MIDL 2023). proceedings.mlr.press/v227/rachmadi24a.html.",
    "Ronneberger et al. U-Net (2015). arXiv:1505.04597.",
    "Oktay et al. Attention U-Net (2018). arXiv:1804.03999.",
    "Zhou et al. U-Net++ (2018). arXiv:1807.10165.",
    "Chen et al. DeepLabV3+ (2018). arXiv:1802.02611.",
    "Isensee et al. nnU-Net. Nature Methods 18, 203-211 (2021). doi:10.1038/s41592-020-01008-z.",
    "Chen et al. TransUNet (2021). arXiv:2102.04306.",
    "Ma et al. MedSAM. Nature Communications 15, 654 (2024). doi:10.1038/s41467-024-44824-z.",
    "Kervadec et al. Boundary loss for highly unbalanced segmentation. PMLR 102:285-296 (2019). proceedings.mlr.press/v102/kervadec19a.html.",
]


def number(value, digits=3):
    return f"{float(value):.{digits}f}" if pd.notna(value) and np.isfinite(float(value)) else "N/A"


def analysis(study):
    study = Path(study)
    status = json.loads((study / "study_status.json").read_text())
    if not status.get("completed"):
        raise ValueError("Final deliverables require a completed study; unfinished results will not be invented")
    table = pd.read_csv(study / "comparison.csv")
    if set(table.task) != {"binary"}:
        raise ValueError("This report template is for binary segmentation only")
    main = table[table.seed == 42].copy()
    by_name = main.set_index("experiment")
    if not {"baseline", "ici_2d_adaptation", "proposed"}.issubset(set(by_name.index)):
        raise ValueError("Completed baseline, ICI and proposed results are required")
    comparisons = []
    for label, a, b, filename in [
        ("C1 vs ICI", "ici_2d_adaptation", "geometry", "geometry_vs_ici_seed_42_bootstrap.json"),
        ("C1 geometry control", "geometry_square_control", "geometry", "adaptive_geometry_vs_square_seed_42_bootstrap.json"),
        ("C2 scale control", "boundary_fixed", "boundary_adaptive", "adaptive_vs_fixed_seed_42_bootstrap.json"),
        ("Combined vs ICI", "ici_2d_adaptation", "proposed", "proposed_vs_ici_seed_42_bootstrap.json"),
        ("C3 compute control", "contrast_control", "full", "consistency_vs_control_seed_42_bootstrap.json"),
    ]:
        if a in by_name.index and b in by_name.index:
            delta = by_name.loc[b, "fracture_positive_dice"] - by_name.loc[a, "fracture_positive_dice"]
            path = study / filename
            interval = json.loads(path.read_text())["ci95"] if path.exists() else None
            comparisons.append({"comparison": label, "difference": float(delta), "ci95": interval,
                                "interpretation": "positive conditional evidence" if interval and interval[0] > 0 else "negative conditional evidence" if interval and interval[1] < 0 else "no clear positive difference"})
    best = main.loc[main.fracture_positive_dice.idxmax()]
    return table, main, comparisons, best


def architecture(path):
    fig, axis = plt.subplots(figsize=(10, 3.7))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 4)
    axis.axis("off")
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    nodes = [(0.2, "X-ray\n1 x 384 x 384"), (2.6, "U-Net\n16/32/64/128/256"), (5.0, "2 logits\nSoftmax"), (7.4, "Fracture mask\n0 / 1")]
    for x, label in nodes:
        axis.add_patch(FancyBboxPatch((x, 2.2), 2, 1.1, boxstyle="round,pad=0.08", facecolor="#e0f2fe", edgecolor="#0369a1"))
        axis.text(x + 1, 2.75, label, ha="center", va="center", fontsize=10)
    for x in (2.2, 4.6, 7.0):
        axis.add_patch(FancyArrowPatch((x, 2.75), (x + 0.4, 2.75), arrowstyle="->", mutation_scale=16))
    axis.text(5, 1.1, "Training only: global CE + Dice, ICI component matching,\ngeometry-local supervision, scale-adaptive boundary weighting,\noptional equal-compute two-view consistency", ha="center", fontsize=11)
    axis.text(5, 0.1, "Ground-truth geometry is never required at inference.", ha="center", color="#0369a1", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def study_plots(study, main, evaluations, output):
    output.mkdir(parents=True, exist_ok=True)
    architecture(output / "architecture.png")
    fig, axis = plt.subplots(figsize=(10, 5))
    labels = [LABELS.get(n, n) for n in main.experiment]
    axis.barh(labels, main.fracture_positive_dice, color="#0284c7")
    axis.invert_yaxis()
    axis.set_xlabel("Test fracture-positive image Dice (seed 42)")
    axis.set_xlim(0, max(0.1, main.fracture_positive_dice.max() * 1.15))
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for experiment in ("ici_2d_adaptation", "proposed", "contrast_control", "full"):
        entry = next((e for e in evaluations if e["experiment"] == experiment and e["seed"] == 42), None)
        if not entry:
            continue
        directory = Path(entry["evaluation"])
        instances = pd.read_csv(directory / "instances.csv")
        if len(instances):
            grouped = instances.groupby("size_group").matched_dice.mean().reindex(["small", "medium", "large"])
            axes[0].plot(grouped.index, grouped.values, marker="o", label=LABELS[experiment])
        path = directory / "robustness.csv"
        if path.exists():
            table = pd.read_csv(path)
            table = table[table.target_pixels > 0]
            curve = table[table.kind == "contrast"].groupby("severity").dice.mean()
            axes[1].plot(curve.index, curve.values, marker="o", label=LABELS[experiment])
    axes[0].set_ylabel("One-to-one matched instance Dice; misses = 0")
    axes[0].set_xlabel("Training-defined instance size group")
    axes[1].set_xlabel("Synthetic contrast-reduction severity")
    axes[1].set_ylabel("Fracture-positive image Dice")
    for axis in axes:
        if axis.lines:
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "size_robustness.png", dpi=180)
    plt.close(fig)
    return output


def build_report(path, main, comparisons, best, frame, figures_dir, evaluations, author, roll):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    document = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    margin = 42
    available = width - 2 * margin

    def page(title, index):
        document.setFillColor(colors.HexColor("#0369a1"))
        document.rect(0, height - 12, width, 12, fill=1, stroke=0)
        document.setFont("Helvetica-Bold", 17)
        document.setFillColor(colors.black)
        document.drawString(margin, height - 52, title)
        document.setFont("Helvetica", 8)
        document.setFillColor(colors.grey)
        document.drawString(margin, 22, "ShapeGuard-Frac | Binary X-ray fracture-region segmentation | Research prototype")
        document.drawRightString(width - margin, 22, f"{index}/8")
        return height - 80

    def paragraph(text, y, size=10, leading=14):
        document.setFont("Helvetica", size)
        document.setFillColor(colors.black)
        # Conservative wrapping for Helvetica and A4. Paragraph text is ASCII by design.
        for line in textwrap.wrap(text, max(35, int(available / (size * 0.51)))):
            document.drawString(margin, y, line)
            y -= leading
        return y - 8

    def heading(text, y):
        document.setFont("Helvetica-Bold", 11)
        document.setFillColor(colors.HexColor("#0369a1"))
        document.drawString(margin, y, text)
        return y - 20

    def image(image_path, y, max_height=260):
        from PIL import Image
        image_path = Path(image_path)
        if not image_path.exists():
            return paragraph("Figure unavailable; inspect run artifacts.", y)
        with Image.open(image_path) as im:
            iw, ih = im.size
        factor = min(available / iw, max_height / ih)
        dw, dh = iw * factor, ih * factor
        document.drawImage(str(image_path), margin + (available - dw) / 2, y - dh, dw, dh, preserveAspectRatio=True, mask="auto")
        return y - dh - 18

    def table(headers, rows, y, widths=None, size=8):
        widths = widths or [available / len(headers)] * len(headers)
        row_height = 22
        document.setFont("Helvetica-Bold", size)
        document.setFillColor(colors.HexColor("#e0f2fe"))
        document.rect(margin, y - row_height + 4, available, row_height, fill=1, stroke=0)
        x = margin + 3
        document.setFillColor(colors.black)
        for header, column_width in zip(headers, widths):
            document.drawString(x, y - 10, str(header))
            x += column_width
        y -= row_height
        document.setFont("Helvetica", size)
        for row in rows:
            x = margin + 3
            for value, column_width in zip(row, widths):
                document.drawString(x, y - 10, str(value))
                x += column_width
            document.setStrokeColor(colors.HexColor("#cbd5e1"))
            document.line(margin, y - row_height + 4, width - margin, y - row_height + 4)
            y -= row_height
        return y - 15

    y = page("ShapeGuard-Frac: problem and contributions", 1)
    y = paragraph(f"Author: {author or 'Student project'}" + (f" | Roll number: {roll}" if roll else " | Add roll number before submission"), y)
    y = heading("Abstract", y)
    y = paragraph(f"We investigate binary semantic segmentation of annotated fracture regions in musculoskeletal X-rays. The source audit retains {len(frame):,} FracAtlas v7 images, including {int(frame.fractured.sum()):,} positive images. A common U-Net backbone is compared with an explicit ICI-inspired 2D adaptation and controlled geometry, boundary and contrast modifications. Every reported value is generated from completed experiment artifacts. The highest observed seed-42 test fracture-positive Dice among the measured configurations is {number(best.fracture_positive_dice)}, obtained by {LABELS.get(best.experiment, best.experiment)}. This is not a claim of clinical readiness or state-of-the-art performance.", y)
    y = heading("Problem", y)
    y = paragraph("Fracture evidence can be low contrast, anatomically overlapped, small and irregular. Background pixels, including unfractured bone, dominate the image. Global overlap objectives can conceal poor small-instance performance. The professor-approved scope is fracture versus non-fracture pixels; bone masks and image-level classification are not the main task.", y)
    y = heading("Proposed contributions", y)
    for text in ["C1: aspect-ratio-aware local supervision, evaluated against both the ICI adaptation and an identical-objective square-support control.", "C2: boundary weighting with an instance-scale-dependent band, evaluated against a fixed-width control.", "C3: appearance consistency, evaluated against an equal two-view supervised-compute control. Existing techniques are cited; global novelty is not established by these experiments alone."]:
        y = paragraph(text, y)
    document.showPage()

    y = page("Related work and research gap", 2)
    for title, text in [
        ("CNN and self-configuring segmentation", "U-Net offers an interpretable encoder-decoder baseline. Attention U-Net gates skip features; it is also measured here. U-Net++, DeepLabV3+ and nnU-Net are relevant alternatives but are surveyed, not claimed as implemented comparisons."),
        ("Transformers and foundation models", "TransUNet and MedSAM illustrate global-context and promptable approaches. Their computational/data/prompt requirements motivate evaluating a compact convolutional model. We do not claim these models were experimentally outperformed."),
        ("Instance imbalance", "ICI compares overlapping predicted and ground-truth components and normalizes component centers to fixed shapes. Its original 3D stroke experiments motivate an X-ray extension. The PDF header refers to MIDL 2023; the PMLR publication page is 2024."),
        ("Boundary objectives", "Boundary-aware learning is established, including Kervadec's signed-distance formulation. Our normalized distance-band-weighted BCE is a different objective; the contribution studied is scale adaptation, not the invention of boundary learning."),
        ("Specific gap and controls", "We ask whether adapting local support to fracture aspect ratio and boundary width to annotation scale helps under fixed architecture/data conditions. Equal-objective and equal-compute controls isolate changes better than merely stacking modules."),
    ]:
        y = heading(title, y)
        y = paragraph(text, y)
    y = paragraph("Claim boundary: this is a student research extension. Related instance-wise, crop, boundary and consistency methods prevent an unqualified world-first claim.", y)
    document.showPage()

    y = page("Dataset, preprocessing and leakage controls", 3)
    splits = frame.groupby("split").agg(images=("image_id", "count"), positives=("fractured", "sum"))
    y = table(["Split", "Images", "Fracture positive"], [[name, row.images, row.positives] for name, row in splits.iterrows()], y)
    y = paragraph("FracAtlas v7 is pinned with a source checksum. Duplicate filenames with identical bytes are logged; corrupted JPEGs, nontrivial EXIF orientation and invalid polygons are quarantined rather than silently repaired. Original fracture polygon masks are preserved and combined into semantic foreground; original instance masks remain available for analysis.", y)
    y = paragraph("Seed 42 splits are disjoint at conservative exact/near-duplicate group level. Ratios are approximate because groups are indivisible. Reliable patient IDs are unavailable, so patient-independent evaluation is not claimed. Splitting precedes model training and augmentation.", y)
    y = paragraph("Preprocessing uses grayscale percentile normalization, aspect-ratio-preserving 384-pixel letterboxing, nearest-neighbor mask resizing and modest paired affine augmentation. Padding is excluded from losses and metrics. Instance disappearance after resizing is audited. Balanced training draws positive/negative images with replacement; validation/test retain natural prevalence.", y)
    eda_path = Path("data/prepared/eda/dataset_eda.png")
    y = image(eda_path, y, 270)
    document.showPage()

    y = page("Methodology and mathematical formulation", 4)
    y = image(figures_dir / "architecture.png", y, 190)
    y = paragraph("Global objective: 0.5 * weighted CE + foreground soft Dice. CE weights are [1,10], normalized by the valid target-weight sum. U-Net widths are 16,32,64,128,256 with group normalization. All loss ablations keep architecture and common training settings fixed.", y)
    y = paragraph("ICI adaptation: exact detached CPU 8-connected component matching selects intersecting predicted supports; retained probabilities have gradients. The no-TP variant excludes other true-instance intersections. Fixed-center maps use mean component confidence, not upstream constant assignment. Centroid/connectivity gradients are not claimed; missed instances rely on global supervision. This is not an exact reproduction.", y)
    y = paragraph("C1 replaces center confidence with mean instance-local Dice+BCE on expanded aspect-ratio-aware boxes. Its enclosing-square control uses the identical local objective and weight. Away from clipping, adaptive/square support area is approximately h/w for w>=h, reducing unnecessary background for elongated instances. Targets include other fracture polygons within the crop.", y)
    y = paragraph("C2: tau_k = clip(c*A_k/max(P_k,1), 1, 8); w_i = exp(-distance_i/tau_k). Before clipping, A/P scales linearly under uniform resizing, motivating scale-consistent band width. The boundary term is sum(w_i*BCE_i)/sum(w_i) in valid expanded support. P is a discrete inner-boundary proxy. Fixed control tau=3. Geometry/boundary weights are 0.25; retained instance weight is 0.5.", y)
    y = paragraph("C3 averages full supervised losses from two gamma views and adds 0.1 * symmetric Jensen-Shannon divergence. Gamma lies in [0.85,1.15]; no geometry changes occur between views. The compute control performs both supervised passes with consistency weight zero.", y, size=9, leading=12)
    document.showPage()

    y = page("Experimental setup and quantitative results", 5)
    y = paragraph("AdamW: learning rate 3e-4, weight decay 1e-4, batch size 4, maximum 16 epochs, patience 16, fixed seeds. The common budget was fixed after runtime preflight and before comparative test evaluation. Best checkpoints use validation fracture-positive image Dice. CUDA AMP and an approximately 3 GB allocator cap run sequentially on one shared A100 40 GB. No test-driven hyperparameter search is claimed.", y, size=9, leading=12)
    rows = [[LABELS.get(row.experiment, row.experiment), number(row.fracture_positive_dice), number(row.fracture_positive_iou), number(getattr(row, "fracture_pooled_precision", np.nan)), number(getattr(row, "fracture_pooled_recall", np.nan)), number(row.negative_image_fp_rate)] for row in main.itertuples()]
    y = table(["Method", "Dice+", "IoU+", "Prec.*", "Recall*", "Neg.FP"], rows, y, widths=[170, 60, 60, 70, 70, available - 430], size=7.5)
    y = paragraph("Dice+/IoU+: mean over positive test images. Prec./Recall*: pooled fracture-pixel counts over the full test cohort. Neg.FP: fraction of negative images with any predicted fracture pixel. Pixel accuracy and complete per-class tables are supplied in CSV; background dominance makes accuracy a weak headline metric.", y, size=9, leading=12)
    y = image(figures_dir / "comparison.png", y, 230)
    document.showPage()

    y = page("Ablation, uncertainty and computational cost", 6)
    for result in comparisons:
        interval = "[" + ", ".join(number(v) for v in result["ci95"]) + "]" if result["ci95"] else "unavailable"
        y = paragraph(f"{result['comparison']}: Dice difference {number(result['difference'])}; paired bootstrap 95% interval {interval}; {result['interpretation']}.", y, size=9, leading=12)
    y = paragraph("Bootstrap resamples duplicate groups among positive test images. It measures conditional test-sample variability, not training-seed variability or patient-level uncertainty. Repeated ICI/proposed seeds are reported in the full table; avoid inferring clinical significance from a small interval alone.", y, size=9, leading=12)
    rows = [[LABELS.get(row.experiment, row.experiment), f"{int(row.parameters):,}", number(row.conv_flops / 1e9, 2), number(row.latency_median_ms, 2)] for row in main.itertuples() if row.experiment in {"baseline", "ici_2d_adaptation", "proposed", "full", "attention_unet"}]
    y = table(["Method", "Parameters", "Conv GFLOPs", "Latency ms"], rows, y, widths=[190, 100, 105, available - 395], size=8)
    y = paragraph("FLOPs are convolution-only (2 FLOPs/MAC); normalization, interpolation and attention products are excluded. Latency is batch-one float32, warmed and synchronized. The GPU is shared, so observed timing differences among identical U-Nets do not establish architectural speed gains. Loss-only changes add no inference module but increase training work.", y, size=9, leading=12)
    y = paragraph("Boundary metrics: both-empty HD=0, one-empty HD=infinity. Report positive finite-case means together with missed/infinite counts; never silently discard failures. Units are pixels at 384-pixel letterboxed resolution, not millimetres.", y, size=9, leading=12)
    document.showPage()

    y = page("Small instances, robustness and qualitative failures", 7)
    y = image(figures_dir / "size_robustness.png", y, 220)
    y = paragraph("Size groups use training-only fractional-area tertiles. Hungarian one-to-one IoU matching (minimum 0.1) prevents a merged prediction counting as multiple detections; missed instances receive zero matched Dice. Synthetic contrast/noise/gamma/blur tests examine appearance sensitivity, not proven scanner or hospital transfer.", y, size=9, leading=12)
    proposed_entry = next(e for e in evaluations if e["experiment"] == "proposed" and e["seed"] == 42)
    y = image(Path(proposed_entry["evaluation"]) / "figures/best_worst_examples.png", y, 270)
    y = paragraph("Examples are deterministic best/worst positive test cases, not a random representative sample. Inspect missed thin regions, anatomical overlap, noisy predictions and mask-boundary ambiguity. FracAtlas annotates regions rather than guaranteed physical crack centerlines.", y, size=9, leading=12)
    document.showPage()

    y = page("Conclusion, limitations and references", 8)
    y = paragraph(f"Best observed seed-42 test Dice within this measured study: {LABELS.get(best.experiment, best.experiment)} ({number(best.fracture_positive_dice)}). Component effects are described by their controlled comparisons above; non-positive effects remain negative findings. The framework supplies reproducible segmentation experiments and keeps inference architecture compact.", y, size=9, leading=12)
    y = paragraph("Limitations: no verified patient-level split; heterogeneous regions/projections; approximate source masks; possible resizing loss; detached connectivity/centroid operations; no exhaustive tuning; shared-GPU timing; synthetic rather than external-domain robustness; no clinical validation or proof of global novelty. Future work: independent-site testing, original-reference reproduction, larger repeat-seed studies and better label-quality analysis.", y, size=9, leading=12)
    y = heading("References", y)
    for index, reference in enumerate(REFERENCES, 1):
        y = paragraph(f"[{index}] {reference}", y, size=8, leading=10)
    document.showPage()
    document.save()


def build_slides(path, main, comparisons, best, frame, figures_dir, evaluations, author, roll):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(13.333), Inches(7.5)

    def slide(title, bullets=None, picture=None):
        page = presentation.slides.add_slide(presentation.slide_layouts[6])
        box = page.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
        paragraph = box.text_frame.paragraphs[0]
        paragraph.text = title
        paragraph.font.size = Pt(30)
        paragraph.font.bold = True
        paragraph.font.color.rgb = RGBColor(3, 105, 161)
        if bullets:
            box = page.shapes.add_textbox(Inches(0.65), Inches(1.35), Inches(6 if picture else 12), Inches(5.5))
            box.text_frame.word_wrap = True
            for i, text in enumerate(bullets):
                p = box.text_frame.paragraphs[0] if i == 0 else box.text_frame.add_paragraph()
                p.text = text
                p.font.size = Pt(20)
                p.space_after = Pt(14)
        if picture and Path(picture).exists():
            from PIL import Image
            with Image.open(picture) as im:
                ratio = im.width / im.height
            max_width, max_height = (6.0 if bullets else 12.0), 5.7
            w, h = min(max_width, max_height * ratio), min(max_height, max_width / ratio)
            page.shapes.add_picture(str(picture), Inches(6.8 if bullets else 0.65), Inches(1.35 + (max_height - h) / 2), width=Inches(w), height=Inches(h))
        footer = page.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.2), Inches(0.25))
        footer.text_frame.paragraphs[0].text = f"ShapeGuard-Frac | Student research prototype | {len(presentation.slides)}/10"
        footer.text_frame.paragraphs[0].font.size = Pt(10)
        return page

    slide("ShapeGuard-Frac: binary X-ray segmentation", [
        "Target: annotated fracture region versus non-fracture pixels.",
        "Background includes bone; this is segmentation, not classification.",
        "Geometry-adaptive supervision, scale-adaptive boundaries, contrast consistency.",
        (author or "Student project") + (f" | {roll}" if roll else " | Add roll number before submission"),
    ])
    slide("Base paper and research gap", [
        "ICI: Rachmadi et al., PMLR 227 (2024; MIDL 2023 PDF).",
        "Original task: instance-imbalanced 3D stroke lesion segmentation.",
        "Our comparator is a documented 2D adaptation, not exact reproduction.",
        "Fracture geometry, scale and contrast require controlled investigation.",
        "U-Net / Attention U-Net measured; other architectures surveyed only.",
    ])
    slide("Dataset and preprocessing", [
        f"FracAtlas v7 audit: {len(frame):,} usable images, {int(frame.fractured.sum()):,} positives.",
        "Original polygons retained; malformed/orientation/corruption cases quarantined.",
        "Group-disjoint seed-42 splits; not verified patient-level splits.",
        "384-pixel letterboxing, paired augmentation, balanced training image draws.",
    ], "data/prepared/eda/dataset_eda.png")
    slide("Architecture and training-only modifications", picture=figures_dir / "architecture.png")
    slide("Contributions and fair controls", [
        "C1: aspect-ratio local support versus identical-objective square support.",
        "C2: adaptive boundary band versus fixed-width weighting.",
        "C3: JS consistency versus equal two-view supervised-compute control.",
        "Same U-Net, split, resolution, optimizer and common imbalance settings.",
        "No guaranteed gains or unqualified world-first claim.",
    ])
    slide("Baseline comparison: actual test results", picture=figures_dir / "comparison.png")
    slide("Ablation and paired uncertainty", [
        f"{r['comparison']}: Delta Dice {number(r['difference'])}; " + (f"95% CI [{number(r['ci95'][0])}, {number(r['ci95'][1])}]" if r["ci95"] else "CI unavailable") for r in comparisons
    ] + ["Bootstrap resamples duplicate groups, not patients; repeat-seed analysis is separate."])
    slide("Small instances and appearance robustness", picture=figures_dir / "size_robustness.png")
    proposed = next(e for e in evaluations if e["experiment"] == "proposed" and e["seed"] == 42)
    slide("Qualitative successes and failures", picture=Path(proposed["evaluation"]) / "figures/best_worst_examples.png")
    slide("Conclusion and limits", [
        f"Best observed seed-42 positive-image Dice: {LABELS.get(best.experiment, best.experiment)} = {number(best.fracture_positive_dice)}.",
        "Report negative component findings honestly; all results are artifact-derived.",
        "Loss modifications add no inference module; shared-GPU latency is variable.",
        "Limits: annotation ambiguity, no patient IDs, no external-site/clinical validation.",
        "References: FracAtlas; ICI; U-Net; Attention U-Net; boundary learning. Full citations in report.",
    ])
    presentation.save(path)


def deliverables(study="results/binary_study", manifest="data/prepared/manifest.csv", output="deliverables", author=None, roll=None, report_only=False):
    study = Path(study)
    table, main, comparisons, best = analysis(study)
    status = json.loads((study / "study_status.json").read_text())
    frame = read_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plots = study_plots(study, main, status["runs"], output / "figures")
    stem = f"{roll}_MLProject" if roll else "ShapeGuardFrac_MLProject"
    report = output / f"{stem}.pdf"
    slides = output / f"{stem}.pptx"
    build_report(report, main, comparisons, best, frame, plots, status["runs"], author, roll)
    if not report_only:
        build_slides(slides, main, comparisons, best, frame, plots, status["runs"], author, roll)
    table.to_csv(output / "complete_comparison.csv", index=False)
    save_json(output / "contribution_analysis.json", {"comparisons": comparisons, "best_observed_test_configuration": best.experiment,
                                                    "report_pages": 8, "slides": 0 if report_only else 10, "roll_number_missing": roll is None})
    return {"report": str(report), "slides": None if report_only else str(slides), "figures": str(plots), "roll_number_missing": roll is None}
