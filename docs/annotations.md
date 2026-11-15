# Bone annotation protocol

## Scope

Use a reasonably consistent limb scope first. Keep all original fracture masks. Bone masks are **student/AI-assisted annotations**, not clinically validated anatomy.

1. Outline all visible bones within the image; include interiors, not only cortical edges.
2. Exclude surrounding soft tissue, text and side markers.
3. Include visible separated bone fragments. Do not bridge an actual gap solely to make the mask connected.
4. Treat hardware as non-bone where visibly separable. Record difficult cases; prefer excluding complex hardware/multiview cases from the initial cohort.
5. For cropped bones, stop at the image boundary. For overlap, annotate the visible union rather than inventing per-bone depth order.
6. For uncertain outlines, use consistent rules and note uncertainty in your project limitations.
7. Preserve the original fracture polygons, even where they extend beyond the bone annotation. The exported fracture class takes precedence. This is fracture-region segmentation, not guaranteed crack-line tracing.

## Workflow

- Select the pilot using source attributes/splits before predictions are inspected.
- Start with brightness or prompted SAM proposals; inspect every mask.
- Green/red overlays are only a visual aid. Use brush or polygons to correct proposals at native resolution.
- `draft` masks are excluded from multiclass datasets.
- `accepted` means a student inspected/corrected the mask, not expert review.
- The local editor records mask hashes, status, timestamps, and fracture-outside-bone counts.
- Accepted masks are read-only. Training freezes the cohort and validates hashes.
- Preserve a separate dataset version for any later correction or expanded annotation effort.

Do not correct test labels to resemble the proposed model's output. Ideally finalize the full test cohort before training. Report annotation time based on a measured pilot, not an assumed rate.

## Class IDs

Native binary bone masks use 0/1. Proposal PNGs use 0/255 for visual inspection. Multiclass masks use 0=background, 1=bone outside fracture regions, 2=fracture region. Network padding uses ignore ID 255; native multiclass masks may not contain 255.

Bone absence in an unlabeled image is **unknown**, not background. The initial multiclass implementation trains only fully accepted images. Remaining source images can be used in a separately labeled binary fracture experiment; no partial-label multiclass objective is silently applied.

## Data versioning

Do not store source images or large proposal/checkpoint archives in Git. Keep prepared data and annotation histories backed up on approved storage. Cite FracAtlas and identify the derived annotation cohort precisely. Its source license is CC BY 4.0.
