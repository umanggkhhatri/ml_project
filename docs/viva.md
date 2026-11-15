# Technical viva guide

Use actual final tables for numerical answers. This guide deliberately supplies no invented performance values.

## Problem and data

**What does binary mean here?** Every valid image pixel receives either non-fracture (0) or annotated fracture region (1). Non-fracture includes normal bone. This is segmentation, not one-label-per-image classification.

**Why are bone masks unnecessary?** The professor-approved binary task uses the original FracAtlas fracture polygons directly. The unused bone-annotation tools are only an optional extension.

**What exactly is a fracture mask?** A polygon-defined fracture region, not necessarily a precisely traced physical crack or complete fractured bone fragment.

**Why not quote the dataset paper's counts?** The official release changed. The notebook-free preparation pipeline audits the pinned release and reports actual usable images, positives and polygon instances. Invalid records are quarantined and source files are preserved.

**How was leakage reduced?** Exact and conservative perceptual near-duplicate groups stay within one split. Source splitting precedes augmentation. There are no reliable patient identifiers, so a patient-level split is not claimed.

**Why balanced sampling?** Fracture-positive images are a minority. Training draws use inverse class-frequency image weights, giving approximately equal positive/negative draw probability. Validation and test keep their original prevalence. This is a common setting across methods, not a contribution.

## Architecture and optimization

**Why U-Net?** An encoder supplies context; skip connections retain spatial detail; a decoder maps features to two logits at input resolution. A small model is interpretable and fits the approved shared GPU budget.

**Why two logits rather than one?** Two-channel softmax and categorical CE provide a consistent binary/multiclass code path. A one-logit sigmoid implementation is also valid but was not mixed into the comparisons.

**Why Dice plus CE?** CE supervises individual pixels; foreground Dice emphasizes overlap despite severe pixel imbalance. Weighted CE is normalized by valid target-weight sum. Padding is excluded. All compared methods share the same global objective.

**How are class weights different from contributions?** `[1,10]` weights are a common imbalance-handling baseline setting. Geometry, boundary scale adaptation and consistency are tested separately.

**How are checkpoints chosen?** By validation fracture-positive image Dice, never test performance. The common 16-epoch budget was fixed after a runtime preflight, before comparative test evaluation.

**What does AMP change?** CUDA autocast reduces activation precision for supported operations; losses are evaluated in float32 and GradScaler protects gradients. Ordinary FP16 overflows cause skipped optimizer updates and a reduced scale, not corrupted weights or a forced update. Skipped steps are logged; non-AMP nonfinite gradients remain fatal. Parameters/optimizer master state remain float32. Inference timing is separately measured in float32.

## Base paper and contributions

**What did ICI originally propose?** Instance-wise loss compares a ground-truth component with overlapping predicted components. Center-of-instance loss compares fixed-size shapes around component centers. Original experiments were 3D stroke MRI.

**Is this an exact reproduction?** No. It is explicitly labeled an ICI-inspired 2D adaptation. Exact detached CPU CCA is used, selected soft probabilities retain gradients, and center shapes carry component-mean confidence rather than constant hard assignments. Centroid/connectivity gradients are not claimed.

**Why does a missed ICI component need the global loss?** With no overlapping thresholded prediction, selected support is empty. The local selection term cannot supply a recovery gradient there. Global supervision still penalizes the missed foreground.

**What is C1?** A local Dice+BCE objective uses a context-expanded rectangle preserving each fracture instance's aspect ratio. It replaces the fixed-center confidence term. An enclosing-square control uses the identical local objective, so a difference can be attributed more specifically to support geometry.

**What is C2?** Boundary band width uses `tau = clip(c*A/max(P,1),1,8)`, where P is a discrete inner-boundary pixel-count proxy. Distance weights `exp(-d/tau)` multiply local BCE and are normalized by valid weight sum. Fixed `tau=3` is the control.

**Is C2 Kervadec's boundary loss?** No. Kervadec uses a signed-distance formulation. This project uses nonnegative distance-band-weighted BCE and investigates scale adaptation. Existing boundary learning must be cited.

**What is C3?** Two gamma-perturbed views receive supervised loss; symmetric JS divergence encourages compatible predictions. Its control performs the same two supervised passes with JS weight zero, controlling added exposure and compute.

**Are any annotations needed at inference?** No. Ground-truth geometry is used only during training. Each final U-Net predicts from the image in one pass.

**What if a contribution fails?** Report its negative result and limitations. Adding a term is an implemented modification, not automatically an improvement or proof of global novelty.

## Metrics and interpretation

**Dice versus IoU?** Dice is `2TP/(2TP+FP+FN)`; IoU is `TP/(TP+FP+FN)`. For one image, Dice = `2*IoU/(1+IoU)`. That nonlinear relationship does not necessarily hold after independently averaging over images.

**Why not headline pixel accuracy?** Correct background dominates. An all-background prediction can have high accuracy while missing every fracture.

**Why fracture-positive Dice separately?** Correctly empty negative masks would inflate an all-image average. Negative-image false positives are reported independently, and pooled precision/recall include the full cohort.

**Why HD95 and infinite cases?** Hausdorff detects extreme boundary deviation. HD95 reduces sensitivity to isolated outliers. If only one mask is empty, distance is undefined/infinite; those failures must be counted, not hidden by averaging only finite cases. Distances are network-resolution pixels, not millimetres.

**How are instances matched?** One-to-one Hungarian IoU matching with a preset minimum of 0.1. A merged prediction cannot count as multiple detections; missed instances receive zero matched Dice. Size thresholds come only from training data.

**What does bootstrap measure?** Conditional test-sample uncertainty, resampling duplicate groups among paired positive test images. It is not patient-level uncertainty, training-seed variability or clinical significance. Repeat-seed experiments address a different uncertainty source.

**Does corruption testing prove hospital generalization?** No. It tests controlled appearance sensitivity. Independent-site data are needed for that stronger claim.

**Are FLOPs complete?** The reported estimate covers convolutions at 2 FLOPs/MAC, excluding normalization, interpolation, activations and attention products. It must be labeled as convolution-only.

**Why can identical U-Nets have different measured latency?** Shared GPU load, scheduling, thermal/driver effects and measurement variability. Loss-only changes add no inference module; timing variation alone is not an architectural speed gain.

## Code demo

1. Show pinned source metadata, audit exclusions and group-disjoint split policy.
2. Show the baseline/ICI/proposed YAML differences and common training settings.
3. Run tests or show the saved verification output.
4. Show a real checkpoint's configuration, data hashes, training curve and evaluation table.
5. Run `shapeguard predict` on a held-out image; explain its input, probability map and native-resolution mask.
6. Show at least one failure and the paired ablation controls.

No notebook is required. Do not demo synthetic smoke outputs as if they were medical results.
