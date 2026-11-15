# Completed binary segmentation study

All **14 runs** completed on the approved shared A100 allocation. Each uses a fixed 16-epoch budget, original FracAtlas masks and validation-based checkpoint selection. Ten primary configurations use seed 42; ICI and the combined proposed method additionally use seeds 7 and 123.

## Primary test results (seed 42)

Dice+/IoU+ are positive-image means. Precision/recall are pooled fracture-pixel counts over the complete test cohort. Negative FP is the fraction of negative images with any predicted fracture pixel, not clinical diagnosis specificity.

| Configuration | Dice+ | IoU+ | Pooled precision | Pooled recall | Negative FP |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.300 | 0.207 | 0.066 | 0.512 | 0.863 |
| ICI 2D adaptation | 0.302 | 0.209 | 0.048 | 0.517 | 0.947 |
| Square-local control | 0.376 | 0.273 | 0.082 | 0.529 | 0.949 |
| Geometry adaptive | 0.344 | 0.245 | 0.127 | 0.429 | 0.812 |
| Fixed boundary | 0.267 | 0.179 | 0.031 | 0.601 | 1.000 |
| Adaptive boundary | 0.323 | 0.225 | 0.053 | 0.516 | 0.961 |
| Geometry + boundary | 0.327 | 0.224 | 0.057 | 0.538 | 0.993 |
| Two-view control | 0.351 | 0.249 | 0.070 | 0.517 | 0.986 |
| + consistency | 0.321 | 0.227 | 0.060 | 0.501 | 0.983 |
| Attention U-Net | 0.252 | 0.172 | 0.041 | 0.497 | 0.896 |

The complete unrounded table is `results/binary_study/comparison.csv`.

## Controlled differences

Paired positive-image bootstrap resamples **83 duplicate groups / 90 images**, not patients. Intervals are conditional on the trained checkpoints and this test set.

| Comparison | Dice difference | 95% interval | Interpretation |
|---|---:|---|---|
| Geometry versus ICI | +0.04185 | [0.00579, 0.07735] | Local supervision helped this comparator. |
| Adaptive geometry versus square-local | -0.03269 | [-0.06619, 0.00089] | No established shape-adaptive advantage. |
| Adaptive versus fixed boundary | +0.05655 | [0.02265, 0.08892] | Positive evidence for band scale adaptation. |
| Combined versus ICI | +0.02498 | [-0.01040, 0.05854] | Seed-42 interval includes zero. |
| Consistency versus equal two-view control | -0.03036 | [-0.05892, -0.00130] | Clean-image Dice decreased. |

## Repeated training seeds

| Method | Seeds | Mean Dice+ | Sample SD |
|---|---:|---:|---:|
| ICI adaptation | 3 | 0.27991 | 0.03332 |
| Geometry + boundary | 3 | 0.33736 | 0.02375 |

The combined method has a positive difference in all three paired seeds, but three seeds are a small sample; this does not override the seed-42 conditional interval or prove general superiority.

## Important limitations

Performance is modest and false positives are high. The model is **not clinically ready**. Geometry-aware support did not beat the matched square control, and JS regularization did not improve clean-image Dice. Do not advertise all three contributions as successful accuracy improvements or claim unmeasured SOTA models were outperformed.

Identical U-Nets have 1,811,746 parameters and a 13.57 convolution-GFLOP estimate at 384 pixels. Shared-GPU timing varied; it is not evidence of architectural speed gains among identical models. Complete HD/HD95 finite-case means and infinite/missed-case counts are exported separately.

## Deliverables and provenance

- Report: `deliverables/2403AI03_MLProject.pdf` (8 pages).
- Poster: `deliverables/2403AI03_MLProject_Poster.pdf` (1 page).
- Editable poster: `deliverables/2403AI03_MLProject_Poster.pptx` (1 poster slide).
- Public per-run metrics: `results/binary_study/experiments/`.
- Full checkpoints and raw/predicted images remain outside Git under ignored `runs/` and `data/`.

Operational corrections are documented in `runtime_corrections.json`. Failed attempts were preserved but excluded. Exported summaries clarify a legacy annotation-description field; numerical values are not changed.
