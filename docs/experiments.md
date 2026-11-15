# Experiment checklist

| ID | Config | Purpose |
|---|---|---|
| E0 | baseline.yaml | U-Net + CE/foreground Dice |
| E1 | ici.yaml | Explicit ICI-inspired 2D adaptation |
| E2 | geometry.yaml | Replace center confidence with geometry supervision |
| E2-control | geometry_square.yaml | Same local objective with enclosing-square support |
| E3 | boundary_fixed.yaml | ICI + fixed boundary control |
| E4 | boundary_adaptive.yaml | ICI + scale-adaptive boundary |
| E5 | proposed.yaml | Geometry + adaptive boundary |
| E6 | contrast_control.yaml | E5 + two supervised gamma views |
| E7 | full.yaml | E6 + JS consistency |
| A0 | attention.yaml | Optional architectural comparator |

Before starting:

- [ ] Dataset source/version/checksum saved.
- [ ] Quarantined source records and near-duplicate candidates examined.
- [ ] Group-disjoint splits and anatomical distribution inspected.
- [ ] Main binary study uses original fracture masks; no bone annotation dependency.
- [ ] `audit-ready` confirms positive cases and reports disappearing instances.
- [ ] Unit tests and device smoke tests pass.
- [ ] Final resolution selected before main comparisons.
- [ ] Balanced image sampling and weighted CE settings identical across all methods.
- [ ] Main experiment budgets and tuning ranges fixed; validation-only tuning.

After training:

- [ ] Evaluate best-validation checkpoints on the same frozen test cohort.
- [ ] Report class metrics, fracture-positive metrics and negative false positives separately.
- [ ] Report HD infinite/missed cases, not just finite-case means.
- [ ] State distances are network-resolution pixels, not clinical millimetres.
- [ ] One-to-one instance matching threshold = 0.1, with merged/missed predictions handled explicitly.
- [ ] Size groups use training-only fractional-area tertiles.
- [ ] Include convolution-only FLOP caveat and synchronized timing hardware.
- [ ] Repeat E1/E5 with more seeds if possible.
- [ ] Compare C2 against fixed-width control and C3 against equal two-view control.
- [ ] Compare C1 with identical local objective using square versus aspect-ratio-aware support.
- [ ] Include failures and negative results; no promised accuracy numbers.
- [ ] Regenerate every report table/figure from actual saved artifacts.

The deadline in the supplied brief is 15 November 2026. A completed smaller study is preferable to reporting unfinished experiments as completed. Synthetic smoke runs cannot be used as medical results.
