# Methodology and claim boundaries

## Reference

Rachmadi et al., “Improving Segmentation of Objects with Varying Sizes in Biomedical Images using Instance-wise and Center-of-Instance Segmentation Loss Function.” PMLR 227:286–300, publication page 2024; PDF conference header MIDL 2023.

Upstream inspected at commit `3b5965486b775c06c99039a07b2ced5197cfb8ac`:

- https://github.com/BrainImageAnalysis/ICI-loss/blob/3b5965486b775c06c99039a07b2ced5197cfb8ac/losses/ICI_loss.py
- https://github.com/BrainImageAnalysis/ICI-loss/blob/3b5965486b775c06c99039a07b2ced5197cfb8ac/losses/tools.py

No upstream source files are vendored. This is an independent implementation of a documented adaptation, not a published-result reproduction.

The original instance term compares each ground-truth connected component to predicted connected components that intersect it. The center term compares fixed-size squares/cubes around component centers of mass. **It is not a fixed-size crop segmentation loss.** Appendix B excludes intersections with other true instances; our instance selector uses this no-TP variant.

## Explicit changes in our comparator

1. 2D U-Net and two-logit binary softmax, rather than original 3D residual U-Net MRI experiments. Optional multiclass mode is separate from the main study.
2. Global weighted CE plus foreground soft Dice. All main runs use CE weight 0.5, Dice weight 1, class weights [1,10], and balanced image sampling (one epoch retains the original training-image count of draws with replacement). These are common imbalance-handling settings, not contributions. ICI terms apply only to the fracture probability channel.
3. Exact 8-connected CCA on detached CPU arrays, not iterative Kornia-style GPU max pooling. Component identities and thresholds are nondifferentiable. Soft probabilities inside selected supports retain gradients. An entirely missed instance has no gradient from the selected instance term; the global objective supplies its recovery signal.
4. Predicted centroid squares contain **mean component probability**, not upstream's constant assignment. Thus gradients affect component confidence but **not** centroid positions or component connectivity. This is a confidence surrogate, not the original hard-center loss and not a straight-through claim.
5. Center loss averages over images, including negatives; instance loss averages over ground-truth components, with zero local contribution if none exist. Global supervision always remains active.
6. No minimum-component-size filtering or truncated CCA. Runtime can increase for fragmented early predictions; benchmark before large runs.

Always label E1 as **ICI-inspired 2D adaptation**, not “exact original ICI.” A numerical test on selected supports checks the instance equation, but cannot validate the original published training results.

## Global objective

Let `p` be softmax probabilities, `y` the categorical mask, and `V` the valid non-padding support. Weighted global CE is normalized by the sum of target-class weights over V. Foreground Dice is averaged across images and foreground classes, including absent classes to discourage spurious foreground. Epsilon is 1e-6. Binary background means every pixel not assigned to a fracture annotation, including bone; it is not an anatomical background mask.

## C1: geometry-adaptive supervision

For every original fracture polygon mask, construct a bounding rectangle expanded by `ceil(context * height)` and `ceil(context * width)`, with minimum one-pixel context. Evaluate soft Dice + binary CE on the **union fracture target** within that region, excluding padding. Average over instances. The proposed experiment replaces the fixed-center confidence term with this local term while retaining the same instance selector and global objective. C1 also uses polygon instances rather than CCA-normalized center targets; report that distinction, particularly for touching polygons.

Regions preserve aspect ratio but are axis-aligned. The `geometry_square` control uses the **identical local Dice+BCE objective**, but a square enclosing the instance, expanded by the same context fraction and clipped at image edges. This separates support geometry from simply adding a local loss. We do not claim orientation-aware supervision. Ground truth is used for supervision only, never to select inference regions.

## C2: scale-adaptive boundary weighting

For each nonempty instance, area A is its pixel count; P is the count of 8-neighborhood inner-boundary pixels (a discrete proxy, not calibrated Euclidean perimeter). Set `tau = clip(c*A/max(P,1), tau_min, tau_max)` or a fixed tau control. Build a distance-to-boundary map and weights `exp(-distance/tau)` within its expanded local support. Normalize the weighted binary CE by valid weight sum. Preserve global supervision for distant false positives and interiors.

This is not Kervadec's signed-distance boundary loss. Cite related boundary objectives, but describe this one correctly. Distances concern **annotation polygons**, which may not coincide with physical crack lines.

Ground-truth geometry is recomputed after resize/augmentation to maintain alignment. No unbounded cache is used. Cached geometry would require transformation-specific invalidation.

## C3: contrast consistency

Two appearance-only gamma views receive the same complete supervised objective, averaged; add symmetric multiclass Jensen–Shannon divergence on valid pixels. The two-view control runs the identical extra supervised pass with JS weight zero. Both use the same number of forward passes and mild gamma distribution [0.85, 1.15]. This controls for additional exposure/compute better than a one-pass baseline.

## Evidence required

- C1: E1 versus E2 plus aspect-ratio-aware versus square-local support with identical objective, size/aspect-ratio analysis and failure examples.
- C2: E3 versus E4, not only E1 versus E4. Fixed/adaptive controls share all other settings.
- Combined: E1 versus E5 and component controls.
- C3: E6 versus E7 with identical dual-view training.
- Additional seeds for the main pair; group-level paired bootstrap on the same test images.

No gain is promised. A component that fails should be reported as a negative result, not relabeled as an improvement. Existing instance-wise, geometry, boundary, and consistency methods must be acknowledged; this repository alone does not establish global novelty.

## Resume qualification

Checkpoint resume restores model, optimizer, scaler, main-process and sampler RNG states. Exact resume-versus-uninterrupted tests use zero data-loader workers. With persistent multiprocessing workers, their augmentation RNG streams are not checkpointed, so a resumed run can have different augmentations from an uninterrupted one. Fresh full runs use fixed seeds and the same worker count; cross-device bit-identical results are not promised.

CUDA FP16 overflow handling follows GradScaler's unscale/skip/update sequence. Gradient clipping is allowed to observe nonfinite gradients during AMP because unscale has already marked those updates to be skipped. Non-AMP nonfinite gradients remain fatal. Training logs count skipped and successful optimizer updates. The failed first baseline attempt is excluded from the clean comparative study; all methods use the corrected handling and the same fresh-start protocol.

Autocast wraps forward passes only. All loss calculations run outside its context with float32 logits, including probability-space local BCE and JS terms. The CUDA-specific BCE guard required a scope correction when the square control first started; its first attempt produced no completed checkpoint. Previously completed baseline/ICI losses already used float32 arithmetic, so their completed results are retained unchanged.
