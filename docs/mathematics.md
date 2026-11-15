# Mathematical justification (not a proof of accuracy gains)

The equations below justify design properties. Whether the modifications improve segmentation must still be established by controlled experiments.

## C1: support geometry and foreground dilution

Suppose an instance has an axis-aligned bounding rectangle of height `h` and width `w`, with `w >= h`, away from image borders. Both controls use the same context fraction `c`.

The approximate support areas are:

```
Adaptive rectangle: S_a = (1 + 2c)^2 h w
Enclosing square:   S_s = (1 + 2c)^2 w^2
S_a / S_s = h / w = 1 / aspect_ratio
```

For unchanged instance area `A`, the foreground fraction `A/S` is consequently about `w/h` times larger in adaptive support. This reduces geometrically unnecessary background for elongated instances while retaining context along each dimension. It does not prove higher Dice: extra background could also provide useful context, which is why the identical-objective square control is necessary. Pixel rounding and boundary clipping make this relation approximate in code.

The local objective is averaged across annotated instances:

```
L_geometry = mean_k [ L_Dice(p|R_k, union_target|R_k)
                     + L_BCE(p|R_k, union_target|R_k) ]
```

Using the union target prevents another true fracture in a crop from being mislabeled as background. Padding is excluded. The square and adaptive controls share the exact same local objective and coefficient.

## Why small/thin objects are boundary-sensitive

For a region of area `A` and perimeter `P`, a small approximately uniform boundary displacement `delta` changes the symmetric-difference area by roughly `P*abs(delta)`. If prediction and target areas remain close, then:

```
1 - Dice approximately P*abs(delta)/(2A)
```

Thus a large perimeter-to-area ratio makes a fixed pixel displacement more damaging. This is a small-perturbation geometric approximation, not an identity for arbitrary fracture masks.

## C2: a scale-consistent boundary-band parameter

Before clipping, `tau = c*A/P` has units of length. Under continuous uniform spatial scaling by `s`:

```
A' = s^2 A,   P' = s P
tau' = c*A'/P' = s*tau
d' = s*d
exp(-d'/tau') = exp(-d/tau)
```

Therefore the un-clipped distance weighting is invariant in normalized spatial coordinates. This is preferable to a fixed pixel width when instance scale varies. In the implementation, P is a discrete boundary-pixel-count proxy and tau is clipped to `[1,8]`; exact equivariance is not claimed under discretization or clipping.

The normalized local weighted BCE is:

```
L_boundary,k = sum_i v_i exp(-d_i/tau_k) BCE(p_i,y_i)
               / sum_i v_i exp(-d_i/tau_k)
```

Here `v_i` is valid-pixel support. Positive weights and a nonempty accepted instance give a positive denominator. Normalization prevents a larger boundary band from trivially increasing the loss simply by containing more weighted pixels. The common global objective still supervises interiors and distant false positives.

## C3: consistency and the collapse counterexample

For softmax distributions `p` and `q`, let `m=(p+q)/2`:

```
JS(p,q) = 0.5 KL(p || m) + 0.5 KL(q || m)
```

By KL nonnegativity, JS is nonnegative and is zero when the distributions agree. With natural logarithms and equal mixture weights, JS is bounded by `log(2)`.

Agreement alone does not establish correctness: two constant background predictions can have zero JS while missing every fracture. Hence both views retain segmentation supervision:

```
L_two_view = 0.5 [ L_supervised(x,y) + L_supervised(T(x),y) ]
             + lambda_JS JS(f(x),f(T(x)))
```

The appearance transform preserves label geometry. The compute control uses the same two supervised passes with `lambda_JS=0`. Supervision penalizes incorrect constant predictions, but no theorem here guarantees optimization avoids collapse or achieves clinical robustness.

## Metrics and conditional uncertainty

For one image, `Dice = 2*IoU/(1+IoU)` follows by substituting TP/FP/FN definitions. Applying this formula to separately averaged Dice and IoU is invalid in general because averaging does not commute with a nonlinear transformation.

Paired bootstrap estimates differences conditional on trained checkpoints and this test cohort. Resampling duplicate groups addresses known image dependence only. It does not recover unavailable patient IDs, account for all scanner effects, or replace repeated training seeds.
