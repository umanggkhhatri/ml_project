# Implementation verification history

Verified locally on Apple M1 / macOS with Python 3.11.17 and the dependency versions recorded in `uv.lock` and run environment artifacts.

## Binary-study update

The professor-approved main scope is binary fracture-region segmentation. The numerical training suite includes explicit CUDA index selection and GradScaler overflow behavior. Poster tests verify a single PDF page, single editable poster slide, text bounds, preserved branding/template dimensions and removal of sample sensor content. Ruff checks pass. Submission identity is Umang Khatri / 2403AI03. The updated deliverable is a poster using the uploaded institutional template; there is no final slide deck.

The institute A100 CUDA environment has been established in an isolated project directory. A full-data runtime preflight used approximately 940 MB allocated / 1,244 MB reserved PyTorch memory. The common study budget is 16 epochs, fixed before comparative test evaluation. The original first attempt stopped during baseline training because strict clipping incorrectly rejected a recoverable FP16 overflow. Its artifacts and checkpoint are preserved separately and will not be included in comparisons. The corrected code lets GradScaler skip overflowing updates, logs skipped steps, and retains the strict float32 nonfinite-gradient guard.

Final comparative results are not yet claimed in this history file. The original initial-verification notes below refer to the earlier multiclass planning stage, not the current binary-study scope.

## Initial completed checks

- 31 pytest tests pass, including polygon validation, corrupted-source quarantine, duplicate-file handling, mask precedence, padding alignment, annotation freezing, loss gradients, empty-mask metrics, one-to-one instance matching, tiny-sample overfitting, and exact CPU resume-versus-uninterrupted dual-view training.
- Ruff checks pass.
- Python modules compile.
- Synthetic train/resume/evaluate/robustness/figure pipelines pass on CPU and MPS.
- A 20-image **real FracAtlas binary pipeline check** completes two training epochs, evaluation, figures and native-resolution inference on MPS. It is not a multiclass or comparative experiment.
- Annotation editor JavaScript parses successfully. HTTP checks cover 20 image records, native image/mask dimensions, restricted file access and rejected cross-origin writes.
- A wheel builds and includes the editor HTML.

## Actual source audit

Official FracAtlas v7 archive SHA256:

`b67ec2d290a022b3dcf47f78e9a37f7edcc80592c0571f439355bf00bd9f0e23`

The downloaded CSV has 4,083 rows despite the updated landing-page count. After conservative audit exclusions, this run retains **3,997 usable images**, **715 positive images**, and **917 original polygon instances**. Exact duplicate filename copies, truncated JPEGs, EXIF-orientation records and out-of-range polygons are recorded in `data/prepared/audit.json`. Source images remain untouched.

Group-disjoint split from seed 42:

| Split | Images | Fracture-positive |
|---|---:|---:|
| Train | 2,967 | 528 |
| Validation | 525 | 97 |
| Test | 505 | 90 |

Ratios are approximate because duplicate groups are indivisible. These are not verified patient-level splits. Anatomy distributions are saved separately.

## Initial scope: not completed / not claimed

- No accepted real bone annotations at initial verification. Twenty threshold **proposals** exist; inspection shows they need correction.
- No main multiclass baseline, comparative study, ablation results, or proven improvement.
- No institute CUDA execution or CUDA AMP verification.
- No actual SAM-checkpoint execution; that adapter is optional and requires a trusted external model plus prompts.
- No clinical validation and no proof of global novelty.
- No GitHub push: implementation remains in the local clone until publication is requested.

Detailed smoke and pilot outputs live under ignored `runs/`, not in the source repository. They are software checks and must not be presented as the project's final research results.
