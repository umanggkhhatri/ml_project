# ShapeGuard-Frac

Modular Python research project for **binary fracture-region segmentation** of X-rays. No notebook is required.

The main study predicts **0 = non-fracture background (including bone)** and **1 = annotated fracture region**. This is pixel-level segmentation, not image-level classification. It uses FracAtlas's existing fracture masks; **bone annotations are not needed**. Multiclass annotation tools remain an optional extension. This is not a clinical diagnostic system.

## Completed submission

All **14 GPU runs** are complete. The final documents use actual results and explicitly report mixed contribution effects and high false-positive rates.

- [Poster PDF](deliverables/2403AI03_MLProject_Poster.pdf) — one page, supplied IIT Patna template.
- [Editable poster](deliverables/2403AI03_MLProject_Poster.pptx) — one poster slide, not a slide deck.
- [Project report](deliverables/2403AI03_MLProject.pdf) — eight pages.
- [Full comparison table](results/binary_study/comparison.csv), [interpretation](docs/results.md), and [public per-run numerical artifacts](results/binary_study/experiments/).
- [Viva guide](docs/viva.md) and [mathematical justification](docs/mathematics.md).

Best observed seed-42 positive-image Dice is **0.376** (square-local control). Across three seeds, the combined method averages **0.337** versus **0.280** for the ICI adaptation. The seed-42 combined comparison interval crosses zero; contrast consistency reduced clean-image Dice. **These results do not establish clinical readiness or uniformly successful improvements.**

Model checkpoints and datasets are intentionally excluded from Git. The complete trained checkpoints remain in the local/server `runs/binary_study/` directories. Use the CLI and configurations to reproduce training, or inspect the exported numerical artifacts without training.

## Research scope

Base paper: Rachmadi, Poon & Skibbe, [Instance-wise and Center-of-Instance (ICI) loss](https://proceedings.mlr.press/v227/rachmadi24a.html), PMLR 227 (2024; PDF describes the MIDL 2023 conference). Original code: [BrainImageAnalysis/ICI-loss](https://github.com/BrainImageAnalysis/ICI-loss).

- **C1:** replace the fixed-center confidence surrogate with aspect-ratio-aware local fracture supervision.
- **C2:** instance-scale-adaptive boundary weighting, with a fixed-width control.
- **C3 (optional):** contrast consistency, with an equal two-view supervised control.

Our ICI comparator is an **explicitly modified 2D adaptation**, not an exact reproduction. See [methodology](docs/methodology.md) for the formulas, gradient limitations, departures from upstream, and claim boundaries; [mathematical justification](docs/mathematics.md) distinguishes design properties from unproven accuracy claims.

## Install on macOS or Linux

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --extra dev
uv run shapeguard doctor
uv run pytest
uv run shapeguard smoke --device cpu
```

`smoke` creates clearly labeled **synthetic** images, tests every configured loss, runs a short training/evaluation/reporting cycle, and verifies checkpoint resume. Its numbers are software checks, not medical results. Use `--device mps` to test Apple Silicon.

## Prepare FracAtlas

```bash
uv run shapeguard download --version 7
uv run shapeguard prepare --source data/raw/fracatlas-v7
uv run shapeguard split --manifest data/prepared/manifest.csv --seed 42
uv run shapeguard select --manifest data/prepared/manifest.csv --count 20 --output data/prepared/pilot.json
```

The official release is [FracAtlas v7](https://figshare.com/articles/dataset/The_dataset/22363012). Counts are derived from downloaded files, not copied from the older paper. Source discrepancies and EXIF-orientation issues are quarantined in `audit.json`. Exact/near-duplicate groups stay in one split; **these are not patient-level splits**, because reliable patient identifiers are unavailable.

Paths inside the manifest are relative to its directory. Keep the same `data/raw` and `data/prepared` layout when transferring data to the GPU server.

## Optional: create bone annotations for a future multiclass extension

Start with inexpensive **brightness proposals only**:

```bash
uv run shapeguard propose --manifest data/prepared/manifest.csv --selection data/prepared/pilot.json --method threshold
uv run shapeguard annotate --manifest data/prepared/manifest.csv --selection data/prepared/pilot.json
uv run shapeguard eda --manifest data/prepared/manifest.csv --selection data/prepared/pilot.json
```

Open **http://127.0.0.1:8765**. Paint/erase or draw polygons, save drafts, then explicitly accept inspected masks. Green is candidate bone; red is the existing fracture region. Thresholds often include soft tissue/text and miss bone. They must be corrected. Masks are stored at native resolution. The editor binds only to localhost and has no external service dependency.

Alternatively, install the optional original SAM adapter and supply your own trusted checkpoint plus bone-specific bounding boxes:

```bash
uv sync --extra dev --extra sam
uv run shapeguard propose --manifest data/prepared/manifest.csv --selection data/prepared/pilot.json --method sam --sam-checkpoint /path/to/sam_vit_b.pth --prompts /path/to/prompts.json
```

Prompt JSON uses native `(x0, y0, x1, y1)` coordinates:

```json
{"IMG0000001": {"boxes": [[10, 20, 150, 300], [170, 30, 260, 310]]}}
```

SAM is an annotation aid, not a verified bone model. Its optional path must be tested with an actual checkpoint before relying on it. No model checkpoint is bundled or automatically downloaded.

Read [annotation protocol](docs/annotations.md). Accepted masks are immutable in that dataset version. Multiclass training freezes the entire accepted cohort and mask hashes before experiments. To expand or revise a frozen cohort, create a new prepared dataset version; do not edit the original run's data.

## Train the binary study

All default and GPU study configs are binary. Check eligibility and resizing first:

```bash
uv run shapeguard audit-ready --manifest data/prepared/manifest.csv --size 384 --task binary
uv run shapeguard train --config configs/baseline.yaml
uv run shapeguard train --config configs/ici.yaml
uv run shapeguard train --config configs/geometry.yaml
uv run shapeguard train --config configs/boundary_fixed.yaml
uv run shapeguard train --config configs/boundary_adaptive.yaml
uv run shapeguard train --config configs/proposed.yaml
```

Optional: `configs/attention.yaml`, `configs/contrast_control.yaml`, `configs/full.yaml`. Every component is independently configurable. `configs/binary.yaml` runs fracture-only training without bone labels; it must not be reported as multiclass performance.

All main runs share balanced positive/negative **image sampling**, the same 384-pixel resolution, U-Net widths, and weighted CE + foreground Dice objective. CE class weights are `[1, 10]`; these are common baseline settings, not claimed contributions. The 16-epoch common budget was fixed after a shared-GPU runtime preflight and before comparative test evaluation. GPU configs enable CUDA AMP and cap the PyTorch allocator at 7.5% of a 40 GB device (approximately 3 GB).

For a complete sequential CUDA study, set the assigned GPU through `CUDA_VISIBLE_DEVICES`, then:

```bash
uv run shapeguard study --configs configs/gpu --output results/binary_study --repeat-seeds 7 123
```

This runs the six core experiments plus a square-local geometry control, equal-compute contrast control, consistency model, Attention U-Net comparator, and ICI/proposed repetitions. It saves resumable status, actual comparison tables, paired positive-image bootstrap intervals and figures. No parallel GPU training processes are started. Existing completed runs are reused only with matching configurations; interrupted training resumes from its checkpoint.

Resume your own trusted checkpoint:

```bash
uv run shapeguard train --config configs/proposed.yaml --resume
```

Only the epoch limit may change when resuming. Changed data/configuration is rejected. Runs save environment, eligibility lists, content hashes, validation curves, optimizer/RNG state, `best.pt` and `last.pt`. Do not load untrusted checkpoints: optimizer/RNG checkpoint loading uses pickle.

## Evaluate and produce report-ready figures

```bash
uv run shapeguard evaluate --checkpoint runs/binary_study/proposed/seed_42/best.pt --robustness
uv run shapeguard figures --checkpoint runs/binary_study/proposed/seed_42/best.pt --evaluation runs/binary_study/proposed/seed_42/evaluation_test
uv run shapeguard compare --evaluations runs/binary_study/ici_2d_adaptation/seed_42/evaluation_test runs/binary_study/proposed/seed_42/evaluation_test --output results/comparison.csv
uv run shapeguard bootstrap --first runs/binary_study/ici_2d_adaptation/seed_42/evaluation_test/fracture_metrics.csv --second runs/binary_study/proposed/seed_42/evaluation_test/fracture_metrics.csv --positive-only --output results/paired_bootstrap.json
uv run shapeguard predict --checkpoint runs/binary_study/proposed/seed_42/best.pt --image /path/to/xray.jpg --output runs/demo
```

Evaluate every run before comparison. Results include class metrics, fracture-positive summaries, negative-image false positives, one-to-one matched instance performance, boundary distances, resize audits, parameter counts, synchronized latency, and convolution-only FLOP estimates.

**Metric policies:** both-empty Dice/IoU = 1 and HD = 0; one-empty HD = infinity, with infinite cases separately reported. Undefined precision/recall are missing, not zero. Boundary distances are in pixels at the stated network resolution, not millimetres. Headline fracture Dice uses positive images, not many correctly empty masks. Bootstrap resamples duplicate groups, not patients. Synthetic corruption tests do not prove hospital generalization.

## Institute GPU

See [GPU guide](docs/gpu.md). Check the assigned GPU/scheduler and create an isolated environment. The same CLI supports CPU, MPS and CUDA; CUDA AMP is optional. Start with a measured one-epoch benchmark before budgeting all experiments.

**Never put passwords, SSH private keys or tokens in Git.** Rotate any credential previously disclosed. No credentials or automatic remote login scripts are included.

## Submission assets

After the study is complete, generate an **8-page report and single-page poster** from actual artifacts:

```bash
uv sync --extra dev --extra report
uv run shapeguard deliverables --study results/binary_study --output deliverables
```

The builder refuses unfinished studies. CSVs and figures remain the source of every reported number. Component effects are described according to their measured differences and paired intervals, including negative findings. No notebook or slide deck is created in the configured submission workflow. See [experiment checklist](docs/experiments.md).

Submission identity is stored in `submission.json`. The updated deliverable is a poster based on the supplied institutional template, not a slide deck. Outputs include `2403AI03_MLProject.pdf` (report) and `2403AI03_MLProject_Poster.pdf` / `.pptx` (single-page poster). Use `--report-only` for the report independently. See [poster specification](docs/poster.md) and [viva guide](docs/viva.md).

## Attribution

FracAtlas is CC BY 4.0; cite its [dataset paper](https://www.nature.com/articles/s41597-023-02432-4). Attribute ICI and all related techniques. Dataset/checkpoint licenses remain separate from this code. Student-generated bone masks must be identified as student/AI-assisted annotations.
