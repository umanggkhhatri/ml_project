# Poster submission

The presentation deliverable has been replaced by a **single-page poster** using the uploaded `updated poster sample.pptx`. The reusable copy is `assets/poster_template.pptx`: it retains layout and logos but removes unrelated sample sensor text/figures. `assets/poster_template.provenance.json` records both hashes. The original in the parent project folder is not modified.

## Confirmed format and identity

- One landscape page, **13 1/3 x 7 1/2 inches (16:9)**, matching the sample.
- Editable one-page PowerPoint poster plus a matching vector-text PDF.
- Author: **Umang Khatri**.
- Roll number: **2403AI03**.
- Email: **umang_2403ai03@iitp.ac.in**.
- Affiliation: **Department of Computer Science and Engineering, Indian Institute of Technology Patna**.
- Supplied IIT Patna / IMPACT logos retained.

There is no ten-slide deck in the final submission. The old slide generator remains an optional legacy utility, not the active deliverable.

## Layout

The three-column template is retained, with minor text-box reflow to avoid overlaps:

1. **Left:** abstract, introduction, problem statement.
2. **Centre:** contributions, methodology, compact architecture diagram, algorithm.
3. **Right:** actual baseline/ablation table, comparison plot and a qualitative example, conclusion.
4. **Bottom:** project-relevant references, replacing all sample photonic-sensor references.

Every statistic comes from the completed binary study. The figure example is selected deterministically at median positive-image Dice, then cropped around the annotation for visualization **after full-image inference**. That crop is not an inference prompt. The conclusion must reflect the actual paired difference, including negative or mixed findings.

## Build

```bash
uv sync --extra dev --extra report
uv run shapeguard deliverables --study results/binary_study --output deliverables
```

This generates the eight-page report and the poster, using `submission.json`. To generate only the poster:

```bash
uv run shapeguard poster --study results/binary_study --output deliverables
```

Expected poster outputs:

- `deliverables/2403AI03_MLProject_Poster.pdf`
- `deliverables/2403AI03_MLProject_Poster.pptx`
- `deliverables/poster_provenance.json`

Generation refuses an unfinished study. No sample sensor results or fabricated fracture metrics may be retained. Both poster outputs preserve the institutional logos and the template's page dimensions. The PDF uses Times-family vector text; the editable poster uses Times New Roman.
