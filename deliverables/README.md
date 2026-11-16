# Submission files

**Author:** Umang Khatri, **roll number:** 2403AI03.

- `2403AI03_MLProject.pdf`: latest 15-page report following the updated submission structure.
- `2403AI03_MLProject.docx`: editable Times New Roman report following the newer `report_guidlines..pdf` structure, with numbered chapters, equations, figures and tables. Recheck pagination and export a fresh PDF after editing in Word.
- `2403AI03_MLProject_Poster.pdf`: single-page poster matching the uploaded template.
- `2403AI03_MLProject_Poster.pptx`: editable one-page poster, not a slide deck.
- `2403AI03_MLProject_Code.zip`: checked source/submission archive, excluding datasets/checkpoints/private driver metadata.
- `complete_comparison.csv` and `repeat_seed_summary.csv`: actual experiment results.
- `contribution_analysis.json`: controlled comparisons and conditional uncertainty.
- `poster_provenance.json`: template/example provenance.
- `figures/` and `poster_figures/`: report-ready images and a real-checkpoint native-resolution demo.
- `word_report_figures/`: Times New Roman charts and labelled annotation/prediction comparisons embedded in the Word draft, generated from actual saved artifacts.

The Word report began as an assisted draft, not a certification of independent authorship. The updated guidelines prohibit AI-written content: review and rewrite the text yourself, verify citations and follow the instructor's permitted-assistance policy before submission. To regenerate the original draft, use `uv run scripts/build_word_report.py --overwrite` (this replaces the Word file, including any manual edits, but not the existing PDF or poster).

No notebook is required. Code is available at the repository root. Raw datasets and checkpoint binaries are not included in Git. All numerical results are real; improvements are mixed and the model has substantial false-positive limitations.
