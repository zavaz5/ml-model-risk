# CLAUDE.md

HSE Research Seminar, module 2 (credit scoring): replication of Alonso Robisco & Carbó Martínez (2022), *Measuring the model risk-adjusted performance of machine learning algorithms in credit default prediction*, Financial Innovation 8:70 (CC BY 4.0, in `paper/`). The assignment for this paper: **reproduce every result it lists and propose an improvement for each.** Team: Group 8 (Nazarov, Doroshenko, Burlakov, Daunov, Sergeev).

Start every session by reading [docs/HANDOFF.md](docs/HANDOFF.md).

## Standing rules

1. **Numbers are never typed by hand.** Every number of ours in the README, slides and speaker script is read by a script from `results/`. The paper's numbers live once, with page references, in `src/mrisk/paper_numbers.py`; values read off its vector figures come from `scripts/extract_paper.py`. Edit generators (`scripts/make_report.py`, `scripts/make_slides.py`, `scripts/make_speaker_script.py`), never their outputs.
2. **Never put the paper's results and ours in the same table.** Separate tables or panels, each labelled with its source. (Instructor's rule.)
3. **Every metric gets ±**: DeLong standard errors for AUC, bootstrap (1,000 replications, seed 0) for everything else, Monte Carlo standard errors for simulations. (Instructor's rule.)
4. **State objectives explicitly:** what is fitted (to what loss), what is chosen (by which criterion, on which rows), what is evaluated (on which rows). (Instructor's question last time.)
5. **Every content slide has a number.** The HSE template's master provides it; check after export.
6. **The test set is locked.** The paper's 80/20 split (seed 0, stratified) is fixed in `results/reproduction/models/split_indices.npz`. Reproduction fits use the 80% training rows and report on the test rows, as the paper does. Every design choice of an improvement is made on the validation rows (20% of training) and pre-registered in `docs/preregistration.md`; test rows are read only once a choice is frozen.
7. **Never tune toward the paper's numbers.** Report what we get; explain gaps.
8. **Timing hygiene:** time single-threaded, on an otherwise idle machine, never while parallel jobs run; warm up first; report median and interquartile range; say what was timed.
9. **Code we can defend:** small functions, docstrings that say why, tests in `tests/`, `pytest -q` green before any commit. Libraries: numpy, pandas, scipy, scikit-learn, xgboost, shap, matplotlib, python-pptx, PyMuPDF, pytest, certifi.
10. **Git:** commit at milestones; push only after the user confirms. No author names and no Co-Authored-By lines (the team adds names itself). Never commit downloaded data (Kaggle's licence forbids redistribution; regulatory PDFs are third-party): `data/raw/` is ignored, `data/corpus_manifest.csv` records sources and hashes.
11. **The deck is short:** at most 15 minutes, about 11 main slides (30–40% shorter than the team's 16-slide VPIN deck), English throughout, HSE template.

## Pipeline (order matters)

```bash
source .venv/bin/activate
pytest -q
python scripts/download_data.py      # Kaggle GMSC (accept the competition rules first), lexicon, OpenML
python scripts/extract_paper.py      # Table 11 dictionary, Fig. 1 / Fig. 5 values from the PDF
python scripts/reproduce_models.py   # Tables 5 and 7, Figs. 6-8 (~25 min; timings first, then parallel retrains)
python scripts/reproduce_nlp.py      # Tables 2, 3, 8-10, Fig. 3 (needs data/raw/corpus/)
python scripts/extract_fig9_fig4.py && python scripts/reproduce_scorecard.py && python scripts/reproduce_extra.py
python scripts/improve_nlp.py && python scripts/improve_models.py && python scripts/improve_risk.py   # I1-I13
python scripts/appendix_factors.py && python scripts/extended_scorecard.py   # A1-A5, I14
python scripts/explore_logit_benchmark.py   # exploratory, not pre-registered
python scripts/reproduce_shap_figures.py && python scripts/make_paper_figures.py   # our versions of Figs. 1, 4, 6-8, 9
python scripts/make_report.py        # README results section
python scripts/make_slides.py        # docs/presentation.pptx (export the PDF with PowerPoint, see HANDOFF)
python scripts/make_speaker_script.py
```
