# Handoff: state of the project and what to do next

Numbers live in `results/`; documents are regenerated from them (CLAUDE.md, rule 1). This file says where things are and what is left.

## 1. Context

- **Course:** HSE Research Seminar (DSBA, 3rd year), module 2 (credit scoring), instructor Petr Lukianchenko. Rubric (lecture 1 of module 1): 4–5 presented, 6–7 own code, 8 similar results with discrepancies explained, 9–10 upgrades.
- **Assignment for this paper:** reproduce every result it lists and propose an improvement for each.
- **Paper:** Alonso Robisco & Carbó Martínez (2022), Financial Innovation 8:70, CC BY 4.0, in `paper/`.
- **Talk:** English, HSE template (`docs/hse_template.pptx`, from the team's VPIN deck). On 20 November the user asked for the main slides to show every result of the paper next to ours, one result per slide, with an easy-to-read explanation on the slide wherever they differ, and to set the 15-minute limit aside for now: 16 main slides (title to final, about 17.5 minutes by the notes), 11 backups. Each result slide has the paper's version left (cropped from its PDF or its table), ours right, a green "same" or orange "why it differs" box, and a blue box with our improvement of that result. Extra checks beyond the paper's results (appendix factors, the log-input logit, the three-dataset benchmark, the Monte Carlo chart) are backups. There is no verdict scorecard slide; the README keeps the verdicts.
- **Team:** Group 8: Nazarov, Doroshenko, Burlakov, Daunov, Sergeev. The paper's number in the seminar list and the talk date are not known yet (the title slide has no number).
- **Instructor's rules from earlier talks:** slide numbers; say what is fitted, chosen and evaluated; never mix the paper's numbers and ours in one table; ± on metrics.

## 2. Where things are

| What | Where |
|---|---|
| Code | `src/mrisk/` (data, models, proxies, uncertainty, nlp, corpus, scorecard, plots, paper_numbers), `scripts/`, `tests/` |
| Raw data (not committed) | `data/raw/gmsc/cs-training.csv` (Kaggle), `data/raw/corpus/*.pdf` (regulatory texts), `data/raw/lexicon/`, `data/raw/openml/` |
| Committed data | `data/dictionary.csv` (the paper's Table 11), `data/corpus_manifest.csv` (sources, versions, hashes), `data/checksums.json` |
| Results | `results/paper_extracted/` (values behind the paper's vector figures), `results/reproduction/{models,nlp,scorecard}`, `results/improvements/{models,nlp,risk}`, `results/report/key_numbers.json` |
| Documents | `README.md` (results section generated), `DEVIATIONS.md`, `docs/preregistration.md`, `docs/presentation.pptx` + `.pdf`, `docs/speaker_script.pdf` |

## 3. Gotchas

- **Kaggle:** the account must accept the GiveMeSomeCredit competition rules once on kaggle.com, or the API answers 403.
- **XGBoost on macOS** needs `brew install libomp`.
- **Lasso solver:** liblinear's L1 solver takes over 90 s per fit on the squared features; we use saga (DEVIATIONS D4).
- **pandas 3 and OpenML:** scikit-learn's pandas ARFF parser fails; `download_data.py` uses `parser="liac-arff"`.
- **Corpus hosts:** banque-france.fr refuses scripted downloads; EUR-Lex shows a bot challenge (we used the Publications Office copy); two files come from the Internet Archive. `download_corpus.py` reports any file it cannot fetch or whose hash differs.
- **Timing hygiene:** `reproduce_models.py` times first (single thread), then runs the parallel refits. Do not run other heavy jobs during its first ~20 minutes.
- **PDF export of the deck:** PowerPoint on this Mac has no subscription and opens files read-only, so it cannot save a PDF. Keynote can (LibreOffice is not installed), from the project root:

  ```
  A=$(pwd); osascript -e 'with timeout of 300 seconds' -e 'tell application id "com.apple.Keynote"' -e "set d to open (POSIX file \"$A/docs/presentation.pptx\")" -e 'delay 3' -e "export d to (POSIX file \"$A/docs/presentation.pdf\") as PDF" -e 'close d saving no' -e 'end tell' -e 'end timeout'
  ```

  Then check that every content slide shows its number (the final slide has none).

## 4. State on 20 November (evening)

- 16 results checked (R1–R16), 11 errata, improvements I1–I14, appendix factors A1–A5, one exploratory analysis (X1: a logit with log inputs nearly matches XGBoost).
- Fig. 1 was checked against its sources: a literature search for the open-access ones, and copies of Jones et al. (2015) and Chen & Xiang (2017) that the team downloaded (subscription and browser). Every one of its 20 points is now checked; `data/fig1_sources.csv` gives the table or page for each number.

## 5. Open tasks

1. Fill in the paper's number in the seminar list on the title slide (`make_slides.py`, title placeholder) once known.
2. Rehearse with `docs/speaker_script.pdf`; the notes give a time per slide (14.5 minutes in all).
3. If the team gets PwC Viewpoint access, add `prov_03_ifrs_pwc` to the corpus and rerun from `reproduce_nlp.py`.
