# Handoff: state of the project and what to do next

Numbers live in `results/`; documents are regenerated from them (CLAUDE.md, rule 1). This file says where things are and what is left.

## 1. Context

- **Course:** HSE Research Seminar (DSBA, 3rd year), module 2 (credit scoring), instructor Petr Lukianchenko. Rubric (lecture 1 of module 1): 4–5 presented, 6–7 own code, 8 similar results with discrepancies explained, 9–10 upgrades.
- **Assignment for this paper:** reproduce every result it lists and propose an improvement for each.
- **Paper:** Alonso Robisco & Carbó Martínez (2022), Financial Innovation 8:70, CC BY 4.0, in `paper/`.
- **Talk:** at most 15 minutes, 11 main slides (title to final) plus backups, English, HSE template (`docs/hse_template.pptx`, from the team's VPIN deck). The user asked for a deck 30–40% shorter than VPIN's 16 slides.
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
- **PDF export of the deck** (LibreOffice is not installed), from the project root:

  ```
  osascript -e 'with timeout of 240 seconds' -e 'tell application "Microsoft PowerPoint" to open (POSIX file "<abs>/docs/presentation.pptx")' -e 'delay 3' -e 'tell application "Microsoft PowerPoint" to save active presentation in (POSIX file "<abs>/docs/presentation.pdf") as save as PDF' -e 'tell application "Microsoft PowerPoint" to close active presentation saving no' -e 'end timeout'
  ```

## 4. Open tasks

1. Fill in the paper's number in the seminar list on the title slide (`make_slides.py`, title placeholder) once known.
2. Rehearse with `docs/speaker_script.pdf`; the notes give a time per slide (14.5 minutes in all).
3. If the team gets PwC Viewpoint access, add `prov_03_ifrs_pwc` to the corpus and rerun from `reproduce_nlp.py`.
