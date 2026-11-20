# What graders asked of talks this year, and where our deck answers it

Written on 20 November 2026 from the graders' comments on other teams' talks in this seminar and from two
replications that received 10/10 (a bond-risk-premia project and a "virtue of complexity" project). Slide
numbers refer to `docs/presentation.pdf`.

| Comment from graders (paraphrased) | Where we answer it |
|---|---|
| Make slides minimalistic; do not put too much text on them | Results slides 5–10: one row per result, one short line in each box; details in the speaker notes and the appendix (slides 15–25) |
| Decode the formula; what are the notations? | Slide 3: every symbol of the one equation in a legend, with units and a worked number from the paper |
| Which method, and why those settings? | Slide 4: the paper's settings (p. 17, its 5-fold CV), kept on purpose so that differences are the paper's, not ours |
| What was the training objective? Training vs calibration objective | Slide 4: log-loss or Gini to fit, AUC to evaluate; calibration added because capital and provisions need calibrated PDs (appendix) |
| What was the improvement, and why is it one? ("simplicity is not an explanation") | Blue boxes on slides 5–10: each says what we changed and what it shows; the cue cards' question "What exactly is an improvement" |
| Out-of-sample testing not addressed | Slide 12: locked test rows, two more datasets, and out of time named as impossible here (no dates) and as the first next step (slide 13) |
| Dimensionality and explainability | Slide 5: model size as fitted parameters (Lasso vs RF); slide 9: SHAP; slide 12: RF memorises its training rows |
| Do not put results from different sources in the same table; disclaim numbers that are not comparable | Paper left, ours right, always separate; times marked orange and "not comparable" (slide 5) |
| Slide numbers; titles on graphs; units of measurement | Every slide numbered by the template; every figure has a title; units in every table header and axis (README, "Conventions and units") |
| Check slide layout carefully | Every slide rendered and checked after each build |
| Do not read; present in your own words | Speaker notes are cues, not sentences (`docs/speaker_script.pdf`) |
| Only one student answered questions | Each likely question names the speaker whose slides it is about (Speakers 1–5) |
| A well-composed repository, not only notebooks; submit the slides | Code in `src/` and `scripts/`, tests, one-command run (`scripts/run_all.py`), slides and cue cards in `docs/` |
| How could the results be overfit, and how did the authors and you check? | Slide 12 |

What the two 10/10 replications did that we adopted:

- **One command, quick and full modes, with consistency checks** (`scripts/run_all.py`).
- **A table mapping each result of the paper to the script and output that make it** (README).
- **A falsification check in the critique**: the placebo of slide 11 replaces the regulatory-text weights by random ones (exploratory X2).
- **Conventions and units stated once** (README).
- **"Holds up / contested" verdicts** on the critique slide.
