# Model Risk-Adjusted Performance of ML in Credit Default Prediction

**Contributors:** Group 8: Nazarov, Doroshenko, Burlakov, Daunov, Sergeev

A replication of Alonso Robisco and Carbó Martínez, *Measuring the model risk-adjusted performance of machine learning algorithms in credit default prediction* (Financial Innovation 8:70, 2022), for the HSE Research Seminar. The assignment for this paper: **reproduce every result it lists and propose an improvement for each.**

The paper scores five credit-default models (Lasso, CART, random forest, XGBoost, a multilayer perceptron) for the model risk a bank supervisor would see. Phase 1 gives each model a "darkness" score from 1 to 5 in three risk categories (statistics, technology, market conduct), from five measured proxies and the authors' judgement. Phase 2 weights the categories for three uses of the model (regulatory capital, credit scoring, provisions) by how often the category's terms appear in regulatory texts. Risk is then set against the AUC gain over a logistic regression.

We rebuild both phases from scratch: the six models on Kaggle's Give Me Some Credit with the paper's settings, the five proxies, and the term counts on 35 of the 36 regulatory documents the paper lists. We check the paper's own arithmetic, and we add one pre-registered improvement per result: uncertainty for every number, multiple-testing corrections, a common explainer for SHAP, a rule instead of expert scores, and the same protocol on two more datasets.

- Presentation: [docs/presentation.pptx](docs/presentation.pptx) ([PDF](docs/presentation.pdf)); what to say on each slide and likely questions: [docs/speaker_script.pdf](docs/speaker_script.pdf).
- Every departure from the paper and every decision where it is silent: [DEVIATIONS.md](DEVIATIONS.md).
- Verdict rules and improvements, fixed before they were run: [docs/preregistration.md](docs/preregistration.md).
- The paper (CC BY 4.0): [paper/alonso_carbo_2022.pdf](paper/alonso_carbo_2022.pdf).

## How to run

Python 3.13 on macOS (Apple silicon); XGBoost needs the OpenMP runtime (`brew install libomp`).

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
pytest -q
python scripts/download_data.py        # needs ~/.kaggle/kaggle.json and the GiveMeSomeCredit rules accepted on kaggle.com
python scripts/download_corpus.py      # the regulatory PDFs listed in data/corpus_manifest.csv
python scripts/extract_paper.py        # Table 11 and the values behind Figs. 1 and 5, from the paper's PDF
python scripts/reproduce_models.py     # Tables 5 and 7, Figs. 6-8 (~25 min on 14 cores)
python scripts/reproduce_nlp.py        # Tables 2, 3, 8-10
python scripts/reproduce_scorecard.py  # Tables 4 and 6, Fig. 5, errata
python scripts/improve_nlp.py          # I2-I6
python scripts/improve_models.py       # I1, I7, I11, I12
python scripts/improve_risk.py         # I13
python scripts/make_report.py          # this README's results section and results/report/key_numbers.json
python scripts/make_slides.py          # docs/presentation.pptx
python scripts/make_speaker_script.py  # docs/speaker_script.pdf
```

Data are not committed: Kaggle's licence forbids redistribution, and the regulatory texts belong to their publishers. `data/corpus_manifest.csv` lists every document's source URL, version and SHA-256.

## Results
