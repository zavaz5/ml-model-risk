# Pre-registration: verdict rules and improvements

Written on 20 November 2026, before any improvement was run and before we read the full-data outputs of the reproduction. Later changes go to the change log at the end, with date and reason.

The assignment is to reproduce every result the paper lists and to propose an improvement for each. Below, every result has an ID (R1–R13), the rule that decides our verdict, and one improvement (I1–I13) with its decision rule. Data: Give Me Some Credit, complete cases, stratified 80/20 train-test split (seed 0). Improvements make their choices on the validation rows (20% of training) and read the test rows only after they are frozen (CLAUDE.md, rule 6).

## Verdicts for the reproduction

- **Reproduced**: the paper's value lies inside our 95% interval, or, for an ordering, our point estimates give the same order.
- **Close**: the value lies outside our interval but the direction or ordering the paper draws from it holds.
- **Not reproduced**: the direction or ordering does not hold.
- **Not testable**: the paper gives too little to recompute it (we say what is missing).

Errata (internal inconsistencies of the paper) are reported separately from verdicts.

## Results and improvements

| ID | Paper's result (page) | Improvement | Decision rule |
|---|---|---|---|
| R1 / I1 | Literature: ML beats the logit by up to 20% AUC, and the gain is not monotone in algorithmic complexity (Fig. 1, p. 4) | A same-protocol benchmark: the five models on three public datasets (GMSC, Taiwan default, German credit), paper hyperparameters, AUC gain over the logit with a paired DeLong interval. Replaces a scatter that mixes datasets and metrics (recall, KS) | XGBoost "best everywhere" if its gain is the highest point estimate on all three datasets; "significantly best" where its paired DeLong test against the runner-up gives p < 0.05. Monotonicity: Spearman correlation of transparency class and gain, per dataset |
| R2 / I2 | "Stability" is mentioned 6.4 times per document, far more than the other statistical terms (p. 11) | Bootstrap interval of each mean over documents; share of mentions in the three documents that mention it most | The ordering is robust if the 95% intervals of "stability" and the next term do not overlap |
| R3 / I3 | Weights H̄ (Table 2, Fig. 3): statistics weighs most for capital; technology and conduct for scoring; provisions has the lowest total (p. 15) | (a) Document bootstrap (resample documents within each use and within the common area, 1,000 times): ± on every weight and the probability of each claim. (b) Dictionary audit: drop every term that alone accounts for more than 10% of its category's count in the whole corpus, then recompute | A claim is robust if it holds in at least 95% of bootstrap draws and also with the trimmed dictionary |
| R4 / I4 | Table 3: six of nine differences significant at 10% (p. 16) | Welch degrees of freedom (the paper states n − 1), a permutation test (10,000 draws), and Holm and Benjamini–Hochberg corrections over the nine tests | Report how many of the nine survive Holm at 10% and at 5% |
| R5 / I5 | Sentiment (Tables 8–10): technology and conduct terms are surrounded by more negative words in scoring documents (p. 28) | Divide by the base rate: lift = negatives per mention / (2d × share of negative words in that use's documents). Bootstrap over documents | Robust if scoring has the highest lift for technology and conduct in at least 95% of draws, at d = 10 |
| R6 / I6 | Worked example (Table 4): a model with Z = (2, 4, 3) has riskiness 0.523 / 0.554 / 0.536 for capital / scoring / provisions (p. 16) | Propagate the weight uncertainty (I3 bootstrap) into the three riskiness values | Report ± and the probability that scoring is the riskiest use |
| R7 / I7 | Stability (Table 5): Lasso and XGBoost are the most stable, CART and the MLP the least (p. 18) | Both of the paper's definitions: spread of test AUC and spread of individual PDs over the same 100 subsamples (60–100% of training rows), with bootstrap standard errors | The ordering holds under a definition if the paper's two most stable models have lower spread than its two least stable in at least 95% of paired bootstrap draws |
| R8 / I8 | Number of hyperparameters (Table 5) | An objective size measure: the number of fitted quantities (coefficients, tree nodes, network weights) | Report Spearman correlation with the paper's count |
| R9 / I9 | Training latency (Table 5): CART < XGBoost < Lasso < RF < MLP (p. 18) | Median and interquartile range of five single-threaded fits after a warm-up; times relative to the logit, which removes most of the hardware effect | Reproduced if our median order matches |
| R10 / I10 | SHAP latency (Table 5): XGBoost 8 s < RF 60 s < MLP 2,000 s (p. 18) | The same permutation explainer, background (100 training rows) and probability scale for all models, explaining the same 1,000 test rows | Report whether the order survives when the explainer is held fixed |
| R11 / I11 | SHAP rankings (Figs. 6–8): 90-days-late and 30–59-days in the top three for RF, XGBoost and the MLP; revolving utilisation first only for XGBoost; age fourth everywhere (p. 26) | All models on the probability scale (the paper's XGBoost plot is in log-odds); rankings over five seeds | A claim holds if it holds in at least four of five seeds on the probability scale |
| R12 / I12 | Table 7: AUC 80.0 (logit) to 85.3 (XGBoost); gains 0.2 to 5.3 points (p. 21) | Paired DeLong test of each gain; the use-specific metrics the authors ask for (p. 21): Brier score, expected calibration error (10 bins), calibration slope | Report whether the AUC order and the Brier order agree |
| R13 / I13 | Table 6 and Fig. 5: supervisory risk per model and use; XGBoost is the most risk-efficient model; provisions is the least risky use (pp. 19–21) | Replace the expert scores Z by a rule: each proxy mapped to [1, 5] by min–max on the log scale across the five models (linear for the AUC spread); a category's Z is the mean of its proxies. Then Monte Carlo (1,000 draws) over weights (I3), stability (I7), timings (I9) and AUC gains (DeLong) | A model is risk-efficient if no other model has both a larger gain and a lower risk (Pareto frontier). Report each model's probability of being on the frontier, per use. "XGBoost is the most risk-efficient" is supported if it is on the frontier in at least 95% of draws for all three uses |

## Change log

- **20 November 2026, before I13 was run.** I13 did not say which measurement feeds each category. We use the paper's own proxies as we measure them: statistics = mean of the rule scores for the spread of test AUC and the paper's hyperparameter count; technology = mean of the scores for training time (log scale) and the transparency class; conduct = the score for SHAP time with each library's default explainer (log scale). Gains in the Monte Carlo are redrawn by resampling test rows (paired across models) rather than from independent normal draws, so that the correlation between models' AUCs is kept. Reason: the rule should replace only the expert judgement, not the proxies.
- **20 November 2026, before the full run.** The Lasso is fitted with the saga solver instead of liblinear (DEVIATIONS D4): liblinear needed over 90 s per fit. This affects the Lasso's training time (R9, I9) and nothing else we pre-registered.
