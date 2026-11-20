# Deviations and decisions

Every place where we depart from the paper, and every decision the paper leaves open. Page numbers refer to the journal PDF in `paper/`. Internal inconsistencies of the paper itself (errata E1–E8) are listed separately, by code, in `results/reproduction/errata.json` and in the README.

## Data and models (pp. 17–21)

| ID | Paper | Ours | Why |
|---|---|---|---|
| D1 | 120,000 loans from "Give Me Some Credit" | The 120,269 complete cases of `cs-training.csv` (150,000 rows) | Dropping rows with a missing value is the only rule we found that gives the paper's count; the other 29,731 rows lack monthly income |
| D2 | 11 characteristics and their squares, 22 variables | 10 characteristics and their squares, 20 variables | The file has 10 borrower characteristics. Its eleventh column is an unnamed row index, which carries no information about the borrower, so we drop it. The paper's SHAP figures (Figs. 6–8) also show 10 features |
| D3 | 80/20 train-test split | Stratified 80/20 split, seed 0, saved in `results/reproduction/models/split_indices.npz` | The paper gives no seed and does not say whether it stratifies |
| D4 | Hyperparameters chosen by 5-fold CV (AUC) | The paper's reported values for CART, RF, XGBoost and the MLP; the Lasso penalty chosen by 5-fold CV over 11 values of C from 10⁻⁴ to 10, fitted with scikit-learn's saga solver | The paper reports no Lasso penalty, grid or solver. With the heavy-tailed squared features, liblinear's L1 solver took over 90 s per fit at C = 1; saga converges in about 18 s. The solver choice drives the Lasso's training time (R9) |
| D5 | – | Linear models and the MLP see standardised inputs; trees see raw values | The squared features span many orders of magnitude; neither the solver nor Adam converges on raw values. Trees are unaffected by monotone rescaling |
| D6 | MLP: 3 hidden layers (300, 200, 100), ReLU, sigmoid output, Adam | scikit-learn's `MLPClassifier` with those layers, early stopping on 10% of the training rows (patience 10, at most 200 epochs) | The paper does not report epochs, batch size or learning rate. Early stopping keeps the fit time near the paper's 31 s |
| D7 | XGBoost: depth 5, 40 trees; RF: depth 20, 100 trees; CART: depth 7 | Same, with the libraries' defaults for everything else (XGBoost learning rate 0.3, histogram method; RF √p features per split) | Not reported |
| D8 | Stability: standard deviation over 100 retrains on 60–100% of the training set (p. 12); "SD in the AUC-ROC for 100 simulations with different train–test partitions" (p. 18) | 100 retrains on random subsamples of the training rows, size drawn uniformly from 60% to 100%, the same 100 subsamples for every model; spread of test AUC (and, as an improvement, of individual PDs) | The paper describes two different procedures; ours follows p. 12 and keeps the test rows fixed so that only the training sample varies |
| D9 | Training latency in seconds | Median of five single-threaded fits on the full training rows after one warm-up fit, with the interquartile range | The paper does not say how it timed, or on what hardware |
| D10 | SHAP latency: 0 s (Lasso, CART), 8 s (XGBoost), 60 s (RF), 2,000 s (MLP) | Seconds to explain 1,000 random test rows: path-dependent TreeSHAP for the trees (raw output: probability for RF and CART, log-odds for XGBoost, as in the paper's Fig. 7), the permutation explainer on the probability with 100 training rows as background for the MLP and the linear models | The paper does not say how many rows it explained, or with which explainer. Fig. 7's axis (−1 to 3) shows XGBoost was explained in log-odds |
| D11 | Feature importance plots with 10 features | Mean absolute SHAP value per characteristic, with the contributions of x and x² summed per row first (SHAP values are additive) | Gives the 10-feature ranking of Figs. 6–8 from a 20-column design |
| D12 | Darkness scores Z set by the authors' judgement (Table 5) | Reproduction uses the paper's Z as given; improvement I13 replaces them by a pre-registered rule | Judgement cannot be recomputed |

## Regulatory texts (pp. 13–16, 26–28)

| ID | Paper | Ours | Why |
|---|---|---|---|
| D13 | Documents of Table 12 | Official public copies where they exist; the version, URL and SHA-256 of each file are in `data/corpus_manifest.csv`, with the documents we could not obtain marked `missing` | Several are drafts later replaced by final versions, and some IIF reports are not public |
| D14 | Table 12 assigns documents to uses | We follow Fig. 9 (erratum E8): the EBA report on innovative uses of consumer data counts in the common area, and IIF (2018) is left out | Fig. 9 shows the documents the computation actually used |
| D15 | Terms "plus their lemmatization"; no code | PyMuPDF text; words hyphenated across lines joined; lower-case tokens; plural nouns lemmatised to the singular (what NLTK's WordNet lemmatiser does with its default part of speech); a hyphenated or two-word term also matches its closed form ("back-testing" and "backtesting"); every term counted independently, so overlapping terms both count | The paper's tokenisation, lemmatiser and overlap rule are not reported |
| D16 | Table 11 says 54 statistical terms | The table lists 53; the technology column lists "Cyber-risk" twice (30 distinct terms, as stated) | Erratum E7; we use the terms as listed, without duplicates |
| D17 | Negative words "within d words" of a mention | Up to d tokens on each side of the first token of the mention, excluding the token itself; Hu and Liu's list as distributed with NLTK | Window definition not reported |
| D18 | t-test with "n − 1 degrees of freedom" on the per-document shares | Reproduction: Welch's t-test (unequal variances, Welch–Satterthwaite degrees of freedom). Improvement I4 adds permutation tests and multiple-testing corrections | The paper's degrees of freedom are not defined for two samples of different size |
