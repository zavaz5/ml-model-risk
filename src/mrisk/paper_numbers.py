"""Every number we quote from the paper, typed once, with its source.

Alonso Robisco, A. and Carbó Martínez, J. M. (2022). Measuring the model
risk-adjusted performance of machine learning algorithms in credit default
prediction. Financial Innovation 8:70. Page numbers refer to the journal PDF
("Page X of 35").

Nothing in this module is computed. Values read off the paper's vector
figures are produced by ``scripts/extract_paper.py`` instead and saved under
``results/paper_extracted/``.
"""

MODELS = ["Lasso", "CART", "RF", "XGBoost", "MLP"]
CATEGORIES = ["statistics", "technology", "conduct"]
USES = ["capital", "scoring", "provisions"]

# --- Section "First phase", p. 11: mentions per document of the statistical terms.
MENTIONS_PER_DOC = {  # p. 11
    "stability": 6.4,
    "hyperparameters": 0.2,
    "overfitting": 0.16,
    "dynamic calibration": 0.02,
    "feature engineering": 0.33,
}

# --- Dictionary sizes, p. 13 (the terms themselves are Table 11, p. 30).
DICTIONARY_SIZE = {"statistics": 54, "technology": 30, "conduct": 51}  # p. 13

# --- Table 2, p. 15: overall relative frequency H̄_ij, in percent of all words.
TABLE2 = {  # p. 15
    "capital": {"statistics": 7.90, "technology": 2.50, "conduct": 3.75, "total": 14.15},
    "scoring": {"statistics": 6.41, "technology": 3.09, "conduct": 5.13, "total": 14.63},
    "provisions": {"statistics": 6.38, "technology": 2.36, "conduct": 3.90, "total": 12.64},
}

# --- Table 3, p. 16: p-values of the t-tests between uses ("bold" = significant at 10%).
TABLE3 = {  # p. 16
    ("capital", "scoring"): {"statistics": 0.0181, "technology": 0.1897, "conduct": 0.003},
    ("capital", "provisions"): {"statistics": 0.0133, "technology": 0.0180, "conduct": 0.5618},
    ("scoring", "provisions"): {"statistics": 0.9404, "technology": 0.1097, "conduct": 0.0128},
}
TABLE3_BOLD = {  # p. 16, which cells the paper marks as significant
    ("capital", "scoring"): {"statistics", "conduct"},
    ("capital", "provisions"): {"statistics", "technology"},
    ("scoring", "provisions"): {"technology", "conduct"},
}

# --- Table 4 and the worked example, pp. 16-17.
TABLE4_Z = {"statistics": 2, "technology": 4, "conduct": 3}  # p. 17
TABLE4_RISKINESS = {"capital": 0.523, "scoring": 0.554, "provisions": 0.536}  # p. 16

# --- Empirical example, p. 17.
DATA = {"loans": 120_000, "characteristics": 11, "features": 22,
        "train_share": 0.80, "cv_folds": 5}  # p. 17
HYPERPARAMETERS = {  # p. 17
    "CART": {"max_depth": 7},
    "RF": {"max_depth": 20, "n_estimators": 100},
    "XGBoost": {"max_depth": 5, "n_estimators": 40},
    "MLP": {"hidden_layer_sizes": (300, 200, 100), "activation": "relu", "solver": "adam"},
}
STABILITY_RETRAINS = 100  # p. 12 and p. 18
STABILITY_SAMPLE_RANGE = (0.60, 1.00)  # p. 12: "between 60 and 100% of the training set"

# --- Table 5, p. 18: phase-1 scorecard.
TABLE5 = {  # p. 18
    "stability_sd": {"Lasso": 0.001, "CART": 0.005, "RF": 0.003, "XGBoost": 0.001, "MLP": 0.002},
    "n_hyperparameters": {"Lasso": 0, "CART": 1, "RF": 2, "XGBoost": 2, "MLP": 5},  # MLP printed "+ 5"
    "z_statistics": {"Lasso": 1, "CART": 3, "RF": 3, "XGBoost": 2, "MLP": 5},
    "training_seconds": {"Lasso": 3.92, "CART": 0.67, "RF": 19.25, "XGBoost": 1.29, "MLP": 31.42},
    "transparency": {"Lasso": 1, "CART": 2, "RF": 3, "XGBoost": 3, "MLP": 5},
    "z_technology": {"Lasso": 1, "CART": 1, "RF": 3, "XGBoost": 2, "MLP": 5},
    "shap_seconds": {"Lasso": 0, "CART": 0, "RF": 60, "XGBoost": 8, "MLP": 2000},
    "z_conduct": {"Lasso": 1, "CART": 1, "RF": 2, "XGBoost": 2, "MLP": 3},
}

# --- Table 6, p. 20: Z used in the phase-2 table and the resulting risk (sum of H̄·Z, H̄ in %).
TABLE6_Z = {  # p. 20 (note: MLP statistics is 4 here, 5 in Table 5)
    "statistics": {"Lasso": 1, "CART": 3, "RF": 3, "XGBoost": 2, "MLP": 4},
    "technology": {"Lasso": 1, "CART": 1, "RF": 3, "XGBoost": 2, "MLP": 5},
    "conduct": {"Lasso": 1, "CART": 1, "RF": 2, "XGBoost": 2, "MLP": 3},
}
TABLE6_RISK = {  # p. 20
    "capital": {"Lasso": 14.15, "CART": 29.95, "RF": 38.7, "XGBoost": 28.3, "MLP": 55.35},
    "scoring": {"Lasso": 14.63, "CART": 27.45, "RF": 38.76, "XGBoost": 29.26, "MLP": 56.48},
    "provisions": {"Lasso": 12.64, "CART": 25.4, "RF": 34.02, "XGBoost": 25.28, "MLP": 49.02},
}

# --- Table 7, p. 21: AUC-ROC on the test set, percent.
TABLE7_AUC = {"Logit": 80.0, "Lasso": 80.2, "CART": 80.4, "RF": 84.2, "XGBoost": 85.3, "MLP": 81.7}  # p. 21
TABLE7_CI = {  # p. 21, 95% interval
    "Logit": (79.3, 80.7), "Lasso": (79.6, 80.8), "CART": (79.7, 81.2),
    "RF": (83.7, 84.8), "XGBoost": (84.8, 85.8), "MLP": (81.0, 82.3),
}

# --- Text claims, each with its page.
CLAIMS = {
    "lit_max_gain_pct": 20,  # p. 4: "reaching up to 20%"
    "xgb_gain_pct": 5,  # p. 20: "approximately 5% in the AUC-ROC"
    "rf_gain_pct": 4,  # p. 20
    "mlp_gain_pct": 1.7,  # p. 20
    "most_unstable": ("MLP", "CART"),  # p. 18: highest standard deviations
    "most_stable": ("Lasso", "XGBoost"),  # p. 18
    "best_risk_adjusted": "XGBoost",  # p. 21
    "lowest_risk_use": "provisions",  # p. 20
    "shap_top3_all": ("NumberOfTimes90DaysLate", "NumberOfTime30-59DaysPastDueNotWorse"),  # p. 26
    "shap_top_xgb": "RevolvingUtilizationOfUnsecuredLines",  # p. 26
    "shap_age_rank": 4,  # p. 26: age is always fourth
}

# --- Sentiment robustness, Tables 8-10, p. 28: negative words around risk terms.
TABLE8_D10 = {  # p. 28
    "capital": {"statistics": 0.79, "technology": 0.59, "conduct": 0.78},
    "scoring": {"statistics": 0.93, "technology": 1.06, "conduct": 0.95},
    "provisions": {"statistics": 0.89, "technology": 0.42, "conduct": 0.75},
}
TABLE9_D20 = {  # p. 28
    "capital": {"statistics": 1.53, "technology": 1.17, "conduct": 1.46},
    "scoring": {"statistics": 1.85, "technology": 2.04, "conduct": 1.76},
    "provisions": {"statistics": 1.76, "technology": 0.96, "conduct": 1.38},
}
TABLE10_NEG_SHARE = {"capital": 3.87, "scoring": 3.09, "provisions": 4.88}  # p. 28, percent
LEXICON_SIZE = 4780  # p. 27, Hu and Liu (2004) negative words
