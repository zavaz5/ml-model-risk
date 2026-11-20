"""Load the credit datasets and build the paper's design matrix and splits.

The paper (p. 17) uses Kaggle's "Give Me Some Credit": 120,000 loans, a
binary default flag, the borrower characteristics plus their squares, and an
80/20 train-test split. The public file has 150,000 rows; dropping the rows
with a missing value leaves 120,269, which matches the paper's count, so we
use complete cases (DEVIATIONS.md, D1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

GMSC_TARGET = "SeriousDlqin2yrs"
GMSC_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
]
SPLIT_SEED = 0  # the paper gives no seed; we fix one and never change it


@dataclass(frozen=True)
class Split:
    """Row indices into the full design matrix.

    ``train`` is the paper's 80% (used for every reproduction fit). For our
    improvements, ``fit`` and ``valid`` partition ``train`` (80/20) so that
    design choices never look at ``test``.
    """
    train: np.ndarray
    test: np.ndarray
    fit: np.ndarray
    valid: np.ndarray


def load_gmsc(path: Path | None = None) -> pd.DataFrame:
    """Read cs-training.csv, drop its unnamed row-index column and incomplete rows."""
    frame = pd.read_csv(path or RAW / "gmsc" / "cs-training.csv")
    frame = frame.drop(columns=[c for c in frame.columns if c.startswith("Unnamed")])
    return frame.dropna(subset=GMSC_FEATURES + [GMSC_TARGET]).reset_index(drop=True)


def add_squares(features: pd.DataFrame) -> pd.DataFrame:
    """Append x² for every column, as the paper does to capture non-linearity (p. 17)."""
    squares = features.pow(2).add_suffix("_sq")
    return pd.concat([features, squares], axis=1)


def base_name(column: str) -> str:
    """Map a design-matrix column back to the borrower characteristic it came from."""
    return column[:-3] if column.endswith("_sq") else column


def design_matrix(frame: pd.DataFrame, features: list[str], target: str,
                  squares: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    x = frame[features].astype(float)
    return (add_squares(x) if squares else x), frame[target].to_numpy(dtype=int)


def make_split(y: np.ndarray, seed: int = SPLIT_SEED, test_size: float = 0.20,
               valid_size: float = 0.20) -> Split:
    """Stratified 80/20 train-test split, then a stratified 80/20 fit-validation split of train."""
    index = np.arange(len(y))
    train, test = train_test_split(index, test_size=test_size, stratify=y, random_state=seed)
    fit, valid = train_test_split(train, test_size=valid_size, stratify=y[train], random_state=seed)
    return Split(np.sort(train), np.sort(test), np.sort(fit), np.sort(valid))


def load_benchmark(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Two further public credit datasets for the multi-dataset benchmark (numeric design, 1 = default)."""
    frame = pd.read_csv(RAW / "openml" / f"{name}.csv")
    if name == "taiwan_default":
        y = frame.pop("y").astype(int).to_numpy()
        return frame.astype(float), y
    if name == "german_credit":
        y = (frame.pop("class") == "bad").astype(int).to_numpy()
        return pd.get_dummies(frame, drop_first=True, dtype=float), y
    raise ValueError(name)
