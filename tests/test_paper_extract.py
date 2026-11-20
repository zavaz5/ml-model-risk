"""Regression tests for the values we read off the paper's PDF (committed under paper/, CC BY 4.0)."""
import importlib.util
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = pymupdf.open(ROOT / "paper" / "alonso_carbo_2022.pdf")


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_table11_term_counts():
    terms = load("extract_paper").table11(DOC).drop_duplicates()
    assert terms.groupby("category").size().to_dict() == {"conduct": 51, "statistics": 53, "technology": 30}


def test_fig5_random_forest_is_drawn_with_statistics_score_2():
    fig5 = load("extract_paper").fig5(DOC)
    rf = fig5[fig5["model"] == "RF"].set_index("use")["risk"]
    # Table 2 weights with Z = (2, 3, 2): capital 0.308, scoring 0.3235, provisions 0.2764
    assert rf.to_dict() == pytest.approx({"capital": 0.308, "scoring": 0.3235, "provisions": 0.2764}, abs=0.002)


def test_fig4_radar_scores():
    scores = load("extract_fig9_fig4").fig4(DOC).set_index("model").round()
    assert scores.loc["RF", "statistics"] == 2 and scores.loc["MLP", "statistics"] == 4
    assert scores.loc["MLP", "technology"] == 5 and scores.loc["Lasso"].tolist() == [1, 1, 1]


def test_fig9_has_one_unlabelled_series_in_the_common_area():
    f9 = load("extract_fig9_fig4").fig9(DOC)
    assert (f9["markers_found"] == 1).all()
    unlabelled = f9[f9["doc_id"] == "unlabelled"]
    assert set(unlabelled["panel"]) == {"common"} and len(unlabelled) == 3
