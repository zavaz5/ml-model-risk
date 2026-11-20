import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("improve_risk", ROOT / "scripts" / "improve_risk.py")
risk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk)


def test_minmax_maps_to_one_and_five():
    z = risk.minmax(np.array([1.0, 10.0, 100.0]), log=True)
    assert z.tolist() == pytest.approx([1.0, 3.0, 5.0])
    assert risk.minmax(np.array([2.0, 2.0]), log=False).tolist() == [3.0, 3.0]


def test_frontier_drops_dominated_models():
    gain = np.array([0.0, 5.0, 4.0, 1.0])
    rsk = np.array([0.2, 0.4, 0.5, 0.6])
    assert risk.frontier(gain, rsk).tolist() == [True, True, False, False]


def test_normalised_risk_is_one_for_the_darkest_model():
    z = {"statistics": np.array([5.0, 1.0]), "technology": np.array([5.0, 1.0]), "conduct": np.array([5.0, 1.0])}
    w = np.array([[0.08, 0.03, 0.04]])
    assert np.allclose(risk.normalised_risk(z, w), [[1.0, 0.2]])


def test_paper_table6_arithmetic():
    from mrisk import paper_numbers as P

    for use in P.USES:
        for m in P.MODELS:
            value = sum(P.TABLE2[use][c] * P.TABLE6_Z[c][m] for c in P.CATEGORIES)
            assert value == pytest.approx(P.TABLE6_RISK[use][m], abs=1e-6)


_spec = importlib.util.spec_from_file_location("placebo", ROOT / "scripts" / "explore_placebo_weights.py")
placebo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(placebo)


def test_placebo_frontier_keeps_ties_and_drops_dominated():
    risk_ = np.array([1.0, 2.0, 2.0, 3.0])
    gain = np.array([0.0, 4.0, 3.0, 4.0])  # model 2 is beaten by 1 (same risk, more gain); model 3 by 1 (more risk)
    assert placebo.on_frontier(risk_, gain).tolist() == [True, True, False, False]


def test_dominance_needs_no_higher_score_in_every_category():
    z = np.array([[2.0, 3.0, 1.0], [2.0, 3.0, 5.0]])  # categories × models
    gain = np.array([5.0, 4.0, 1.0])
    assert placebo.dominates_everywhere(z, gain, 0, 1)      # 0 ≤ 1 in both categories and more gain
    assert not placebo.dominates_everywhere(z, gain, 0, 2)  # model 2 has a lower score in the first category
