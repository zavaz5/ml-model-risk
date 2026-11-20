"""Run the project with one command, then check that its parts agree.

    python scripts/run_all.py            # quick, a few seconds: tests, then every figure, the README results,
                                         # the slides and the cue cards rebuilt from the committed results/
    python scripts/run_all.py --full     # everything from the raw data (downloads first; about 45 minutes)

Quick mode needs no data: results/ is committed, so the instructor can rebuild every output and inspect it
without Kaggle access. Full mode needs ~/.kaggle/kaggle.json and the Give Me Some Credit rules accepted.
The PDF of the deck is exported separately (docs/HANDOFF.md), because it needs Keynote or PowerPoint.

The consistency checks at the end recompute a few of the paper's numbers from other numbers in the paper
and compare our stored results with each other, so a broken step shows up as FAILED rather than as a
quietly wrong slide.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mrisk import paper_numbers as P  # noqa: E402

FULL = [
    "download_data.py", "download_corpus.py", "extract_paper.py", "reproduce_models.py", "reproduce_nlp.py",
    "extract_fig9_fig4.py", "reproduce_scorecard.py", "reproduce_extra.py", "diagnose_cart.py", "diagnose_nlp_levels.py",
    "improve_nlp.py", "improve_models.py", "improve_risk.py", "appendix_factors.py", "extended_scorecard.py",
    "explore_logit_benchmark.py", "reproduce_shap_figures.py",
]
OUTPUTS = ["explore_placebo_weights.py", "make_paper_figures.py", "make_report.py", "make_slides.py", "make_speaker_script.py"]


def step(args: list[str]) -> bool:
    print(f"\n=== {' '.join(args)}", flush=True)
    t0 = time.time()
    ok = subprocess.run([sys.executable, *args], cwd=ROOT).returncode == 0
    print(f"--- {'ok' if ok else 'FAILED'} ({time.time() - t0:.0f} s)", flush=True)
    return ok


def checks() -> bool:
    """Each check prints one line and returns whether it passed."""
    k = json.loads((ROOT / "results" / "report" / "key_numbers.json").read_text())
    placebo = json.loads((ROOT / "results" / "improvements" / "appendix" / "placebo_weights.json").read_text())
    out = []
    # The paper's Table 6 from its own Table 2 weights and Table 6 scores.
    worst = max(abs(sum(P.TABLE2[u][c] * P.TABLE6_Z[c][m] for c in P.CATEGORIES) - P.TABLE6_RISK[u][m])
                for u in P.USES for m in P.MODELS)
    out.append(("paper's Table 6 = Table 2 weights × Table 6 scores", worst < 0.01, f"largest difference {worst:.3f}"))
    # The paper's Table 4 from its own weights, truncated to three decimals as the paper does.
    ex = k["scorecard_checks"]["table4_example"]
    same = all(int(1000 * ex[u]) / 1000 == P.TABLE4_RISKINESS[u] for u in P.USES)
    out.append(("paper's Table 4 recomputed from its weights", same, " / ".join(f"{ex[u]:.4f}" for u in P.USES)))
    # Our AUCs against the paper's: the paper's value inside our 95% interval.
    auc = k["models"]["auc"]
    outside = [m for m in auc if not 100 * auc[m]["lo"] <= P.TABLE7_AUC[m] <= 100 * auc[m]["hi"]]
    out.append(("paper's AUC inside our 95% interval for all but CART, as the slides say", outside == ["CART"],
                f"outside: {', '.join(outside) or 'none'}"))
    # Two independent routes to the same conclusion: the I13 Monte Carlo and the placebo both keep XGBoost on the frontier.
    mc = min(r["p_frontier"] for r in k["improvements"]["I13"]["table"] if r["model"] == "XGBoost")
    pl = placebo["results"]["paper"]["random_weights_share_on_frontier"]["XGBoost"]
    out.append(("XGBoost on the frontier: Monte Carlo and placebo", mc == 1 and pl == 1, f"{mc:.2f} and {pl:.2f}"))
    print("\n=== consistency checks")
    for name, ok, detail in out:
        print(f"{'ok    ' if ok else 'FAILED'} {name}: {detail}")
    return all(ok for _, ok, _ in out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="rerun everything from the raw data")
    full = ap.parse_args().full
    ok = step(["-m", "pytest", "-q"])
    for script in (FULL if full else []) + OUTPUTS:
        ok = step([f"scripts/{script}"]) and ok
        if not ok:
            sys.exit(f"stopped at {script}")
    ok = checks() and ok
    print("\nall steps and checks passed" if ok else "\nsome checks FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
