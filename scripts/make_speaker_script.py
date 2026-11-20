"""Write docs/speaker_script.pdf: what to say on each slide, the formulas, and likely questions.

Slide numbers, titles and talking points are read from docs/presentation.pptx
(run make_slides.py first); numbers come from results/report/key_numbers.json.
The script fails if a main slide has no notes or the timings exceed 15 minutes.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pymupdf
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "presentation.pptx"
OUT = ROOT / "docs" / "speaker_script.pdf"
LIMIT_MIN = 15.0

CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.35; color: #222; }
h1 { font-size: 17pt; color: #0F2C68; margin-bottom: 4pt; }
h2 { font-size: 12.5pt; color: #0F2C68; margin-top: 12pt; margin-bottom: 3pt; }
p { margin-top: 2pt; margin-bottom: 4pt; }
.meta { color: #595959; font-size: 9.5pt; }
li { margin-bottom: 3pt; }
"""


def minutes(note: str) -> float:
    """Talk time written in a slide's notes as '~2 min' or '~1.5 min'."""
    found = re.search(r"~(\d+(?:\.\d+)?) min", note)
    return float(found.group(1)) if found else 0.0


def main_slides(prs) -> list[tuple[int, str, str]]:
    out = []
    for number, slide in enumerate(prs.slides, start=1):
        if slide.slide_layout.name == "Final_1":
            break
        title = slide.shapes.title.text if slide.shapes.title is not None else "Title slide"
        notes = slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ""
        if not notes.strip():
            raise SystemExit(f"slide {number} ({title}) has no speaker notes")
        out.append((number, title or "Title slide", notes))
    return out


def questions(k: dict) -> list[tuple[str, str]]:
    m, meta = k["models"], k["models"]["meta"]
    a = m["auc"]
    err = {e["id"]: e["what"] for e in k["errata"]}
    return [
        ("Why do you have 20 features when the paper has 22?",
         "The file has 10 borrower characteristics; with squares that is 20. Its eleventh column is an unnamed row index, which "
         "the paper probably counted as a characteristic. The paper's SHAP figures show 10 features, which supports this."),
        ("How did you get the paper's 120,000 loans?",
         f"cs-training.csv has {meta['rows_raw']:,} rows; dropping rows with a missing value leaves {meta['rows']:,}. "
         "No other rule we tried gives the paper's count."),
        ("Why is your Lasso almost the logit?",
         f"Five-fold cross-validated AUC chose C = {meta['lasso_c']:g}, the weakest penalty on the grid: with 96,000 rows and 20 "
         "features there is nothing to shrink. The paper's Lasso is also only 0.2 points above its logit."),
        ("What is the DeLong test?",
         "An exact variance for the empirical AUC built from each default's and non-default's placement among the other "
         "class's scores. Because both models are scored on the same rows, it gives a paired test of the AUC difference."),
        ("Is the test set used to choose anything?",
         "No. The reproduction uses the paper's settings and reports on the test rows, as the paper does. Every improvement was "
         "pre-registered and makes its choices on validation rows (20% of training)."),
        ("Why are your training times different?",
         "Times depend on hardware, threads and solver. We time single-threaded, median of five fits after a warm-up. The Lasso "
         "needs the saga solver: liblinear took over 90 seconds per fit on these heavy-tailed squared features."),
        ("Why does the SHAP ranking of XGBoost differ from the others in the paper?",
         "Its plot is in log-odds while RF and the MLP are in probabilities (the axes show it). Log-odds stretch the effect of "
         "features in low-risk regions. Improvement I11 puts every model on the probability scale with one explainer."),
        ("Is word frequency a fair measure of what supervisors care about?",
         "It is the paper's assumption. Our audit shows a few generic terms (for example 'model' and 'test') make up much of a "
         "category's count; the bootstrap shows how much the weights move when the document set changes."),
        ("What are the errata?",
         " ".join(f"{i}: {w}." for i, w in err.items())),
        ("What does 'risk-efficient' mean in your Monte Carlo?",
         "A model is on the frontier if no other model has both a larger AUC gain and a lower risk. We count how often each "
         "model is on the frontier when every noisy input is redrawn."),
        ("Is machine learning really better than a logistic regression here?",
         next((r["finding"] for r in k["improvement_rows"] if r["id"] == "X1"),
              "See the exploratory logit benchmark in the README.")
         + ". This was exploratory, not pre-registered: the paper's logit gets raw, heavy-tailed ratios; a credit scorer would transform them."),
        ("Why does the paper have three different statistics scores?",
         "Table 5 gives RF 3 and the MLP 5; Table 6 uses RF 3 and MLP 4; Figs. 4 and 5 are drawn with RF 2 and MLP 4. We read "
         "Figs. 4 and 5 off the vector PDF, so this is exact, not eyeballed. No ranking changes."),
        ("What about the risk factors the paper does not measure?",
         "Its appendix proposes measures; we computed five on validation rows (overfitting, inputs used, preprocessing "
         "sensitivity, surrogate fidelity, age counterfactual). " + next((r["finding"] for r in k["improvement_rows"] if r["id"] == "A1–A5"), "")
         + ". Third-party, cyber and privacy risk need a bank's own data."),
        ("Which model would you recommend to a bank?",
         f"On these data XGBoost: the highest AUC ({100 * a['XGBoost']['auc']:.1f}%) with moderate risk scores. But the "
         "ranking is not an approval: the supervisor's own trade-off is unknown."),
    ]


FORMULAS = [
    ("Share of risk terms in a document", "H<sub>ijk</sub> = W<sub>ijk</sub> / N<sub>k</sub>: occurrences of category i's terms "
     "in document k of use j, divided by the document's word count."),
    ("Weight of a category for a use", "H̄<sub>ij</sub> = mean of H<sub>ijk</sub> over use j's documents + mean of H over the "
     "common-area documents."),
    ("Supervisory risk", "Risk<sub>mj</sub> = Σ<sub>i</sub> H̄<sub>ij</sub> Z<sub>im</sub>, with Z from 1 (clear) to 5 (dark); "
     "normalised by 5 Σ<sub>i</sub> H̄<sub>ij</sub> it lies between 0.2 and 1."),
    ("Welch t-test", "t = (x̄<sub>1</sub> − x̄<sub>2</sub>) / √(s<sub>1</sub>²/n<sub>1</sub> + s<sub>2</sub>²/n<sub>2</sub>), with "
     "Welch–Satterthwaite degrees of freedom."),
    ("Negative-word lift", "negatives within d words of a mention, per mention, divided by 2d × the share of negative words "
     "in that use's documents: 1 means no more negative than a random spot in the text."),
]


def main() -> None:
    k = json.loads((ROOT / "results" / "report" / "key_numbers.json").read_text())
    slides = main_slides(Presentation(DECK))
    total = sum(minutes(n) for _, _, n in slides)
    if total > LIMIT_MIN:
        raise SystemExit(f"talk is {total} minutes, over {LIMIT_MIN}")
    parts = ["<h1>Speaker script</h1>",
             f"<p class='meta'>Alonso Robisco and Carbó Martínez (2022). Group 8. {len(slides)} main slides, "
             f"about {total:g} minutes of talk, then 10 minutes of questions. Slide numbers as in docs/presentation.pptx.</p>"]
    for number, title, notes in slides:
        parts.append(f"<h2>Slide {number}. {html.escape(title)}</h2><p>{html.escape(notes)}</p>")
    parts.append("<h2>Formulas, in words</h2><ul>")
    parts += [f"<li><b>{html.escape(t)}.</b> {f}</li>" for t, f in FORMULAS]
    parts.append("</ul><h2>Likely questions</h2><ul>")
    parts += [f"<li><b>{html.escape(q)}</b> {html.escape(ans)}</li>" for q, ans in questions(k)]
    parts.append("</ul>")
    story = pymupdf.Story(html="".join(parts), user_css=CSS)
    writer = pymupdf.DocumentWriter(str(OUT))
    rect, where = pymupdf.paper_rect("a4"), pymupdf.paper_rect("a4") + (50, 50, -50, -50)
    more = True
    while more:
        dev = writer.begin_page(rect)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    print(f"saved {OUT} ({total:g} min of talk)")


if __name__ == "__main__":
    main()
