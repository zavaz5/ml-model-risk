"""Write docs/speaker_script.pdf: cue cards for each slide, the formulas, and likely questions with an owner.

Graders of earlier talks asked speakers not to read: the notes are short cues, to be said in one's own words.
They also marked teams down when one student answered every question, so each question names the speaker
whose slides it is about (Speaker 1-5; the team fills in names).

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
LIMIT_MIN = 15.0  # the seminar's limit (lecture 1): 15 minutes of talk, then 10 minutes of questions

CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.35; color: #222; }
h1 { font-size: 17pt; color: #0F2C68; margin-bottom: 4pt; }
h2 { font-size: 12.5pt; color: #0F2C68; margin-top: 12pt; margin-bottom: 3pt; }
p { margin-top: 2pt; margin-bottom: 4pt; }
.meta { color: #595959; font-size: 9.5pt; }
li { margin-bottom: 3pt; }
.owner { color: #1656A6; font-weight: bold; }
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


OWNERS = {  # question → speaker whose slides it is about (see the cue cards)
    "features": 2, "120,000": 2, "Lasso almost": 2, "DeLong": 2, "test set": 5, "training times": 2, "SHAP ranking": 3,
    "word frequency": 4, "errata": 4, "risk-efficient": 3, "logistic regression": 4, "CART": 2, "three different": 3,
    "risk factors": 5, "recommend": 1, "symbols": 1, "objective": 2, "settings": 2, "improvement": 5, "out of sample": 5,
    "placebo": 4, "dimensionality": 2,
}


def owner(question: str) -> int:
    return next(v for key, v in OWNERS.items() if key.lower() in question.lower())


def new_questions(k: dict) -> list[tuple[str, str]]:
    """Questions graders asked other teams this year, answered for our paper."""
    a, meta, imp = k["models"]["auc"], k["models"]["meta"], k["improvements"]
    cal = imp["I12"]["table"]["XGBoost"]
    gaps = k["appendix"]["factors"]
    return [
        ("What do the symbols in the formula mean?",
         "m is the model, j the use (capital, credit scoring, provisions), i the risk category (statistics, technology, conduct). "
         "Z is the darkness score of model m in category i, from 1 (transparent) to 5 (black box). H-bar is the weight of "
         "category i for use j: the share of the category's terms among all words of that use's regulatory texts, in percent."),
        ("What is the training objective, and how does it differ from the evaluation and calibration objectives?",
         "Training: log-loss for the logit, the Lasso (plus an L1 penalty), XGBoost and the MLP; Gini impurity at each split "
         "for CART and RF. Evaluation in the paper: AUC, which only measures how well defaults are ranked. Calibration asks "
         "whether a predicted 5% PD really defaults 5% of the time, which capital and provisions need; we measure it with the "
         f"Brier score and the calibration slope (XGBoost: Brier {cal['brier']:.4f}, slope {cal['slope']:.2f})."),
        ("Why did you choose those settings?",
         "They are the paper's own (p. 17), chosen by the authors with five-fold cross-validation. We keep them because the "
         "task is to reproduce; re-tuning would hide where the paper differs. Our only choice is the Lasso penalty, by "
         f"cross-validated AUC on training rows (C = {meta['lasso_c']:g}). Improvements make their choices on validation rows."),
        ("What exactly is an improvement, and why is it one?",
         "Each one fixes a weakness of a specific result. Examples: the paper reports AUCs without testing the differences, so "
         "we add a paired DeLong test; its expert scores cannot be recomputed, so we score by a fixed rule; its SHAP ranks come "
         "from one fit, so we refit with five seeds; its literature figure mixes datasets and metrics, so we run one protocol "
         "on three datasets."),
        ("How did you test out of sample?",
         f"All reported AUCs are on {meta['test']:,} test rows never used for fitting or choosing; our own choices use "
         "validation rows inside the training data. Out of dataset: the same protocol on Taiwan and German credit data. Out of "
         "time is impossible here: the Kaggle data have no dates, so it is our first next step."),
        ("What does the placebo show?",
         "Replacing the weights from the regulatory texts by random ones never moves XGBoost off the frontier, because its "
         "scores are never above RF's or the MLP's and its gain is the largest. The text phase can only decide whether CART "
         "is also efficient. Exploratory, not pre-registered."),
        ("Is dimensionality a problem here?",
         f"There are 20 inputs, but the models' sizes differ hugely: {a['Lasso']['params']:,} fitted parameters for the Lasso, "
         f"{a['RF']['params']:,} for RF. RF fits its training rows {gaps['RF']['overfit_gap']:+.2f} AUC better than new rows: it "
         "memorises, which is why every number we report is on unseen rows."),
    ]


def questions(k: dict) -> list[tuple[str, str]]:
    m, meta = k["models"], k["models"]["meta"]
    a = m["auc"]
    err = {e["id"]: e["what"] for e in k["errata"]}
    return new_questions(k) + [
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
        ("Why is your CART four points better than the paper's?",
         "We fit exactly the tree the paper describes, depth 7. We then tried every other reading of the set-up on validation rows: class weights, entropy, the row index as a feature and "
         "hard-label scoring all leave a depth-7 tree far above 80.4. Only a small pruned tree (8-12 leaves, what R's rpart "
         "builds by default) or a depth-10 tree reaches it. The other four models match the paper, so the data and split are "
         "comparable; the paper's tree was most likely not the depth-7 tree it reports."),
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
    parts = ["<h1>Speaker cue cards</h1>",
             f"<p class='meta'>Alonso Robisco and Carbó Martínez (2022). Group 8. {len(slides)} main slides, "
             f"{total:g} minutes of talk, then 10 minutes of questions. Slide numbers as in docs/presentation.pptx.</p>",
             "<p><b>How to use these cards.</b> Say each cue in your own words; do not read the slides or this page. The slides "
             "carry the numbers, so point at them instead of reciting them. Rehearse once with a timer. Speakers 1–5 split the "
             "talk as marked; each answers the questions about their own slides (listed at the end), so all five take part.</p>"]
    for number, title, notes in slides:
        header, *lines = notes.split("\n")
        items = "".join(f"<li>{html.escape(x.lstrip('• '))}</li>" for x in lines)
        parts.append(f"<h2>Slide {number}. {html.escape(title)}</h2><p class='meta'>{html.escape(header)}</p><ul>{items}</ul>")
    parts.append("<h2>Formulas, in words</h2><ul>")
    parts += [f"<li><b>{html.escape(t)}.</b> {f}</li>" for t, f in FORMULAS]
    parts.append("</ul><h2>Likely questions, and who answers</h2><ul>")
    parts += [f"<li><span class='owner'>Speaker {owner(q)}.</span> <b>{html.escape(q)}</b> {html.escape(ans)}</li>"
              for q, ans in questions(k)]
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
