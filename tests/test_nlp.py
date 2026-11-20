import numpy as np
import pandas as pd

from mrisk.nlp import compile_terms, count_document, category_share, lemma, negatives_around, tokens


def test_lemma_plurals():
    assert lemma("models") == "model"
    assert lemma("biases") == "bias"
    assert lemma("bias") == "bias"
    assert lemma("dependencies") == "dependency"
    assert lemma("boxes") == "box"
    assert lemma("losses") == "loss"
    assert lemma("training") == "training"  # verbs and gerunds are left alone


def test_tokens_join_line_break_hyphen():
    assert tokens("over-\nfitting is bad") == ["overfitting", "is", "bad"]


def test_terms_match_variants_and_count_independently():
    dictionary = pd.DataFrame({"category": ["statistics", "statistics", "technology"],
                               "term": ["Back-testing", "Model", "Model stealing"]})
    terms = compile_terms(dictionary)
    text = "Backtesting and back-testing of models. Model stealing is a threat."
    c = count_document("d", text, terms, {"threat"})
    assert c.term_counts["statistics|Back-testing"] == 2
    assert c.term_counts["statistics|Model"] == 2  # "models" and the "model" in "model stealing"
    assert c.term_counts["technology|Model stealing"] == 1
    assert np.isclose(category_share(c, "statistics"), 4 / c.n_tokens)


def test_negative_window_excludes_the_term_itself():
    dictionary = pd.DataFrame({"category": ["conduct"], "term": ["bias"]})
    terms = compile_terms(dictionary)
    c = count_document("d", "bad bias here and later bad", terms, {"bad", "bias"})
    total, mentions = negatives_around(c, "conduct", d=2)
    assert mentions == 1 and total == 1  # only the first "bad" is within 2 words
