"""Hand-verified checks for Boolean retrieval and lnc.ltc VSM."""
import math
import os
import tempfile

from submission.indexer import InvertedIndex
from submission import boolean_vsm

CORPUS = [
    ("d1", "car insurance policy"),
    ("d2", "auto insurance claim"),
    ("d3", "car car engine"),
]


def _index():
    ix = InvertedIndex()
    ix.build(CORPUS)
    with tempfile.TemporaryDirectory() as td:
        ix.save(td)
        ix2 = InvertedIndex.load(td)
    ix2.decode_all()
    boolean_vsm.build(ix2)
    return ix2


def test_boolean_and():
    _index()
    assert boolean_vsm.boolean_search("car insurance", mode="and") == ["d1"]


def test_boolean_or():
    _index()
    got = boolean_vsm.boolean_search("engine claim", mode="or")
    assert set(got) == {"d2", "d3"}


def test_boolean_missing_term():
    _index()
    assert boolean_vsm.boolean_search("car zebra", mode="and") == []


def test_vsm_ranks_repeated_term_higher():
    _index()
    got = boolean_vsm.vsm_score("car", 10)
    # d3 has tf=2 for "car", d1 has tf=1; both normalised by their norms.
    assert got[0][0] == "d3"
    assert [d for d, _ in got] == ["d3", "d1"]


def test_vsm_cosine_value_matches_hand_calculation():
    ix = _index()
    N = 3
    # query "car": tf=1, df("car")=2
    wq = (1.0 + math.log10(1)) * math.log10(N / 2)      # = log10(1.5)
    # d3: tf("car")=2 -> wd = 1 + log10(2)
    wd = 1.0 + math.log10(2)
    # ||d3||: terms are car(tf=2), engin(tf=1)
    norm_d3 = math.sqrt((1 + math.log10(2)) ** 2 + 1.0 ** 2)
    expected = (wq * wd) / (norm_d3 * math.sqrt(wq * wq))

    got = dict(boolean_vsm.vsm_score("car", 10))
    assert abs(got["d3"] - expected) < 1e-9