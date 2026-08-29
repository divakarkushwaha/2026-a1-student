"""
submission/bm25.py — BM25 (Robertson & Walker 1992; Robertson &
Zaragoza 2009).

    score(q,d) = SUM_t  IDF(t) * tf * (k1+1) / (tf + k1*(1-b + b*dl/avgdl))
    IDF(t)     = ln(1 + (N - df + 0.5) / (df + 0.5))

k1 controls term-frequency saturation: larger k1 means repeated
occurrences keep adding score. b controls length normalisation: b=0
ignores document length entirely, b=1 normalises fully.

The length-normalisation denominator term k1*(1-b+b*dl/avgdl) is
query-independent, so it is precomputed per document at load time.
"""
import math

import numpy as np

from submission.text import analyze

_index = None
_idf = None
_norm = None
_k1 = 1.2


def build(index, k1: float = 1.2, b: float = 0.75) -> None:
    global _index, _idf, _norm, _k1
    _index = index
    _k1 = k1
    N = index.N
    _idf = np.array(
        [math.log(1.0 + (N - df + 0.5) / (df + 0.5)) for df in index.df],
        dtype=np.float64,
    )
    _norm = (k1 * (1.0 - b + b * (index.doc_lens.astype(np.float64) / index.avgdl)))


def score(query: str, k: int = 10):
    N = _index.N
    acc = np.zeros(N, dtype=np.float64)
    hit = False

    for term in analyze(query):
        i = _index.term_ids.get(term)
        if i is None:
            continue
        p = _index.get_postings(term)
        if p is None:
            continue
        docs, tfs = p
        tfs = tfs.astype(np.float64)
        contrib = _idf[i] * (tfs * (_k1 + 1.0)) / (tfs + _norm[docs])
        np.add.at(acc, docs, contrib)
        hit = True

    if not hit:
        return []

    n = min(k, N)
    cand = np.argpartition(-acc, n - 1)[:n]
    cand = cand[acc[cand] > 0]
    cand = cand[np.argsort(-acc[cand], kind="stable")]
    return [(_index.doc_ids[int(d)], float(acc[d])) for d in cand]


def score_internal(query, k):
    """Like score(), but returns internal doc ids — for feedback."""
    N = _index.N
    acc = np.zeros(N, dtype=np.float64)
    hit = False
    for term in analyze(query):
        i = _index.term_ids.get(term)
        if i is None:
            continue
        p = _index.get_postings(term)
        if p is None:
            continue
        docs, tfs = p
        tfs = tfs.astype(np.float64)
        contrib = _idf[i] * (tfs * (_k1 + 1.0)) / (tfs + _norm[docs])
        np.add.at(acc, docs, contrib)
        hit = True
    if not hit:
        return []
    n = min(k, N)
    cand = np.argpartition(-acc, n - 1)[:n]
    cand = cand[acc[cand] > 0]
    cand = cand[np.argsort(-acc[cand], kind="stable")]
    return [(int(d), float(acc[d])) for d in cand]


def score_weighted(term_weights, k=10):
    """Score a bag of {term_id: weight} rather than a query string."""
    N = _index.N
    acc = np.zeros(N, dtype=np.float64)
    hit = False
    for i, qw in term_weights.items():
        if qw <= 0:
            continue
        p = _index.get_postings(_index.terms[i])
        if p is None:
            continue
        docs, tfs = p
        tfs = tfs.astype(np.float64)
        contrib = qw * _idf[i] * (tfs * (_k1 + 1.0)) / (tfs + _norm[docs])
        np.add.at(acc, docs, contrib)
        hit = True
    if not hit:
        return []
    n = min(k, N)
    cand = np.argpartition(-acc, n - 1)[:n]
    cand = cand[acc[cand] > 0]
    cand = cand[np.argsort(-acc[cand], kind="stable")]
    return [(_index.doc_ids[int(d)], float(acc[d])) for d in cand]