"""
submission/custom_scorer.py — optional combined/custom scorer.

Not required, but this is explicitly called out in the assignment
(Section 4.1) as "where separation in the leaderboard tends to happen":
any linear or non-linear combination of your Boolean/VSM and BM25
signals, additional features (e.g. proximity/bigram overlap), or your
own heuristic.

If you use this, wire it in from submission/retrieve.py's retrieve()
instead of calling a single scorer directly, and describe what you did
and why in your report (Section 7, "one-paragraph description of your
final competition entry").
"""
from typing import List, Tuple

from submission.indexer import InvertedIndex



def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(), not retrieve.build_index() — the
    harness runs those two in separate processes. Anything this needs at
    query time either comes from the loaded InvertedIndex or must have
    been written to index_dir by InvertedIndex.save() (which then counts
    toward your index-size score)."""
    raise NotImplementedError


def score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by your
    own combined/custom scoring function, highest score first."""
    raise NotImplementedError

"""
submission/custom_scorer.py — RM3 pseudo-relevance feedback over BM25.

Relevance model (Lavrenko & Croft, 2001), interpolated with the original
query (the "RM3" variant):

    p(t|R)  = SUM_d  P(d) * tf(t,d)/|d|        over the top fb_docs
    w'(t)   = lambda * w_orig(t) + (1-lambda) * p(t|R)

P(d) is the first-pass BM25 score, normalised to sum to 1 over the
feedback set. Expansion terms are the top fb_terms by p(t|R).
"""
import numpy as np

from submission import bm25
from submission.text import analyze

_index = None
_fb_docs = 10
_fb_terms = 20
_lambda = 0.5


def build(index, fb_docs=10, fb_terms=20, lam=0.5):
    global _index, _fb_docs, _fb_terms, _lambda
    _index = index
    _fb_docs, _fb_terms, _lambda = fb_docs, fb_terms, lam


def score(query, k=10):
    # First pass: internal doc ids and scores
    first = bm25.score_internal(query, _fb_docs)
    if not first:
        return []

    docints = np.array([d for d, _ in first], dtype=np.int64)
    scores = np.array([s for _, s in first], dtype=np.float64)
    if scores.sum() <= 0:
        return bm25.score(query, k)
    pd = scores / scores.sum()

    # Estimate p(t|R)
    weights = {}
    for rank, docint in enumerate(docints):
        tids, tfs = _index.doc_terms(int(docint))
        dl = float(tfs.sum())
        if dl == 0:
            continue
        contrib = pd[rank] * (tfs.astype(np.float64) / dl)
        for tid, w in zip(tids, contrib):
            weights[int(tid)] = weights.get(int(tid), 0.0) + float(w)

    if not weights:
        return bm25.score(query, k)

    top = sorted(weights.items(), key=lambda x: -x[1])[:_fb_terms]
    total = sum(w for _, w in top)
    if total <= 0:
        return bm25.score(query, k)

    expansion = {tid: w / total for tid, w in top}

    # Original query terms, uniformly weighted
    orig = {}
    qterms = analyze(query)
    if qterms:
        for t in qterms:
            i = _index.term_ids.get(t)
            if i is not None:
                orig[i] = orig.get(i, 0.0) + 1.0 / len(qterms)

    final = {}
    for tid, w in orig.items():
        final[tid] = _lambda * w
    for tid, w in expansion.items():
        final[tid] = final.get(tid, 0.0) + (1.0 - _lambda) * w

    return bm25.score_weighted(final, k)