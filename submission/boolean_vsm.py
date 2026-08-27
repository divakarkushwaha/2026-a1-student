"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1).

Weighting scheme: lnc.ltc in SMART notation.
  Documents (lnc): logarithmic tf, no idf, cosine normalised.
  Queries  (ltc): logarithmic tf, idf, cosine normalised.

    w(t,d) = 1 + log10(tf(t,d))                       [no idf on docs]
    w(t,q) = (1 + log10(tf(t,q))) * log10(N / df(t))
    sim(q,d) = (q . d) / (||q|| * ||d||)

Leaving idf off the document side is the standard lnc.ltc choice: idf is
a collection-level property, so applying it to both sides squares its
effect on any term present in both query and document.
"""
import math
from typing import Dict, List, Tuple

import numpy as np

from submission.indexer import InvertedIndex
from submission.text import analyze

_index: InvertedIndex = None
_doc_norms = None   # ||d|| under lnc weighting, per internal doc id


def build(index: InvertedIndex) -> None:
    """Precompute document vector norms for cosine normalisation.

    Called from retrieve.load_index(). The norms are derived from the
    loaded index rather than persisted, so they cost nothing against the
    index-size score — one pass over the postings at load time.
    """
    global _index, _doc_norms
    _index = index
    sq = np.zeros(index.N, dtype=np.float64)
    for i in range(len(index.terms)):
        p = index.get_postings(index.terms[i])
        if p is None:
            continue
        docs, tfs = p
        w = 1.0 + np.log10(np.asarray(tfs, dtype=np.float64))
        np.add.at(sq, np.asarray(docs), w * w)
    _doc_norms = np.sqrt(sq)
    _doc_norms[_doc_norms == 0.0] = 1.0     # avoid divide-by-zero


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Unranked doc_ids matching `query` as a conjunction or disjunction."""
    if _index is None:
        raise RuntimeError("boolean_vsm.build() must be called first")

    terms = analyze(query)
    if not terms:
        return []

    sets = []
    for t in terms:
        p = _index.get_postings(t)
        sets.append(np.asarray([], dtype=np.int64) if p is None
                    else np.asarray(p[0], dtype=np.int64))

    if mode == "and":
        result = sets[0]
        for s in sets[1:]:
            result = np.intersect1d(result, s, assume_unique=True)
    elif mode == "or":
        result = sets[0]
        for s in sets[1:]:
            result = np.union1d(result, s)
    else:
        raise ValueError(f"mode must be 'and' or 'or', got {mode!r}")

    return [_index.doc_ids[int(d)] for d in np.sort(result)]


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Up to k (doc_id, score) pairs ranked by lnc.ltc cosine similarity."""
    if _index is None:
        raise RuntimeError("boolean_vsm.build() must be called first")

    terms = analyze(query)
    if not terms:
        return []

    qtf: Dict[str, int] = {}
    for t in terms:
        qtf[t] = qtf.get(t, 0) + 1

    N = _index.N
    acc = np.zeros(N, dtype=np.float64)
    qsq = 0.0
    hit = False

    for t, tf in qtf.items():
        i = _index.term_ids.get(t)
        if i is None:
            continue
        df = int(_index.df[i])
        if df == 0:
            continue
        wq = (1.0 + math.log10(tf)) * math.log10(N / df)
        if wq == 0.0:
            continue
        qsq += wq * wq

        docs, tfs = _index.get_postings(t)
        wd = 1.0 + np.log10(np.asarray(tfs, dtype=np.float64))
        np.add.at(acc, np.asarray(docs), wq * wd)
        hit = True

    if not hit or qsq == 0.0:
        return []

    acc /= (_doc_norms * math.sqrt(qsq))

    n = min(k, N)
    cand = np.argpartition(-acc, n - 1)[:n]
    cand = cand[acc[cand] > 0]
    cand = cand[np.argsort(-acc[cand], kind="stable")]
    return [(_index.doc_ids[int(d)], float(acc[d])) for d in cand]