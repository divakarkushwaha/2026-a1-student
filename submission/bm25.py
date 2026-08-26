"""BM25 (Robertson & Zaragoza, 2009)."""
import math
from submission.text import analyze

_index = None
_idf = None
_norm = None   # precomputed k1*(1-b+b*dl/avgdl) per doc


def build(index, k1=1.2, b=0.75):
    global _index, _idf, _norm, _k1
    _index = index
    _k1 = k1
    N = index.N
    _idf = [math.log(1.0 + (N - df + 0.5) / (df + 0.5)) for df in index.df]
    _norm = (k1 * (1.0 - b + b * (index.doc_lens / index.avgdl))).astype("float64")


def score(query, k=10):
    scores = {}
    for term in analyze(query):
        i = _index.term_ids.get(term)
        if i is None:
            continue
        p = _index.get_postings(term)
        if p is None:
            continue
        idf = _idf[i]
        for docint, tf in zip(*p):
            scores[docint] = scores.get(docint, 0.0) + \
                idf * (tf * (_k1 + 1.0)) / (tf + _norm[docint])
    top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
    return [(_index.doc_ids[d], float(s)) for d, s in top]