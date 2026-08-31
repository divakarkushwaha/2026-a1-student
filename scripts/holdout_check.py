"""Train/test split within the dev set.

The held-out leaderboard collapsed relative to dev, which suggests the
RM3 settings are fitting dev topics rather than generalising. This
splits the 50 dev topics in half, picks parameters on one half, and
reports the score on the other — a cheap proxy for the real held-out
behaviour that does not cost a daily submission.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from submission.indexer import InvertedIndex
from submission import bm25, custom_scorer
from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries

ix = InvertedIndex.load("runs/tune_index")
ix.decode_all()
ix.build_forward()

queries = read_queries("data/full/queries_dev.tsv")
qrels = read_qrels("data/full/qrels_dev.txt")

odd = [q for i, q in enumerate(queries) if i % 2]
even = [q for i, q in enumerate(queries) if not i % 2]


def score(qs, use_rm3, **kw):
    bm25.build(ix, k1=kw.get("k1", 2.0), b=kw.get("b", 0.6))
    if use_rm3:
        custom_scorer.build(ix, fb_docs=kw["fb_docs"],
                            fb_terms=kw["fb_terms"], lam=kw["lam"])
        fn = custom_scorer.score
    else:
        fn = bm25.score
    run = {qid: fn(text, 10) for qid, text in qs}
    sub_qrels = {qid: qrels[qid] for qid, _ in qs if qid in qrels}
    return evaluate_run(run, sub_qrels, k=10)["aggregate"]["ndcg@10"]


configs = [
    ("BM25 only",            False, {}),
    ("RM3 current",          True,  dict(fb_docs=50, fb_terms=40, lam=0.35)),
    ("RM3 conservative",     True,  dict(fb_docs=10, fb_terms=20, lam=0.6)),
    ("RM3 very conservative", True, dict(fb_docs=5,  fb_terms=10, lam=0.7)),
]

print(f"{'config':<24} {'half A':>8} {'half B':>8} {'gap':>8}")
for name, use_rm3, kw in configs:
    a = score(odd, use_rm3, **kw)
    b = score(even, use_rm3, **kw)
    print(f"{name:<24} {a:>8.4f} {b:>8.4f} {abs(a-b):>8.4f}")