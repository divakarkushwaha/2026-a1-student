"""Sweep RM3 feedback parameters on the dev set using the harness's own
metrics.

Builds the index once, then re-scores across parameter combinations:
fb_docs, fb_terms and lambda affect scoring only, not indexing, so there
is no reason to pay the build cost per configuration.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from submission.indexer import InvertedIndex
from submission import bm25, custom_scorer
from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries

INDEX_DIR = "runs/tune_index"
CORPUS = "data/full/corpus.jsonl"
QUERIES = "data/full/queries_dev.tsv"
QRELS = "data/full/qrels_dev.txt"
OUT = "runs/sweep_rm3.json"

if not os.path.exists(os.path.join(INDEX_DIR, "terms.txt")):
    print("Building index once...")
    from submission.retrieve import build_index
    build_index(CORPUS, INDEX_DIR)

print("Loading index...")
ix = InvertedIndex.load(INDEX_DIR)
ix.decode_all()
ix.build_forward()

queries = read_queries(QUERIES)
qrels = read_qrels(QRELS)
print(f"{len(queries)} queries, {len(qrels)} with qrels")

bm25.build(ix, k1=2.0, b=0.6)

results = []
best = (None, None, None, -1.0)

for fb_docs in [30, 40, 50, 75, 100]:
    for fb_terms in [15, 20, 30, 40]:
        for lam in [0.2, 0.25, 0.3, 0.35, 0.4]:
            custom_scorer.build(ix, fb_docs=fb_docs, fb_terms=fb_terms, lam=lam)
            run = {qid: custom_scorer.score(text, 10) for qid, text in queries}
            agg = evaluate_run(run, qrels, k=10)["aggregate"]
            ndcg = agg["ndcg@10"]
            print(f"docs={fb_docs:<3} terms={fb_terms:<3} lam={lam:<4} "
                  f"nDCG@10={ndcg:.4f}  MAP@10={agg['map@10']:.4f}")
            results.append({"fb_docs": fb_docs, "fb_terms": fb_terms,
                            "lam": lam, **agg})
            if ndcg > best[3]:
                best = (fb_docs, fb_terms, lam, ndcg)

os.makedirs("runs", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nBest: fb_docs={best[0]}, fb_terms={best[1]}, lam={best[2]} "
      f"-> nDCG@10={best[3]:.4f}")