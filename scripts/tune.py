"""Sweep BM25 k1/b on the dev set using the harness's own metrics.

Builds the index once, then re-scores across parameter pairs: k1 and b
affect scoring only, not indexing, so there is no reason to pay the
build cost per configuration.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from submission.indexer import InvertedIndex
from submission import bm25
from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries

INDEX_DIR = "runs/tune_index"
CORPUS = "data/full/corpus.jsonl"
QUERIES = "data/full/queries_dev.tsv"
QRELS = "data/full/qrels_dev.txt"

if not os.path.exists(os.path.join(INDEX_DIR, "terms.txt")):
    print("Building index once...")
    from submission.retrieve import build_index
    build_index(CORPUS, INDEX_DIR)

print("Loading index...")
ix = InvertedIndex.load(INDEX_DIR)
ix.decode_all()

queries = read_queries(QUERIES)
qrels = read_qrels(QRELS)
print(f"{len(queries)} queries, {len(qrels)} with qrels")

results = []
best = (None, None, -1.0)

for k1 in [1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0]:
    for b in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
        bm25.build(ix, k1=k1, b=b)
        run = {qid: bm25.score(text, 10) for qid, text in queries}
        agg = evaluate_run(run, qrels, k=10)["aggregate"]
        ndcg = agg["ndcg@10"]
        print(f"k1={k1:<4} b={b:<5} nDCG@10={ndcg:.4f}  MAP@10={agg['map@10']:.4f}")
        results.append({"k1": k1, "b": b, **agg})
        if ndcg > best[2]:
            best = (k1, b, ndcg)

os.makedirs("runs", exist_ok=True)
with open("runs/sweep.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nBest: k1={best[0]}, b={best[1]} -> nDCG@10={best[2]:.4f}")