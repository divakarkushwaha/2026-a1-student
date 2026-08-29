"""Parameter sweeps on the dev set, using the harness's own metrics.

The index is built once and reused: every parameter swept here affects
scoring only, not indexing, so there is no reason to pay the build cost
per configuration.

Usage:
    python scripts/tune.py bm25      # k1 x b, plain BM25
    python scripts/tune.py rm3       # fb_docs x fb_terms x lambda
    python scripts/tune.py k1b_rm3   # k1 x b, with RM3 active

Each sweep writes runs/sweep_<name>.json and prints the best
configuration. Results are ranked with scripts/top.py.
"""
import argparse
import itertools
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

# Current best-known settings, used as fixed values for whichever
# parameters a given sweep is not varying.
BM25_DEFAULTS = {"k1": 2.0, "b": 0.6}
RM3_DEFAULTS = {"fb_docs": 50, "fb_terms": 40, "lam": 0.35}

# name -> (parameter grid, scorer)
SWEEPS = {
    "bm25": (
        {"k1": [0.9, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5],
         "b": [0.3, 0.4, 0.5, 0.6, 0.7, 0.75]},
        "bm25",
    ),
    "rm3": (
        {"fb_docs": [30, 40, 50, 75, 100],
         "fb_terms": [15, 20, 30, 40],
         "lam": [0.2, 0.25, 0.3, 0.35, 0.4]},
        "rm3",
    ),
    "k1b_rm3": (
        {"k1": [1.2, 1.5, 1.8, 2.0, 2.2, 2.5],
         "b": [0.3, 0.4, 0.5, 0.6, 0.7, 0.75]},
        "rm3",
    ),
}


def load_index():
    if not os.path.exists(os.path.join(INDEX_DIR, "terms.txt")):
        print("Building index once...")
        from submission.retrieve import build_index
        build_index(CORPUS, INDEX_DIR)
    print("Loading index...")
    ix = InvertedIndex.load(INDEX_DIR)
    ix.decode_all()
    ix.build_forward()
    return ix


def run_sweep(name):
    grid, scorer = SWEEPS[name]
    ix = load_index()
    queries = read_queries(QUERIES)
    qrels = read_qrels(QRELS)
    print(f"{len(queries)} queries, {len(qrels)} with qrels")
    print(f"sweep '{name}' over {grid} using {scorer}\n")

    keys = list(grid)
    results = []
    best = (None, -1.0)

    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        bm25_args = {**BM25_DEFAULTS, **{k: v for k, v in params.items()
                                         if k in BM25_DEFAULTS}}
        bm25.build(ix, **bm25_args)

        if scorer == "rm3":
            rm3_args = {**RM3_DEFAULTS, **{k: v for k, v in params.items()
                                           if k in RM3_DEFAULTS}}
            custom_scorer.build(ix, **rm3_args)
            score_fn = custom_scorer.score
        else:
            score_fn = bm25.score

        run = {qid: score_fn(text, 10) for qid, text in queries}
        agg = evaluate_run(run, qrels, k=10)["aggregate"]
        ndcg = agg["ndcg@10"]

        shown = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"{shown:<45} nDCG@10={ndcg:.4f}  MAP@10={agg['map@10']:.4f}")

        results.append({**params, **agg})
        if ndcg > best[1]:
            best = (params, ndcg)

    os.makedirs("runs", exist_ok=True)
    out = f"runs/sweep_{name}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    shown = ", ".join(f"{k}={v}" for k, v in best[0].items())
    print(f"\nBest: {shown} -> nDCG@10={best[1]:.4f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sweep", choices=sorted(SWEEPS))
    run_sweep(p.parse_args().sweep)