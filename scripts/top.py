"""Rank a sweep's configurations by nDCG@10.

Usage: python scripts/top.py k1b_rm3 [n]
"""
import json
import sys

name = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 15

rows = json.load(open(f"runs/sweep_{name}.json"))
rows.sort(key=lambda x: -x["ndcg@10"])

keys = [k for k in rows[0] if k not in ("ndcg@10", "map@10", "mrr", "p@10")]
for r in rows[:n]:
    shown = "  ".join(f"{k}={r[k]}" for k in keys)
    print(f"{shown:<45} {r['ndcg@10']:.4f}")