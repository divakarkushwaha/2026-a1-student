"""How much of the vocabulary and postings would pruning remove?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from submission.indexer import InvertedIndex

ix = InvertedIndex.load("runs/tune_index")

df = ix.df.astype(np.int64)
total_terms = len(ix.terms)
total_post = int(df.sum())

print(f"vocabulary: {total_terms:,} terms, {total_post:,} postings")
print(f"term string bytes: {sum(len(t) for t in ix.terms):,}")

for thresh in (1, 2, 3, 5):
    mask = df <= thresh
    print(f"df <= {thresh}: {mask.sum():,} terms "
          f"({100*mask.sum()/total_terms:.1f}%), "
          f"{int(df[mask].sum()):,} postings "
          f"({100*df[mask].sum()/total_post:.1f}%), "
          f"{sum(len(ix.terms[i]) for i in np.where(mask)[0]):,} term bytes")

digits = [i for i, t in enumerate(ix.terms) if t.isdigit()]
print(f"\npure-digit terms: {len(digits):,} "
      f"({100*len(digits)/total_terms:.1f}%), "
      f"{int(df[digits].sum()):,} postings")