"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5 of the
assignment spec, "Submission Interface & Conformance Checking") — do not
rename them, change their signatures, or move them out of this file.

    build_index(corpus_path: str, index_dir: str) -> None
    load_index(index_dir: str) -> None
    retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]

Implementation notes:
  - build_index() tokenises the corpus, builds an inverted index with
    delta+VByte-compressed postings, and persists it to index_dir.
  - load_index() reconstructs that index in a fresh process, reading
    only index_dir, and precomputes BM25's IDF and length-normalisation
    arrays so retrieve() does no redundant work per query.
  - retrieve() scores with BM25 (Robertson & Zaragoza, 2009).
"""
from typing import List, Optional, Tuple

from submission.corpus_utils import load_corpus
from submission import bm25
from submission.indexer import InvertedIndex

# ---------------------------------------------------------------------------
# Module-level state. load_index() populates this; retrieve() reads it.
# build_index() runs in a SEPARATE process and cannot rely on this state
# surviving into load_index()/retrieve() — anything needed at query time
# must be written to index_dir in build_index() and read back in
# load_index().
# ---------------------------------------------------------------------------
_INDEX: Optional[InvertedIndex] = None


def build_index(corpus_path: str, index_dir: str) -> None:
    """Load the corpus, build whatever index structures you need, and
    write everything retrieve() will need into `index_dir`.

    Runs once, in its own process, before load_index() ever runs. Heavy
    one-time work — tokenising the whole corpus, building postings lists,
    computing collection statistics — belongs here, not in retrieve(), so
    it doesn't get charged against your per-query latency. Whatever you
    don't write to `index_dir` here does not exist as far as load_index()
    is concerned.
    """
    corpus = load_corpus(corpus_path)
    index = InvertedIndex()
    index.build(corpus)
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    """Reconstruct everything retrieve() needs, reading only from
    `index_dir`. Runs once, in a fresh process, before any retrieve()
    calls — there is no leftover state from build_index() to rely on.
    """
    global _INDEX
    _INDEX = InvertedIndex.load(index_dir)
    _INDEX.decode_all()
    bm25.build(_INDEX, k1=2.0, b=0.6)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, best first."""
    if _INDEX is None:
        raise RuntimeError(
            "retrieve() called before load_index(); the harness always "
            "calls build_index(corpus_path, index_dir) and then "
            "load_index(index_dir) — in that order, in two separate "
            "processes — before any retrieve() calls. If you're testing "
            "manually, do the same."
        )
    return bm25.score(query, k)