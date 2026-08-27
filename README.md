# My Submission

BM25 retriever over a from-scratch inverted index with delta + variable-byte compressed postings. Dev-set results on the TREC-COVID corpus (171,332 documents, 50 topics):

| Metric | Value |
|---|---|
| nDCG@10 | 0.6450 |
| MAP@10 | 0.0161 |
| MRR | 0.8933 |
| P@10 | 0.7220 |
| Index build time | 21.5 s |
| Index load time | 5.1 s |
| Index size on disk | 32.2 MB |
| Mean query latency | 2.1 ms |

## Reproducing

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Fetch the corpus into `data/full/` (needs network; not required for the toy set):

```bash
python scripts/download_full_corpus.py
```

Full harness run:

```bash
python -m harness.run_harness \
  --corpus data/full/corpus.jsonl \
  --queries data/full/queries_dev.tsv \
  --qrels data/full/qrels_dev.txt \
  --run-out runs/full_run.trec \
  --report-out runs/full_report.json
```

Tests (interface conformance, metrics, and my Boolean/VSM unit tests):

```bash
pytest tests/ -v
```

Parameter sweep (builds the index once, re-scores across `k1`/`b`):

```bash
python scripts/tune.py
```

Both `build_index()` and `load_index()` are deterministic: term ordering is lexicographic, document ids are assigned in corpus order, and ties in `retrieve()` break by ascending internal document id.

## Files I wrote

| File | Contents |
|---|---|
| `submission/text.py` | Shared analyzer (lowercase, alphanumeric tokens, stopword removal, Porter stemming with memoisation). Imported by both the indexer and every scorer so index-time and query-time tokenisation cannot drift apart. |
| `submission/indexer.py` | `InvertedIndex`: postings construction, delta + VByte compression, `save()`/`load()`, and `decode_all()`. |
| `submission/bm25.py` | BM25 with tunable `k1`, `b`; vectorised scoring. |
| `submission/boolean_vsm.py` | Boolean AND/OR over postings; `lnc.ltc` TF-IDF cosine ranking. |
| `scripts/tune.py` | Parameter sweep using the harness's own metrics. |
| `tests/test_boolean_vsm.py` | Hand-verified unit tests for Boolean and VSM. |

## Design decisions

**Compressed postings.** Document ids within a postings list are ascending, so I store gaps rather than absolute ids and variable-byte encode them — one byte per value under 128 instead of four. Raw document text is deliberately not persisted: BM25 and VSM need only term frequencies and document lengths.

**Decode once at load.** The compressed blob is the on-disk form; `decode_all()` expands it into flat NumPy arrays at load time so query scoring is array slicing with no per-posting Python loop. This traded ~0.8 s of load time for a drop in mean query latency from 162 ms to 2.1 ms.

**Parameters.** `k1=2.0, b=0.6`, chosen from a joint sweep (see `docs/sweep_refined.json`). The nominal dev-set maximum was `k1=2.2, b=0.65` at 0.6503, but the surface is flat across `k1 ∈ [1.8, 2.5]`, `b ∈ [0.5, 0.7]` — differences under ~0.005 on 50 queries are noise, so I chose central values rather than the argmax to avoid overfitting the dev set.

**VSM norms.** Document vector norms are recomputed at load rather than persisted, keeping them off the index-size metric at a cost of under one second.

---



# A1 — Sparse Retrieval Arena: Starter Repository

This is the starter repository for **Assignment 1: Sparse Retrieval
Arena**. If anything here conflicts with the assignment spec document,
the spec document governs the rules (grading, deadlines, integrity); this
repo governs the exact code interface, which the spec explicitly defers
to it for ("exact signature given in the starter repo").

## What you're building

An inverted-index retrieval engine with Boolean/vector-space and BM25
scorers, tuned to place well on class-wide percentile rank on nDCG@10.
Clearing the instructor's undisclosed floor is necessary but not
sufficient — see "A note on the baseline" below. Full requirements are
in the assignment spec — start there if you haven't read it yet.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # or your preferred env manager
pip install -r requirements.txt

# Run the full harness against the toy set — should work immediately,
# using the trivial starter retrieve() that ships in submission/retrieve.py.
bash scripts/smoke_test.sh
```

You should see a report ending in something like:

```
nDCG@10:              0.29..   (mediocre — this ignores the query entirely, not a real retriever)
...
Index size on disk:   131.0B  (131 bytes)
...
Provisional score (80% weight, nDCG@10 + MAP@10): 0.22..
(Remaining ±10% efficiency modifier and 0-10% index-size score are
 class-relative — applied when course staff aggregate the full leaderboard.)
Local reference nDCG@10: 0.86..  (beat it: False)
(This is YOUR local comparison run, not the official grading baseline — see Section 7.)
```

The starter `retrieve()` isn't near zero — on a 20-document toy corpus,
returning "the first 10 documents" already overlaps with some relevant ones
by chance. It should still be clearly and consistently behind the local
reference. Note that number is *not* your grading target: it's a
same-parameters-as-everyone-else comparison generated by
`harness/generate_baseline.py` on the toy set, useful for catching an
obviously broken ranking, not for judging whether you've done enough. The
assignment's real "beat this" floor is a separately tuned, undisclosed
reference on data you don't have — see Section 7 of the assignment spec.
Your actual target is placing well on class-wide percentile rank, which this
toy-set number tells you nothing about.

## A note on the baseline

Two different things in this repo are both loosely called "baseline,"
and it's worth keeping them straight:

1. **The trivial starter `retrieve()`** in `submission/retrieve.py` —
   ignores the query, returns the first *k* documents. Replace it.
2. **`harness/generate_baseline.py`'s output** — a local BM25 comparison
   run *you* generate, with generic textbook parameters, for your own
   sanity-checking.

Neither of these is the number course staff actually grade you against.
The real floor (assignment Section 7) is tuned separately, on data you
don't have, with parameters that are not published — deliberately, so
that "match the textbook defaults" isn't a substitute for building a
system that's actually good. Treat the floor as a sanity check, not a
target: the leaderboard-performance component of your grade is driven by
percentile rank against the rest of the class, and there is no ceiling
on that but the class itself.

## Where to write code

Everything you implement lives in `submission/`:

| File | What goes here |
|---|---|
| `submission/retrieve.py` | **The required entrypoint** (`build_index`, `load_index`, `retrieve`). Wire your real scorer in here — see the `TODO(you)` markers. Do not change its function signatures. |
| `submission/indexer.py` | Your inverted index: postings, document lengths, collection stats, plus `save()`/`load()` for on-disk persistence. |
| `submission/boolean_vsm.py` | Boolean AND/OR retrieval + TF-IDF cosine vector-space ranking. |
| `submission/bm25.py` | BM25 with tunable `k1`, `b`. |
| `submission/custom_scorer.py` | Optional: your own combined/heuristic scorer. |

Every file above has a docstring with the relevant formula and a
reference back to the assignment section it satisfies — read those before
you start.

**You may not use an existing search/indexing library** (Lucene,
Elasticsearch, Pyserini, Whoosh, `rank_bm25`, etc.) inside `submission/`.
Standard libraries for tokenisation/stemming (e.g. NLTK) and numeric
libraries (NumPy) are fine — and you're not limited to pure Python either:
a C/C++ extension you compile yourself (Cython, already in
`requirements.txt`, or pybind11) is fine too, see
`docs/SUBMISSION_INTERFACE.md`, "Compiled extensions". See the
assignment's Academic Integrity section for the full policy, including
AI-use disclosure and code
provenance requirements.

## Running the harness yourself

```bash
python -m harness.run_harness \
  --corpus data/toy/corpus.jsonl \
  --queries data/toy/queries_dev.tsv \
  --qrels data/toy/qrels_dev.txt \
  --baseline-run data/toy/reference_bm25_run_dev.trec \
  --run-out runs/dev_run.trec \
  --report-out runs/dev_report.json
```

This is the *exact* code path (`harness/run_harness.py`,
`harness/metrics.py`) used to compute your leaderboard score — the only
things that differ at real grading time are which corpus/topics/qrels
file is passed in (the released dev set, then later the private held-out
set you never see) and that course infrastructure runs it for you rather
than you running it locally. See `harness/metrics.py` for exactly how
nDCG@10, MAP@10, MRR, and P@k are computed, and `harness/leaderboard.py` for
how they combine into your leaderboard score.

Under the hood, this one command spawns `build_index()` and
`load_index()`/`retrieve()` as two separate subprocesses of itself, with
a `--index-dir` on disk in between — see the module docstring at the top
of `harness/run_harness.py` for why, and `docs/SUBMISSION_INTERFACE.md`
for the full three-function contract.

To test against the real assignment corpus instead of the toy set, run
`python scripts/download_full_corpus.py` first (see `data/README.md`),
then point `--corpus`/`--queries`/`--qrels` at `data/full/` instead.

## Before you push: run the smoke test

```bash
bash scripts/smoke_test.sh
```

This runs the same interface-conformance tests, metrics unit tests, and
full harness pass that CI runs on every push
(`.github/workflows/conformance.yml`). Fix anything it flags before your
conformance freeze (48 hours before the deadline — see
`docs/SUBMISSION_INTERFACE.md`).

## Repository layout

```
.
├── data/
│   ├── toy/                 # small hand-built set for fast local dev (ships here)
│   ├── README.md            # data format + how to get the real corpus
│   └── full/                # created by scripts/download_full_corpus.py (gitignored)
├── submission/               # <-- you write code here
├── harness/                  # scoring code (read-only reference; don't need to edit)
├── tests/                    # conformance + metrics unit tests
├── scripts/
│   ├── download_full_corpus.py
│   └── smoke_test.sh
├── docs/
│   ├── SUBMISSION_INTERFACE.md   # the exact, binding interface contract
│   └── DOCKER_SUBMISSION.md      # what the Dockerfile is for, and the grading trust boundary
├── Dockerfile                 # how course staff run every submission
└── .github/workflows/conformance.yml   # what runs on every push
```

## Getting help

Discussing high-level strategy with classmates is fine. Sharing code, a
tuned parameter file, or your `submission/` implementation is not — see
the assignment's Academic Integrity section, and remember every team sits
a short oral defense after the leaderboard closes where you'll be asked
to explain and modify your own submission live.
