"""
submission/indexer.py — inverted index with compressed on-disk postings.

Required component (assignment Section 4.1): built from scratch, no
external search/indexing library.

Design notes (for the report and defense):
  - Documents are mapped to internal integer ids at build time, so a
    posting stores an int rather than repeating the doc_id string.
  - Within a postings list doc ids are ascending, so we store GAPS
    (deltas) rather than absolute ids. Gaps are small numbers.
  - Small numbers are then variable-byte encoded: one byte for values
    under 128, instead of the four a fixed-width int would cost.
  - Term frequencies are stored SEPARATELY from gaps, at 4 bits each
    rather than a VByte byte. 73% of postings have tf=1 and 88% have
    tf<=2, so a whole byte per tf wastes most of its range. See
    _pack4() for why capping at 15 is safe.
  - Raw document text is deliberately NOT persisted. BM25 and VSM need
    only term frequencies and document lengths.
  - The compressed form is the on-disk form. decode_all() expands it
    once at load time into flat arrays so query-time scoring is pure
    NumPy with no per-posting Python loop.
"""
import os
from typing import Dict, List, Tuple

import numpy as np

from submission.text import analyze

TF_CAP = 15   # 4 bits


def tokenize(text: str) -> List[str]:
    """Kept for API compatibility; delegates to the shared analyzer so
    index time and query time can never drift apart."""
    return analyze(text)


def _vbyte_encode(nums) -> bytes:
    """Variable-byte encode. Low 7 bits carry payload; the high bit is
    set on the FINAL byte of each number."""
    out = bytearray()
    for n in nums:
        while True:
            b = n & 0x7F
            n >>= 7
            if n:
                out.append(b)
            else:
                out.append(b | 0x80)
                break
    return bytes(out)


def _vbyte_decode(buf, pos, count):
    """Decode `count` numbers starting at byte `pos`. Returns
    (numbers, new_pos)."""
    nums = []
    n = shift = 0
    while len(nums) < count:
        b = buf[pos]
        pos += 1
        if b & 0x80:
            n |= (b & 0x7F) << shift
            nums.append(n)
            n = shift = 0
        else:
            n |= b << shift
            shift += 7
    return nums, pos


def _pack4(tfs) -> bytes:
    """Pack term frequencies at 4 bits each, capped at TF_CAP.

    Capping is safe because BM25's tf*(k1+1)/(tf + K) saturates: with
    k1=2 the contribution at tf=15 is within a few percent of the
    contribution at tf=495 (the corpus maximum), so the ranking effect
    of the cap is negligible. Two tfs share one byte, low nibble first.
    """
    t = np.minimum(np.asarray(tfs), TF_CAP).astype(np.uint8)
    if len(t) % 2:
        t = np.concatenate([t, np.zeros(1, dtype=np.uint8)])
    packed = (t[0::2] | (t[1::2].astype(np.uint8) << np.uint8(4)))
    return packed.astype(np.uint8).tobytes()


def _unpack4(buf, n):
    """Inverse of _pack4; returns the first n term frequencies."""
    a = np.frombuffer(buf, dtype=np.uint8)
    out = np.empty(len(a) * 2, dtype=np.int32)
    out[0::2] = a & 0x0F
    out[1::2] = a >> 4
    return out[:n]


class InvertedIndex:
    def __init__(self):
        self.doc_ids: List[str] = []   # internal int -> external doc_id
        self.doc_lens = None           # tokens per document
        self.terms: List[str] = []     # sorted vocabulary
        self.term_ids: Dict[str, int] = {}
        self.df = None                 # np.uint32, parallel to terms
        self.offsets = None            # np.uint32, byte offset into blob
        self.postings: bytes = b""     # VByte doc-id gaps
        self.packed_tfs: bytes = b""   # 4-bit term frequencies
        self.N: int = 0
        self.avgdl: float = 0.0
        # populated by decode_all()
        self.flat_docs = None
        self.flat_tfs = None
        self.starts = None
        # populated by build_forward()
        self.fwd_terms = None
        self.fwd_tfs = None
        self.doc_starts = None

    # ---------------------------------------------------------------
    def build(self, corpus: List[Tuple[str, str]]) -> None:
        postings: Dict[str, List[Tuple[int, int]]] = {}
        lens: List[int] = []

        for docint, (doc_id, text) in enumerate(corpus):
            toks = analyze(text)
            self.doc_ids.append(doc_id)
            lens.append(len(toks))
            tfs: Dict[str, int] = {}
            for t in toks:
                tfs[t] = tfs.get(t, 0) + 1
            for t, tf in tfs.items():
                postings.setdefault(t, []).append((docint, tf))

        self.doc_lens = np.array(lens, dtype=np.uint32)
        self.N = len(self.doc_ids)
        self.avgdl = float(self.doc_lens.mean()) if self.N else 0.0

        self.terms = sorted(postings)
        self.term_ids = {t: i for i, t in enumerate(self.terms)}

        df, offs, blob = [], [], bytearray()
        all_tfs: List[int] = []
        for t in self.terms:
            plist = postings[t]          # ascending docint by construction
            offs.append(len(blob))
            df.append(len(plist))
            gaps, prev = [], 0
            for docint, _tf in plist:
                gaps.append(docint - prev)
                prev = docint
            blob += _vbyte_encode(gaps)
            all_tfs.extend(tf for _d, tf in plist)

        self.df = np.array(df, dtype=np.uint32)
        self.offsets = np.array(offs, dtype=np.uint32)
        self.postings = bytes(blob)
        self.packed_tfs = _pack4(np.array(all_tfs, dtype=np.int64))

    def document_frequency(self, term: str) -> int:
        i = self.term_ids.get(term)
        return int(self.df[i]) if i is not None else 0

    # ---------------------------------------------------------------
    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, "terms.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.terms))
        with open(os.path.join(index_dir, "docids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.doc_ids))
        with open(os.path.join(index_dir, "postings.bin"), "wb") as f:
            f.write(self.postings)
        with open(os.path.join(index_dir, "tfs.bin"), "wb") as f:
            f.write(self.packed_tfs)
        # uint32 offsets (max ~26M) and uint16 lengths (max well under
        # 65535) rather than uint64/uint32, then zlib on top.
        np.savez_compressed(
            os.path.join(index_dir, "meta.npz"),
            df=self.df.astype(np.uint32),
            offsets=self.offsets.astype(np.uint32),
            doc_lens=self.doc_lens.astype(np.uint16),
        )

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        ix = cls()
        with open(os.path.join(index_dir, "terms.txt"), encoding="utf-8") as f:
            ix.terms = f.read().split("\n")
        with open(os.path.join(index_dir, "docids.txt"), encoding="utf-8") as f:
            ix.doc_ids = f.read().split("\n")
        with open(os.path.join(index_dir, "postings.bin"), "rb") as f:
            ix.postings = f.read()
        with open(os.path.join(index_dir, "tfs.bin"), "rb") as f:
            ix.packed_tfs = f.read()
        m = np.load(os.path.join(index_dir, "meta.npz"))
        ix.df = m["df"]
        ix.offsets = m["offsets"]
        ix.doc_lens = m["doc_lens"].astype(np.uint32)
        ix.N = len(ix.doc_ids)
        ix.avgdl = float(ix.doc_lens.mean()) if ix.N else 0.0
        ix.term_ids = {t: i for i, t in enumerate(ix.terms)}
        return ix

    # ---------------------------------------------------------------
    def decode_all(self) -> None:
        """Expand the compressed form once into flat arrays.

        After this, the postings for term i are
        flat_docs[starts[i]:starts[i+1]] and the parallel tfs — so
        scoring is array slicing, not a Python loop per posting.
        """
        n_terms = len(self.terms)
        total = int(self.df.sum())
        docs = np.empty(total, dtype=np.int32)
        starts = np.empty(n_terms + 1, dtype=np.int64)

        pos = write = 0
        for i in range(n_terms):
            n = int(self.df[i])
            starts[i] = write
            gaps, pos = _vbyte_decode(self.postings, pos, n)
            docs[write:write + n] = np.cumsum(np.asarray(gaps, dtype=np.int32))
            write += n
        starts[n_terms] = write

        self.flat_docs = docs
        self.flat_tfs = _unpack4(self.packed_tfs, total)
        self.starts = starts
        self.postings = b""       # free the compressed copies
        self.packed_tfs = b""

    def get_postings(self, term: str):
        """(docints, tfs) for `term`, or None. decode_all() must have run."""
        i = self.term_ids.get(term)
        if i is None:
            return None
        s, e = self.starts[i], self.starts[i + 1]
        return self.flat_docs[s:e], self.flat_tfs[s:e]

    def build_forward(self) -> None:
        """Invert the flat postings into a document-major layout.

        Needed for relevance feedback, which asks what terms a given
        document contains: the opposite of what an inverted index
        answers. Built at load time from the already-decoded arrays
        rather than persisted, so it costs nothing against the
        index-size metric.
        """
        if self.flat_docs is None:
            raise RuntimeError("call decode_all() first")

        total = self.flat_docs.shape[0]
        term_of = np.empty(total, dtype=np.int32)
        for i in range(len(self.terms)):
            term_of[self.starts[i]:self.starts[i + 1]] = i

        counts = np.bincount(self.flat_docs, minlength=self.N)
        doc_starts = np.empty(self.N + 1, dtype=np.int64)
        doc_starts[0] = 0
        np.cumsum(counts, out=doc_starts[1:])

        order = np.argsort(self.flat_docs, kind="stable")
        self.fwd_terms = term_of[order]
        self.fwd_tfs = self.flat_tfs[order]
        self.doc_starts = doc_starts

    def doc_terms(self, docint):
        """(term_ids, tfs) for one document."""
        s, e = self.doc_starts[docint], self.doc_starts[docint + 1]
        return self.fwd_terms[s:e], self.fwd_tfs[s:e]