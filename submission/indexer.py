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

BLOCK = 128   # gaps per bitpacked block


def _bitpack_encode(gaps) -> bytes:
    """Block-based bitpacking of doc-id gaps.

    VByte cannot spend less than one byte per value, but gaps in a long
    postings list are often 1 or 2 — a common term appears in nearly
    consecutive documents. Splitting into blocks of BLOCK gaps and
    packing each block at the minimum width its largest gap needs gets
    below that floor: 128 gaps of value <= 3 cost 2 bits each (32 bytes
    plus a 1-byte width header) instead of 128 bytes.

    Layout per block: 1 byte width w, then BLOCK values of w bits each,
    little-endian bit order, padded to a byte boundary. A trailing
    partial block is padded with zeros to BLOCK values.
    """
    g = np.asarray(gaps, dtype=np.uint32)
    out = bytearray()
    for start in range(0, len(g), BLOCK):
        block = g[start:start + BLOCK]
        if len(block) < BLOCK:
            block = np.concatenate(
                [block, np.zeros(BLOCK - len(block), dtype=np.uint32)])
        m = int(block.max())
        w = max(1, int(m).bit_length())
        out.append(w)
        bits = np.unpackbits(
            block.view(np.uint8).reshape(-1, 4)[:, ::-1], axis=1)[:, -w:]
        out += np.packbits(bits.ravel()).tobytes()
    return bytes(out)


def _bitpack_decode(buf, pos, count):
    """Decode `count` gaps written by _bitpack_encode. Returns
    (gaps array, new_pos)."""
    n_blocks = (count + BLOCK - 1) // BLOCK
    out = np.empty(n_blocks * BLOCK, dtype=np.uint32)
    for bi in range(n_blocks):
        w = buf[pos]
        pos += 1
        nbytes = (BLOCK * w + 7) // 8
        bits = np.unpackbits(
            np.frombuffer(buf, dtype=np.uint8, count=nbytes, offset=pos))
        pos += nbytes
        vals = bits[:BLOCK * w].reshape(BLOCK, w)
        padded = np.zeros((BLOCK, 32), dtype=np.uint8)
        padded[:, -w:] = vals
        out[bi * BLOCK:(bi + 1) * BLOCK] = np.packbits(
            padded, axis=1)[:, ::-1].copy().view(np.uint32).ravel()
    return out[:count], pos

TF_INLINE_MAX = 3    # tf values 1..3 fit in the 2-bit code
TF_ESCAPE = 3        # code 3 means "read the next exception"


def _pack_tfs(tfs):
    """Pack term frequencies at 2 bits each, with an exception list.

    73% of postings have tf=1 and 88% have tf<=2, so even a 4-bit code
    wastes most of its range. Codes 0,1,2 mean tf=1,2,3; code 3 says the
    true value is the next entry in the exception array, which is stored
    as uint16 (the corpus maximum tf is 495). At ~5% exceptions this
    costs 0.25 bytes per posting plus a small side array, against 0.5
    bytes for the 4-bit scheme.

    Returns (packed_bytes, exceptions_uint16_array).
    """
    t = np.asarray(tfs, dtype=np.int64)
    codes = np.where(t <= TF_INLINE_MAX, t - 1, TF_ESCAPE).astype(np.uint8)
    exceptions = t[t > TF_INLINE_MAX].astype(np.uint16)

    pad = (-len(codes)) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    packed = (codes[0::4]
              | (codes[1::4] << np.uint8(2))
              | (codes[2::4] << np.uint8(4))
              | (codes[3::4] << np.uint8(6)))
    return packed.astype(np.uint8).tobytes(), exceptions


def _unpack_tfs(buf, exceptions, n):
    """Inverse of _pack_tfs; returns the first n term frequencies."""
    a = np.frombuffer(buf, dtype=np.uint8)
    codes = np.empty(len(a) * 4, dtype=np.int32)
    codes[0::4] = a & 0x03
    codes[1::4] = (a >> 2) & 0x03
    codes[2::4] = (a >> 4) & 0x03
    codes[3::4] = (a >> 6) & 0x03
    codes = codes[:n]

    out = codes + 1
    esc = codes == TF_ESCAPE
    out[esc] = exceptions[:esc.sum()]
    return out


class InvertedIndex:
    def __init__(self):
        self.doc_ids: List[str] = []   # internal int -> external doc_id
        self.doc_lens = None           # tokens per document
        self.terms: List[str] = []     # sorted vocabulary
        self.term_ids: Dict[str, int] = {}
        self.df = None                 # np.uint32, parallel to terms
        self.offsets = None            # np.uint32, byte offset into blob
        self.postings: bytes = b""     # VByte doc-id gaps
        self.packed_tfs: bytes = b""
        self.tf_exceptions = None  
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

            # Bitpacking pays off only on lists long enough to fill
            # blocks: 52% of terms are singletons, and padding a
            # 1-gap list to a 128-value block would make it larger,
            # not smaller. Short lists stay on VByte. One flag byte
            # per term records which was used.
            if len(gaps) >= BLOCK:
                blob += b"\x01" + _bitpack_encode(gaps)
            else:
                blob += b"\x00" + _vbyte_encode(gaps)

            all_tfs.extend(tf for _d, tf in plist)

        self.df = np.array(df, dtype=np.uint32)
        self.offsets = np.array(offs, dtype=np.uint32)
        self.postings = bytes(blob)
        self.packed_tfs, self.tf_exceptions = _pack_tfs(
        np.array(all_tfs, dtype=np.int64))

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
            tf_exc=self.tf_exceptions,
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
        ix.tf_exceptions = m["tf_exc"]
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

            flag = self.postings[pos]
            pos += 1
            if flag:
                gaps, pos = _bitpack_decode(self.postings, pos, n)
                docs[write:write + n] = np.cumsum(gaps.astype(np.int32))
            else:
                gaps, pos = _vbyte_decode(self.postings, pos, n)
                docs[write:write + n] = np.cumsum(
                    np.asarray(gaps, dtype=np.int32))

            write += n
        starts[n_terms] = write

        self.flat_docs = docs
        self.flat_tfs = _unpack_tfs(self.packed_tfs, self.tf_exceptions, total)
        self.starts = starts
        self.postings = b""       # free the compressed copies
        self.packed_tfs = b""
        self.tf_exceptions = None

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