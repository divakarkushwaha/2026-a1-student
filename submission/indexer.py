"""Inverted index with delta+VByte compressed postings."""
import numpy as np
import os
from submission.text import analyze


def _vbyte_encode(nums) -> bytes:
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


class InvertedIndex:
    def __init__(self):
        self.doc_ids = []       # internal int -> external doc_id str
        self.doc_lens = None    # np.uint32 array
        self.terms = []         # sorted term list
        self.df = None          # np.uint32, parallel to terms
        self.offsets = None     # np.uint64, byte offset into postings blob
        self.postings = b""     # concatenated compressed postings
        self.N = 0
        self.avgdl = 0.0

    def build(self, corpus):
        postings = {}           # term -> [(docint, tf), ...]
        lens = []
        for docint, (doc_id, text) in enumerate(corpus):
            toks = analyze(text)
            self.doc_ids.append(doc_id)
            lens.append(len(toks))
            tfs = {}
            for t in toks:
                tfs[t] = tfs.get(t, 0) + 1
            for t, tf in tfs.items():
                postings.setdefault(t, []).append((docint, tf))

        self.doc_lens = np.array(lens, dtype=np.uint32)
        self.N = len(self.doc_ids)
        self.avgdl = float(self.doc_lens.mean()) if self.N else 0.0

        self.terms = sorted(postings)
        df, offs, blob = [], [], bytearray()
        for t in self.terms:
            plist = postings[t]          # already in ascending docint order
            offs.append(len(blob))
            df.append(len(plist))
            gaps, prev = [], 0
            for docint, _ in plist:
                gaps.append(docint - prev)
                prev = docint
            blob += _vbyte_encode(gaps)
            blob += _vbyte_encode([tf for _, tf in plist])
        self.df = np.array(df, dtype=np.uint32)
        self.offsets = np.array(offs, dtype=np.uint64)
        self.postings = bytes(blob)

    def save(self, index_dir):
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, "terms.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.terms))
        with open(os.path.join(index_dir, "docids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.doc_ids))
        with open(os.path.join(index_dir, "postings.bin"), "wb") as f:
            f.write(self.postings)
        np.savez(os.path.join(index_dir, "meta.npz"),
                 df=self.df, offsets=self.offsets, doc_lens=self.doc_lens)

    @classmethod
    def load(cls, index_dir):
        ix = cls()
        with open(os.path.join(index_dir, "terms.txt"), encoding="utf-8") as f:
            ix.terms = f.read().split("\n")
        with open(os.path.join(index_dir, "docids.txt"), encoding="utf-8") as f:
            ix.doc_ids = f.read().split("\n")
        with open(os.path.join(index_dir, "postings.bin"), "rb") as f:
            ix.postings = f.read()
        m = np.load(os.path.join(index_dir, "meta.npz"))
        ix.df, ix.offsets, ix.doc_lens = m["df"], m["offsets"], m["doc_lens"]
        ix.N = len(ix.doc_ids)
        ix.avgdl = float(ix.doc_lens.mean()) if ix.N else 0.0
        ix.term_ids = {t: i for i, t in enumerate(ix.terms)}
        return ix

    def get_postings(self, term):
        """Return (docints, tfs) or None."""
        i = self.term_ids.get(term)
        if i is None:
            return None
        pos = int(self.offsets[i])
        n = int(self.df[i])
        gaps, pos = _vbyte_decode(self.postings, pos, n)
        tfs, _ = _vbyte_decode(self.postings, pos, n)
        docints, cur = [], 0
        for g in gaps:
            cur += g
            docints.append(cur)
        return docints, tfs