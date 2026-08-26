"""Shared text analysis. MUST be identical at index time and query time."""
import re
from nltk.stem import PorterStemmer

_TOKEN = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()
_cache: dict[str, str] = {}

STOPWORDS = frozenset("""
a an and are as at be by for from has he in is it its of on that the to was were will with
this these those there their they i you we but not or if which who whom what when where how
""".split())


def analyze(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t in STOPWORDS or len(t) < 2:
            continue
        s = _cache.get(t)
        if s is None:
            s = _stemmer.stem(t)
            _cache[t] = s
        out.append(s)
    return out