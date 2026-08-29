"""Shared text analysis. MUST be identical at index time and query time.

The stopword list is NLTK's English list, embedded here as a literal
rather than loaded via nltk.corpus.stopwords: the grading container runs
with --network none and without nltk_data, so a runtime download would
fail. Instructors confirmed on Piazza that predefined stopword lists are
permitted.
"""
import re
from nltk.stem import PorterStemmer

_TOKEN = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()
_cache: dict[str, str] = {}

STOPWORDS = frozenset("""
a about above after again against ain all am an and any are aren as at be
because been before being below between both but by can couldn d did didn
do does doesn doing don down during each few for from further had hadn has
hasn have haven having he her here hers herself him himself his how i if in
into is isn it its itself just ll m ma me mightn more most mustn my myself
needn no nor not now o of off on once only or other our ours ourselves out
over own re s same shan she should shouldn so some such t than that the
their theirs them themselves then there these they this those through to
too under until up ve very was wasn we were weren what when where which
while who whom why will with won wouldn y you your yours yourself
yourselves
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