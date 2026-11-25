"""Word-set view of a text, shared by everything that compares two of them.

Language-agnostic by construction: runs of letters, digits and combining
marks over a case-folded string keep words in any script and drop
everything else, so the same function segments Spanish, German and Hindi
without knowing which it is looking at.

Chinese and Japanese are written without spaces, so ``\\w+`` alone reads a
whole sentence as one word: two different sentences then share nothing,
an identical restatement shares exactly one token (too few to judge), and
the lexical index cannot find "東京" inside "東京に住んでいます". Runs of
those scripts are therefore cut into overlapping character bigrams — the
standard dictionary-free segmentation for CJK search, and the one Lucene's
CJK analyzer uses. It needs no word list, only the Unicode blocks the
scripts live in, and leaves every spaced script exactly as it was.
"""
from __future__ import annotations

import re
import unicodedata
from itertools import chain


def _combining_marks() -> str:
    """Every combining mark the running Unicode database knows, as the body
    of a regex character class.

    Marks are assigned only in planes 0, 1 and 14, so those are all the scan
    reads: 20 ms once, against a quarter of a second for every code point.
    """
    ranges: list[list[int]] = []
    for code in chain(range(0x20000), range(0xE0000, 0xE1000)):
        if unicodedata.category(chr(code)).startswith("M"):
            if ranges and ranges[-1][1] == code - 1:
                ranges[-1][1] = code
            else:
                ranges.append([code, code])
    return "".join(chr(a) if a == b else f"{chr(a)}-{chr(b)}" for a, b in ranges)


# A word is a run of letters and digits with the combining marks that belong
# to it. ``\w`` alone leaves the marks out, which cut every word of Hindi or
# vowelled Arabic into its consonants — "मेरी बहन" read as "म र बहन" — so two
# sentences sharing one letter shared a "word".
_WORD = re.compile(f"[\\w{_combining_marks()}]+")

# Han ideographs (with extensions A–F and the compatibility block),
# hiragana, katakana (with its phonetic extensions and the halfwidth forms).
# Hangul is spaced between words and is left to the word pattern.
_UNSPACED = (
    "[\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff\u3400-\u4dbf"
    "\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\U00020000-\U0003134f]"
)
_UNSPACED_RUN = re.compile(f"{_UNSPACED}+")

# Below this many words in common, overlap proves nothing: two three-word
# fragments can share every word by accident.
_MIN_COMPARABLE = 3


def _bigrams(run: str) -> list[str]:
    if len(run) < 2:
        return [run]
    return [run[i:i + 2] for i in range(len(run) - 1)]


def _segment(token: str) -> list[str]:
    """A word with every unspaced-script run cut into bigrams."""
    if not _UNSPACED_RUN.search(token):
        return [token]
    out: list[str] = []
    pos = 0
    for match in _UNSPACED_RUN.finditer(token):
        if match.start() > pos:
            out.append(token[pos:match.start()])
        out.extend(_bigrams(match.group()))
        pos = match.end()
    if pos < len(token):
        out.append(token[pos:])
    return out


def words(text: str) -> list[str]:
    """Every word in the text, in order, case-folded; unspaced scripts as bigrams."""
    return [piece for token in _WORD.findall(text.casefold()) for piece in _segment(token)]


def lexical_terms(text: str) -> list[str]:
    """Every word in the text, in order, as the lexical index must be asked
    for it: unfolded, unspaced scripts as bigrams.

    FTS5 passes each quoted query term through the index's own tokenizer,
    which folds it exactly as it folded the indexed text. Folding first in
    Python is what lost the match: ``casefold`` turns "Hauptstraße" into
    "hauptstrasse", "ﬁnal" into "final" and "İstanbul" into an "i" and a
    combining dot, none of which the index holds.
    """
    return [piece for token in _WORD.findall(text) for piece in _segment(token)]


def lexical_text(text: str) -> str:
    """The text as the lexical index should see it.

    The FTS5 tokenizer splits on whitespace and punctuation only, so an
    unspaced-script run is written to the index as its space-separated
    bigrams — the same pieces :func:`words` hands the query side. A text
    with no such run is returned unchanged.
    """
    if not text or not _UNSPACED_RUN.search(text):
        return text
    return _UNSPACED_RUN.sub(lambda m: " " + " ".join(_bigrams(m.group())) + " ", text)


def word_set(text: str) -> set[str]:
    return set(words(text))


def comparable(a: set[str], b: set[str]) -> bool:
    """Whether two word sets are long enough for their overlap to prove anything."""
    return min(len(a), len(b)) >= _MIN_COMPARABLE


def containment(a: set[str], b: set[str]) -> float:
    """Shared words as a fraction of the smaller set.

    Containment rather than Jaccard: a restatement may add a word or drop
    one, and measuring against the longer side would score that as different.
    Returns 0 for a pair too short to judge.
    """
    if not comparable(a, b):
        return 0.0
    return len(a & b) / min(len(a), len(b))


def coverage(a: set[str], b: set[str]) -> float:
    """Shared words as a fraction of the larger set.

    The stricter measure, for a judgement that has to be sure: a text that
    covers the whole of a short query is not thereby a copy of it. "the
    deploy pipeline uses Jenkins" contains every word of "deploy pipeline
    Jenkins" and is the answer, not an echo; measured against the larger
    side it shares three words of five, and is told apart. Whether the pair
    is long enough for the fraction to prove anything is the caller's
    judgement — see :func:`comparable`.
    """
    larger = max(len(a), len(b))
    return len(a & b) / larger if larger else 0.0
