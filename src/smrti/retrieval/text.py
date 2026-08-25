"""Word-set view of a text, shared by everything that compares two of them.

Language-agnostic by construction: ``\\w+`` over a case-folded string keeps
letters and digits in any script and drops everything else, so the same
function segments Spanish, German and Japanese without knowing which it is
looking at.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"\w+")

# Below this many words in common, overlap proves nothing: two three-word
# fragments can share every word by accident.
_MIN_COMPARABLE = 3


def words(text: str) -> list[str]:
    """Every word in the text, in order, case-folded."""
    return _WORD.findall(text.casefold())


def word_set(text: str) -> set[str]:
    return set(words(text))


def containment(a: set[str], b: set[str]) -> float:
    """Shared words as a fraction of the smaller set.

    Containment rather than Jaccard: a restatement may add a word or drop
    one, and measuring against the longer side would score that as different.
    Returns 0 for a pair too short to judge.
    """
    smaller = min(len(a), len(b))
    if smaller < _MIN_COMPARABLE:
        return 0.0
    return len(a & b) / smaller
