"""Resolving relative time expressions against the moment a memory was written.

"La sesión es mañana a las 3", read back a week later, says a session is
tomorrow. The memory is not stale — its deixis is. The engine knows when the
episode was written; the text just never says so.

Two mechanisms the engine already trusts do the work, and neither knows a
word of any language. GLiNER2 tags the temporal spans zero-shot, from a label
list that is configuration rather than code, so the model's multilingual
coverage is the supported-language list. dateparser turns a tagged span into
a date relative to the write time, across roughly 200 locales, which is that
library's entire job.

Both have to agree before anything is written, and only a span the tagger
called a *date* is ever annotated:

* A span nobody could resolve is left exactly as written. A miss costs the
  reader nothing; a wrong date actively misleads.
* A ``time`` span is never annotated. A clock time carries no calendar deixis
  to resolve, and where the parser does shift the date for one it is guessing
  a day number out of a bare hour — "a las 3" read as the 3rd of the month.
  This is also the "por la mañana" (in the morning) versus "mañana"
  (tomorrow) trap: the tagger separates them cleanly, and no word list is
  involved in either language.
* The resolution is appended, never substituted, so the reader still sees
  what was actually said.
* A span that leaves its day open is left alone. "2019" and "March" name no
  day, "two weeks" does not say which way, and "12/03/1990" does not say which
  number is the month; the parser would settle each with a default nobody
  stated.
"""
from __future__ import annotations

import re
import threading
from datetime import datetime, timezone

# Zero-shot labels for the temporal pass. Deliberately not part of the entity
# label list: an entity pass that emitted these would mint a concept atom
# called "mañana", and it runs in the background long after the text has been
# embedded, which is too late to change what was embedded.
TEMPORAL_LABELS = ["date", "time"]

# Only a span the tagger is reasonably sure about. The same floor the entity
# pass uses.
_THRESHOLD = 0.4

_ANNOTATION = re.compile(r"\s*\[resolved: \d{4}-\d{2}-\d{2}\]")

# The two readings every span is put through. They differ in the two things a
# span can leave open: whether an undirected expression points back or ahead
# ("two weeks", "March", "Tuesday"), and which number of a numeric date is the
# day ("12/03/1990"). A span that settles both itself ("tomorrow", "in two
# weeks", "25/03/1990") reads the same both ways and is resolved; one that
# does not reads two ways and is left alone. No word of any language decides
# it.
_READINGS = (("past", "DMY"), ("future", "MDY"))

# The periods dateparser reports for a span that names a day. A month or a
# year ("March", "2019", "last year") names none, and the parser fills the gap
# with today's day of the month: "2019" became the 25th of November 2019. A
# count of weeks from the write time ("in two weeks") does land on a day.
_DAY_PERIODS = frozenset({"day", "week"})

# One parser for the life of the process, with the write time, direction and
# field order set on its settings per call. dateparser keys its per-locale
# caches on the settings object a parser was built with, and the write time
# is a setting: a parser built per span with the base baked in rebuilt every
# locale's caches, about a second a span, where the reused one answers in a
# few milliseconds. Setting the fields per call is why every parse holds the
# lock.
_parser = None
_parser_settings = None
_parser_lock = threading.Lock()


def _now() -> datetime:
    """The write time, in the clock the atom's ``created_at`` is recorded in."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _detect(text: str, ner) -> list[tuple[str, str]]:
    """(span, label) pairs the tagger found, or nothing if it could not run.

    A missing model, a failed download, a build without the runtime: every one
    of them means the text goes in unannotated, which is what it did before
    this module existed.
    """
    try:
        spans = ner.extract(text, TEMPORAL_LABELS, threshold=_THRESHOLD)
    except Exception:
        return []
    return [
        (span["name"], span["type"])
        for span in spans
        if span.get("name") and span.get("type") in TEMPORAL_LABELS
    ]


def _get_parser():
    global _parser, _parser_settings
    if _parser is None:
        from dateparser.conf import settings as defaults
        from dateparser.date import DateDataParser

        prefer, order = _READINGS[0]
        # DATE_ORDER is passed as a caller's setting so the order set per
        # call is the one used, rather than the order of whichever locale
        # happens to match a string of digits.
        chosen = {"PREFER_DATES_FROM": prefer, "DATE_ORDER": order, "RETURN_AS_TIMEZONE_AWARE": False}
        _parser_settings = defaults.replace(mod_settings=chosen, **chosen)
        _parser = DateDataParser(settings=_parser_settings)
    return _parser, _parser_settings


def _resolve(span: str, base: datetime) -> str | None:
    """The ISO date of the day *span* names, or None unless it names one for sure."""
    days: set[str] = set()
    try:
        with _parser_lock:
            parser, settings = _get_parser()
            settings.RELATIVE_BASE = base
            for prefer, order in _READINGS:
                settings.PREFER_DATES_FROM = prefer
                settings.DATE_ORDER = order
                data = parser.get_date_data(span)
                if data.date_obj is None or data.period not in _DAY_PERIODS:
                    return None
                days.add(data.date_obj.date().isoformat())
    except Exception:
        return None
    return days.pop() if len(days) == 1 else None


def _append(text: str, span: str, annotation: str) -> str:
    """Annotate the first occurrence of *span* that is not annotated already.

    An occurrence inside an annotation is not one: "2019" occurs again in
    "[resolved: 2019-11-25]", and annotating it there nested a new annotation
    inside the old one on every run.
    """
    if not span:
        return text
    annotated = [match.span() for match in _ANNOTATION.finditer(text)]
    index = 0
    while True:
        index = text.find(span, index)
        if index < 0:
            return text
        end = index + len(span)
        inside = any(start <= index < stop for start, stop in annotated)
        if inside or _ANNOTATION.match(text, end):
            index = end
            continue
        return text[:end] + annotation + text[end:]


def resolve_spans(
    text: str, base: datetime | None = None, ner=None
) -> list[tuple[str, str]]:
    """(span, ISO date) for every temporal span both tiers agree on."""
    base = base or _now()
    if ner is None:
        from smrti.extraction.ner import get_ner

        ner = get_ner()

    resolved: list[tuple[str, str]] = []
    for span, label in _detect(text, ner):
        if label != "date":
            continue
        iso = _resolve(span, base)
        if iso is None:
            continue
        # A span that already spells the date out gains nothing from being
        # told what it says.
        if iso in span:
            continue
        resolved.append((span, iso))
    return resolved


def resolve(
    text: str, base: datetime | None = None, ner=None
) -> tuple[str, list[dict]]:
    """*text* with each resolvable date span annotated, and the resolutions.

    The second value is the list recall renders from, in the same shape the
    extraction model produces (``{"text": span, "resolved": ISO date}``), so
    both tiers file into one place: the write-time resolution is authoritative
    for the spans it handled, and the model only adds spans the parser could
    not reach. Before this the two tiers wrote to two places and could carry
    two different dates for one span.

    Idempotent: spans are detected against the text stripped of any
    annotations it already carries, and an occurrence that is already
    annotated is skipped rather than annotated twice.
    """
    if not text:
        return text, []
    try:
        probe = _ANNOTATION.sub("", text)
        items: list[dict] = []
        for span, iso in resolve_spans(probe, base, ner):
            text = _append(text, span, f" [resolved: {iso}]")
            items.append({"text": span, "resolved": iso})
        return text, items
    except Exception:
        return text, []


def annotate(text: str, base: datetime | None = None, ner=None) -> str:
    """Return *text* with each resolvable date span followed by its date."""
    return resolve(text, base, ner)[0]
