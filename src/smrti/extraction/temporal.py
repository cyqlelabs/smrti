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
"""
from __future__ import annotations

import re
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


def _resolve(span: str, base: datetime) -> datetime | None:
    try:
        import dateparser

        return dateparser.parse(
            span,
            settings={"RELATIVE_BASE": base, "RETURN_AS_TIMEZONE_AWARE": False},
        )
    except Exception:
        return None


def _append(text: str, span: str, annotation: str) -> str:
    """Annotate the first occurrence of *span* that is not annotated already."""
    if not span:
        return text
    index = 0
    while True:
        index = text.find(span, index)
        if index < 0:
            return text
        end = index + len(span)
        if _ANNOTATION.match(text, end):
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
        parsed = _resolve(span, base)
        if parsed is None:
            continue
        iso = parsed.date().isoformat()
        # A span that already spells the date out gains nothing from being
        # told what it says.
        if iso in span:
            continue
        resolved.append((span, iso))
    return resolved


def annotate(text: str, base: datetime | None = None, ner=None) -> str:
    """Return *text* with each resolvable date span followed by its date.

    Idempotent: spans are detected against the text stripped of any
    annotations it already carries, and an occurrence that is already
    annotated is skipped rather than annotated twice.
    """
    if not text:
        return text
    try:
        probe = _ANNOTATION.sub("", text)
        for span, iso in resolve_spans(probe, base, ner):
            text = _append(text, span, f" [resolved: {iso}]")
        return text
    except Exception:
        return text
