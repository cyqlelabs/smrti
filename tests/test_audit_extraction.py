"""Regression tests for the extraction half of the audit: date resolution,
the extraction endpoint's credentials, tone writes, supersession and entity
resolution."""
from __future__ import annotations

from datetime import datetime

from smrti.extraction import temporal
from smrti.extraction.temporal import annotate, resolve_spans

BASE = datetime(2026, 11, 25, 15, 0, 0)


class SpanNER:
    """Tags the given spans wherever they occur in the text."""

    def __init__(self, spans: list[tuple[str, str]]) -> None:
        self._spans = spans

    def extract(self, text, labels=None, threshold=0.4):
        return [
            {"name": name, "type": label, "score": 0.99}
            for name, label in self._spans
            if name in text
        ]


# ── date resolution ──────────────────────────────────────────────────────────


def test_the_date_parser_is_built_once_and_resolves_against_each_call_s_base():
    """A parser built per span cost about a second a span; the reused one must
    still read "tomorrow" against the write time of the call it serves."""
    ner = SpanNER([("tomorrow", "date")])

    first = resolve_spans("The review is tomorrow", BASE, ner)
    parser = temporal._parser
    second = resolve_spans("The review is tomorrow", datetime(2020, 2, 28, 9), ner)

    assert first == [("tomorrow", "2026-11-26")]
    assert second == [("tomorrow", "2020-02-29")]
    assert temporal._parser is parser


def test_resolving_a_year_twice_nests_no_annotation():
    """"2019" occurs again inside its own annotation; the second run annotated
    that occurrence and nested a new annotation inside the old one."""
    ner = SpanNER([("2019", "date")])
    stored = "I moved to Berlin in 2019 [resolved: 2019-11-25]."

    assert annotate(stored, BASE, ner) == stored
    assert annotate(annotate("I moved to Berlin in 2019.", BASE, ner), BASE, ner) == (
        "I moved to Berlin in 2019."
    )


def test_a_month_or_a_year_is_never_given_a_day():
    """"March" became the 25th of March and "2019" the 25th of November 2019."""
    for span in ("March", "2019", "last year"):
        assert resolve_spans(f"It happened {span}", BASE, SpanNER([(span, "date")])) == []


def test_a_duration_without_a_direction_is_not_resolved_into_the_past():
    """The tagger cut "in two weeks" to "two weeks", which the parser read as
    two weeks ago."""
    ner = SpanNER([("two weeks", "date")])

    assert resolve_spans("She turns 30 in two weeks", BASE, ner) == []
    assert resolve_spans(
        "She turns 30 in two weeks", BASE, SpanNER([("in two weeks", "date")])
    ) == [("in two weeks", "2026-12-09")]


def test_a_numeric_date_is_resolved_only_when_its_numbers_settle_the_order():
    """"12/03/1990" was read month-first whatever language it was written in."""
    assert resolve_spans("Born on 12/03/1990", BASE, SpanNER([("12/03/1990", "date")])) == []
    assert resolve_spans(
        "Born on 25/03/1990", BASE, SpanNER([("25/03/1990", "date")])
    ) == [("25/03/1990", "1990-03-25")]
