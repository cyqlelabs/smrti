"""Tests for resolving relative dates against the moment a memory was written.

The tagger is stood in for so the table below is a test of the resolution
rules rather than of a model download: what GLiNER2 returns for these
sentences is checked once against the real model, and pinned here as the
fixture every case runs through. dateparser is real — it is the half these
tests are actually exercising.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from smrti import Smrti
from smrti.extraction.temporal import annotate, resolve_spans

BASE = datetime(2026, 8, 26, 14, 0, 0)


class StubNER:
    """Returns the spans the real tagger returns for these sentences."""

    def __init__(self, spans: list[tuple[str, str]] | None = None) -> None:
        self._spans = spans or []
        self.calls: list[str] = []

    def extract(self, text, labels=None, threshold=0.4):
        self.calls.append(text)
        return [
            {"name": name, "type": label, "score": 0.99}
            for name, label in self._spans
            if name in text
        ]


class BrokenNER:
    def extract(self, text, labels=None, threshold=0.4):
        raise RuntimeError("model could not be loaded")


# ── resolution across languages ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,span,expected",
    [
        # Spanish
        ("La sesión es mañana a las 3", "mañana", "2026-08-27"),
        ("El deploy fue ayer y rompió producción", "ayer", "2026-08-25"),
        ("Nos mudamos hoy", "hoy", "2026-08-26"),
        # English
        ("The session is tomorrow", "tomorrow", "2026-08-27"),
        ("We shipped it yesterday", "yesterday", "2026-08-25"),
        ("The review is in two weeks", "in two weeks", "2026-09-09"),
        # German, to show the parser is not doing English with extra steps
        ("Der Termin ist übermorgen", "übermorgen", "2026-08-28"),
        ("Wir haben es vorgestern deployed", "vorgestern", "2026-08-24"),
        # Japanese — non-Latin script
        ("会議は明日です", "明日", "2026-08-27"),
        ("昨日デプロイした", "昨日", "2026-08-25"),
    ],
)
def test_a_date_span_resolves_against_the_write_time(text, span, expected):
    ner = StubNER([(span, "date")])

    assert resolve_spans(text, BASE, ner) == [(span, expected)]


@pytest.mark.parametrize(
    "text,span,expected",
    [
        ("La sesión es mañana a las 3", "mañana", "mañana [resolved: 2026-08-27]"),
        ("The session is tomorrow", "tomorrow", "tomorrow [resolved: 2026-08-27]"),
        ("会議は明日です", "明日", "明日 [resolved: 2026-08-27]"),
    ],
)
def test_the_resolution_is_appended_to_the_span(text, span, expected):
    annotated = annotate(text, BASE, StubNER([(span, "date")]))

    assert expected in annotated
    assert text.replace(span, expected) == annotated


# ── the ambiguity guards ─────────────────────────────────────────────────────


def test_a_clock_time_is_never_annotated():
    """"por la mañana" is in the morning; "mañana" alone is tomorrow."""
    ner = StubNER([("por la mañana", "time")])

    assert resolve_spans("Nos vemos por la mañana", BASE, ner) == []
    assert annotate("Nos vemos por la mañana", BASE, ner) == "Nos vemos por la mañana"


def test_the_same_word_tagged_as_a_date_does_resolve():
    ner = StubNER([("mañana", "date")])

    assert resolve_spans("Nos vemos mañana", BASE, ner) == [("mañana", "2026-08-27")]


def test_a_bare_hour_is_left_alone_even_though_the_parser_would_shift_the_date():
    """dateparser reads "a las 3" as the 3rd of the month; the tagger calls it a time."""
    ner = StubNER([("a las 3", "time")])

    assert annotate("La reunión es a las 3", BASE, ner) == "La reunión es a las 3"


@pytest.mark.parametrize(
    "text,span",
    [
        # Idioms and weekday references the parser declines. These are the
        # cases the extraction LLM covers later, from the atom's metadata.
        ("Lo vemos el finde que viene", "el finde que viene"),
        ("The review is next Friday", "next Friday"),
        ("Der Termin ist nächsten Freitag", "nächsten Freitag"),
    ],
)
def test_a_span_no_parser_can_resolve_leaves_the_text_untouched(text, span):
    ner = StubNER([(span, "date")])

    assert resolve_spans(text, BASE, ner) == []
    assert annotate(text, BASE, ner) == text


def test_a_span_that_already_spells_the_date_out_gains_no_annotation():
    ner = StubNER([("2026-09-03", "date")])
    text = "La fecha límite es 2026-09-03"

    assert annotate(text, BASE, ner) == text


def test_text_with_no_temporal_span_is_untouched():
    ner = StubNER([])
    text = "I like coffee and long walks"

    assert annotate(text, BASE, ner) == text


def test_a_tagger_that_cannot_run_leaves_the_text_untouched():
    text = "The session is tomorrow"

    assert annotate(text, BASE, BrokenNER()) == text


def test_empty_text_is_returned_as_is():
    assert annotate("", BASE, StubNER([("tomorrow", "date")])) == ""


# ── idempotence ──────────────────────────────────────────────────────────────


def test_annotating_twice_annotates_once():
    ner = StubNER([("mañana", "date")])
    once = annotate("La sesión es mañana", BASE, ner)
    twice = annotate(once, BASE, ner)

    assert once == twice


def test_the_injected_date_is_not_itself_treated_as_a_span():
    """The tagger runs against the text as written, annotations stripped."""
    ner = StubNER([("mañana", "date"), ("2026-08-27", "date")])
    once = annotate("La sesión es mañana", BASE, ner)

    assert annotate(once, BASE, ner) == once
    assert ner.calls[-1] == "La sesión es mañana"


def test_a_repeated_span_gets_one_annotation_per_run():
    ner = StubNER([("mañana", "date")])
    text = "mañana y mañana"

    once = annotate(text, BASE, ner)
    assert once == "mañana [resolved: 2026-08-27] y mañana"

    assert annotate(once, BASE, ner) == (
        "mañana [resolved: 2026-08-27] y mañana [resolved: 2026-08-27]"
    )


# ── the facade ───────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "temporal.db")


def test_the_facade_stores_text_verbatim_by_default(db_path):
    mem = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    atom_id = mem.remember("The session is tomorrow")

    atom = mem.atomspace.get_atom(atom_id, "test", "default")
    assert atom.content == "The session is tomorrow"


def test_the_resolution_lands_in_the_stored_text_before_it_is_embedded(db_path, monkeypatch):
    import smrti.extraction.temporal as temporal_mod

    monkeypatch.setattr(
        temporal_mod, "annotate",
        lambda text, base=None, ner=None: annotate(text, BASE, StubNER([("tomorrow", "date")])),
    )
    mem = Smrti(db_path=db_path, tenant_id="test", write_space="default", temporal=True)
    atom_id = mem.remember("The session is tomorrow")

    atom = mem.atomspace.get_atom(atom_id, "test", "default")
    assert atom.content == "The session is tomorrow [resolved: 2026-08-27]"
    # Embedded and lexically indexed as stored, so the date is searchable.
    row = mem.db.fetchone("SELECT content FROM atoms_fts WHERE atom_id = ?", (atom_id,))
    assert row["content"] == atom.content


def test_beliefs_get_their_deixis_resolved_too(db_path, monkeypatch):
    import smrti.extraction.temporal as temporal_mod

    monkeypatch.setattr(
        temporal_mod, "annotate",
        lambda text, base=None, ner=None: annotate(text, BASE, StubNER([("tomorrow", "date")])),
    )
    mem = Smrti(db_path=db_path, tenant_id="test", write_space="default", temporal=True)
    atom_id = mem.believe("The release ships tomorrow", probability=0.9)

    atom = mem.atomspace.get_atom(atom_id, "test", "default")
    assert atom.content == "The release ships tomorrow [resolved: 2026-08-27]"


def test_a_failing_tagger_does_not_fail_the_write(db_path, monkeypatch):
    import smrti.extraction.ner as ner_mod

    monkeypatch.setattr(ner_mod, "get_ner", lambda: BrokenNER())
    mem = Smrti(db_path=db_path, tenant_id="test", write_space="default", temporal=True)
    atom_id = mem.remember("The session is tomorrow")

    atom = mem.atomspace.get_atom(atom_id, "test", "default")
    assert atom.content == "The session is tomorrow"


def test_server_modes_resolve_deixis_by_default():
    from smrti.servers import config as cfg

    assert cfg.TEMPORAL is True


# ── failing open ─────────────────────────────────────────────────────────────


def test_a_parser_that_raises_resolves_nothing(monkeypatch):
    """dateparser is a dependency, not a promise — a miss must cost nothing."""
    import dateparser

    def _boom(*args, **kwargs):
        raise RuntimeError("locale data missing")

    monkeypatch.setattr(dateparser, "parse", _boom)

    assert resolve_spans("La sesión es mañana", BASE, StubNER([("mañana", "date")])) == []


def test_an_empty_span_is_never_annotated():
    """Scanning for an empty string would never advance and never return."""
    from smrti.extraction.temporal import _append

    assert _append("La sesión es mañana", "", " [resolved: 2026-08-27]") == (
        "La sesión es mañana"
    )


def test_a_failure_anywhere_leaves_the_text_exactly_as_written(monkeypatch):
    import smrti.extraction.temporal as temporal_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("something unforeseen")

    monkeypatch.setattr(temporal_mod, "resolve_spans", _boom)

    assert annotate("The session is tomorrow", BASE) == "The session is tomorrow"
