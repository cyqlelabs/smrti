"""Regression tests for the extraction half of the audit: date resolution,
the extraction endpoint's credentials, tone writes, supersession and entity
resolution."""
from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from smrti import Smrti
from smrti.extraction import extract, temporal
from smrti.extraction.temporal import annotate, resolve_spans

BASE = datetime(2026, 11, 25, 15, 0, 0)


@pytest.fixture
def mem(tmp_path):
    return Smrti(db_path=str(tmp_path / "memory.db"), tenant_id="t", write_space="s")


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


# ── the extraction endpoint ──────────────────────────────────────────────────


def _chat_request(monkeypatch, upstream, auth, server_key="", extract_key=""):
    """The (url, Authorization) one extraction call sends."""
    from smrti.servers import config as cfg

    monkeypatch.setattr(cfg, "API_KEY", server_key)
    monkeypatch.setattr(extract, "_EXTRACT_API_KEY", extract_key)
    seen = []

    async def endpoint(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    async def call():
        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
            await extract._post_chat(client, upstream, auth, {"messages": []})

    asyncio.run(call())
    return seen[0]


def test_the_smrti_server_key_is_never_forwarded_to_the_extraction_endpoint(monkeypatch):
    """A client that authenticated to Smrti with its key had that key posted
    to whatever endpoint extraction pointed at."""
    _, sent = _chat_request(monkeypatch, "http://llm", "Bearer server-key", server_key="server-key")
    assert sent is None
    _, sent = _chat_request(monkeypatch, "http://llm", "Bearer upstream-key", server_key="server-key")
    assert sent == "Bearer upstream-key"


def test_an_extraction_api_key_is_the_endpoint_s_credential(monkeypatch):
    _, sent = _chat_request(
        monkeypatch, "http://llm", "Bearer upstream-key", server_key="server-key", extract_key="ek",
    )
    assert sent == "Bearer ek"


def test_an_upstream_that_ends_in_v1_is_not_given_a_second_one(monkeypatch):
    """The documented Docker example names .../v1, and the call went to /v1/v1."""
    url, _ = _chat_request(monkeypatch, "http://host:11434/v1/", "")
    assert url == "http://host:11434/v1/chat/completions"
    url, _ = _chat_request(monkeypatch, "http://host:11434", "")
    assert url == "http://host:11434/v1/chat/completions"


def test_local_mode_calls_no_model_for_an_agent_turn(mem):
    """Agent turns took the full LLM path whatever the mode, sending the
    assistant's text to the configured upstream in a mode promised local."""
    episode = mem.remember("I recommend Docker for the project.", metadata={"source": "agent"})
    ner = MagicMock()
    ner.extract.return_value = [{"name": "Docker", "type": "technology"}]
    post = AsyncMock()
    with patch("smrti.extraction.ner.get_ner", return_value=ner), patch.object(extract, "_post_chat", post):
        asyncio.run(extract.extract_and_link_hybrid(
            episode, "I recommend Docker for the project.", mem, "", "m", "http://llm",
            source="agent", mode="local",
        ))
    assert post.await_count == 0
    assert mem.db.fetchone("SELECT 1 FROM atoms WHERE label = 'Docker'") is not None


def test_a_failed_extraction_call_is_logged_as_a_warning(caplog):
    """Failures went only to the in-process call log, so an endpoint failing
    every call left nothing in the server's own log."""
    post = AsyncMock(return_value=httpx.Response(503))
    with patch.object(extract, "_post_chat", post), caplog.at_level("WARNING", logger="smrti.extract"):
        result = asyncio.run(extract.extract_knowledge("Alice met Bob", None, "http://llm", "", "m"))
    assert result is None
    assert any("HTTP 503" in record.getMessage() for record in caplog.records)
