"""Tests for NER provider and hybrid extraction dispatch."""
import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smrti import Smrti


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


# ── NERProvider unit tests ────────────────────────────────────────────────────


def test_ner_extract_deduplicates():
    """One entry per (name_lower, type) pair."""
    from smrti.extraction.ner import NERProvider

    provider = NERProvider()
    mock_model = MagicMock()
    mock_model.extract_entities.return_value = {
        "entities": {
            "tool": ["Python", "python", "Django"],
        }
    }
    provider._model = mock_model
    provider._has_classify = False

    results = provider.extract("I use Python and Django")
    assert len(results) == 2
    names = {r["name"] for r in results}
    # First occurrence wins (python deduped with Python)
    assert "Django" in names


def test_ner_extract_custom_labels():
    """Custom labels are forwarded to GLiNER2."""
    from smrti.extraction.ner import NERProvider

    provider = NERProvider()
    mock_model = MagicMock()
    mock_model.extract_entities.return_value = {
        "entities": {"city": ["Berlin"]}
    }
    provider._model = mock_model
    provider._has_classify = False

    results = provider.extract("I live in Berlin", labels=["city"])
    mock_model.extract_entities.assert_called_once_with(
        "I live in Berlin", ["city"]
    )
    assert len(results) == 1
    assert results[0]["type"] == "city"


def test_ner_extract_empty_text():
    """Empty text returns empty results."""
    from smrti.extraction.ner import NERProvider

    provider = NERProvider()
    mock_model = MagicMock()
    mock_model.extract_entities.return_value = {"entities": {}}
    provider._model = mock_model
    provider._has_classify = False

    results = provider.extract("")
    assert results == []


def test_classify_pronoun_with_classify_text():
    """classify_pronoun delegates to model.classify_text when available."""
    from smrti.extraction.ner import NERProvider

    provider = NERProvider()
    mock_model = MagicMock()
    mock_model.classify_text.return_value = {"type": "pronoun"}
    provider._model = mock_model
    provider._has_classify = True

    assert provider.classify_pronoun("I") is True
    assert provider.classify_pronoun("my") is True

    mock_model.classify_text.return_value = {"type": "proper_name"}
    assert provider.classify_pronoun("Elara") is False


def test_classify_pronoun_without_classify_text():
    """classify_pronoun returns False when model lacks classify_text."""
    from smrti.extraction.ner import NERProvider

    provider = NERProvider()
    mock_model = MagicMock(spec=[])  # no classify_text
    provider._model = mock_model
    provider._has_classify = False

    assert provider.classify_pronoun("I") is False
    assert provider.classify_pronoun("Elara") is False


# ── Hybrid dispatch tests ────────────────────────────────────────────────────


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_mock_ner(extract_return=None, classify_pronoun_return=False):
    """Create a mock NER provider with GLiNER2-compatible API."""
    mock = MagicMock()
    mock.extract.return_value = extract_return or []
    mock.classify_pronoun.return_value = classify_pronoun_return
    return mock


def test_hybrid_single_entity_no_llm(mem):
    """With a single NER entity, no LLM call is made for claims."""
    episode_id = mem.remember("Dave likes Python", type="episode")

    mock_ner = _make_mock_ner([
        {"name": "Dave", "type": "person", "score": 0.9},
    ])

    from smrti.extraction.extract import extract_and_link_hybrid

    with patch("smrti.extraction.ner.get_ner", return_value=mock_ner):
        with patch("smrti.extraction.extract.extract_claims_only") as mock_claims:
            _run(extract_and_link_hybrid(
                episode_id, "Dave likes Python", mem,
                "", "model", "http://localhost", mode="hybrid",
            ))
            mock_claims.assert_not_called()

    # Entity should still be resolved
    row = mem.db.fetchone(
        "SELECT id FROM atoms WHERE LOWER(label) = 'dave' AND tenant_id = 'test'",
        (),
    )
    assert row is not None


def test_hybrid_two_entities_triggers_llm(mem):
    """With 2+ NER entities, LLM is called for claims."""
    episode_id = mem.remember("Dave migrated to GitHub Actions", type="episode")

    mock_ner = _make_mock_ner([
        {"name": "Dave", "type": "person", "score": 0.9},
        {"name": "GitHub Actions", "type": "tool", "score": 0.85},
    ])

    from smrti.extraction.extract import extract_and_link_hybrid

    with patch("smrti.extraction.ner.get_ner", return_value=mock_ner):
        with patch("smrti.extraction.extract.extract_claims_only", new_callable=AsyncMock) as mock_claims:
            mock_claims.return_value = {
                "claims": [
                    {"subject": "Dave", "predicate": "migrated_to", "object": "GitHub Actions"}
                ]
            }
            _run(extract_and_link_hybrid(
                episode_id, "Dave migrated to GitHub Actions", mem,
                "", "model", "http://localhost", mode="hybrid",
            ))
            mock_claims.assert_called_once()

    # Check relation edge was created
    rows = mem.db.fetchall(
        "SELECT * FROM atoms WHERE type = 'relation' AND relation = 'migrated_to' AND tenant_id = 'test'",
        (),
    )
    assert len(rows) >= 1


def test_gliner_failure_falls_back_to_llm_in_hybrid(mem):
    """When GLiNER raises, hybrid mode falls back to full LLM extraction."""
    episode_id = mem.remember("test content", type="episode")

    mock_ner = _make_mock_ner()
    mock_ner.extract.side_effect = RuntimeError("GLiNER failed")

    from smrti.extraction.extract import extract_and_link_hybrid

    with patch("smrti.extraction.ner.get_ner", return_value=mock_ner):
        with patch("smrti.extraction.extract.extract_and_link", new_callable=AsyncMock) as mock_llm:
            _run(extract_and_link_hybrid(
                episode_id, "test content", mem,
                "auth", "model", "http://localhost", mode="hybrid",
            ))
            mock_llm.assert_called_once_with(
                episode_id, "test content", mem,
                "auth", "model", "http://localhost", "user",
            )


def test_local_mode_never_calls_llm(mem):
    """Local mode never calls the LLM, even with 2+ entities."""
    episode_id = mem.remember("Dave and Alice work together", type="episode")

    mock_ner = _make_mock_ner([
        {"name": "Dave", "type": "person", "score": 0.9},
        {"name": "Alice", "type": "person", "score": 0.85},
    ])

    from smrti.extraction.extract import extract_and_link_hybrid

    with patch("smrti.extraction.ner.get_ner", return_value=mock_ner):
        with patch("smrti.extraction.extract.extract_claims_only", new_callable=AsyncMock) as mock_claims:
            with patch("smrti.extraction.extract.extract_and_link", new_callable=AsyncMock) as mock_llm:
                _run(extract_and_link_hybrid(
                    episode_id, "Dave and Alice work together", mem,
                    "", "model", "http://localhost", mode="local",
                ))
                mock_claims.assert_not_called()
                mock_llm.assert_not_called()


def test_agent_source_always_uses_full_llm(mem):
    """source='agent' always takes the full LLM path regardless of mode."""
    from smrti.extraction.extract import extract_and_link_hybrid

    episode_id = mem.remember("agent response", type="episode")

    with patch("smrti.extraction.extract.extract_and_link", new_callable=AsyncMock) as mock_llm:
        _run(extract_and_link_hybrid(
            episode_id, "agent response", mem,
            "auth", "model", "http://localhost",
            source="agent", mode="hybrid",
        ))
        mock_llm.assert_called_once()


def test_llm_mode_uses_full_llm(mem):
    """mode='llm' takes the full LLM path for backward compat."""
    from smrti.extraction.extract import extract_and_link_hybrid

    episode_id = mem.remember("some text", type="episode")

    with patch("smrti.extraction.extract.extract_and_link", new_callable=AsyncMock) as mock_llm:
        _run(extract_and_link_hybrid(
            episode_id, "some text", mem,
            "auth", "model", "http://localhost",
            source="user", mode="llm",
        ))
        mock_llm.assert_called_once()


def test_local_mode_gliner_import_error_noop(mem):
    """Local mode with missing gliner silently no-ops."""
    episode_id = mem.remember("test", type="episode")

    from smrti.extraction.extract import extract_and_link_hybrid

    with patch("smrti.extraction.ner.get_ner", side_effect=ImportError("no gliner")):
        # Should not raise
        _run(extract_and_link_hybrid(
            episode_id, "test", mem,
            "", "model", "http://localhost", mode="local",
        ))
