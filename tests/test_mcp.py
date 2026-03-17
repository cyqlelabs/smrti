"""Tests for MCP server handle_tool dispatch (servers/mcp.py)."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from smrti import Smrti
from smrti.servers.mcp import handle_tool


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


# ── smrti_remember ────────────────────────────────────────────────────────────

def test_remember_returns_atom_id(mem):
    result = handle_tool(mem, "smrti_remember", {"content": "The sky is blue."})
    assert result["status"] == "ok"
    assert result["atom_id"]


def test_remember_with_explicit_valence(mem):
    result = handle_tool(mem, "smrti_remember", {
        "content": "Something happened.",
        "valence": 0.7,
        "type": "episode",
        "probability": 0.9,
    })
    assert result["status"] == "ok"


def test_remember_ignored_content(mem):
    # Patch is_ignored to return True
    with patch.object(mem, "is_ignored", return_value=True):
        result = handle_tool(mem, "smrti_remember", {"content": "ignored"})
    assert result["status"] == "ignored"
    assert result["atom_id"] == ""


def test_remember_zero_valence_triggers_sentiment_estimate(mem):
    with patch("smrti.servers.mcp.estimate_valence", return_value=0.3) as mock_est:
        result = handle_tool(mem, "smrti_remember", {"content": "test", "valence": 0.0})
    mock_est.assert_called_once()
    assert result["status"] == "ok"


def test_remember_none_valence_triggers_sentiment_estimate(mem):
    with patch("smrti.servers.mcp.estimate_valence", return_value=-0.2) as mock_est:
        result = handle_tool(mem, "smrti_remember", {"content": "test"})
    mock_est.assert_called_once()


# ── smrti_recall ──────────────────────────────────────────────────────────────

def test_recall_returns_memories_list(mem):
    mem.remember("Alice prefers Python.")
    result = handle_tool(mem, "smrti_recall", {"query": "Alice Python"})
    assert "memories" in result
    assert isinstance(result["memories"], list)


def test_recall_memory_has_severity_field(mem):
    mem.remember("Critical error in production!", valence=-0.9)
    result = handle_tool(mem, "smrti_recall", {"query": "error production"})
    for m in result["memories"]:
        assert "severity" in m
        assert "salience" in m


def test_recall_empty_when_nothing_stored(mem):
    result = handle_tool(mem, "smrti_recall", {"query": "nonexistent topic xyz"})
    assert result["memories"] == []


# ── smrti_reflect ─────────────────────────────────────────────────────────────

def test_reflect_returns_epoch_result(mem):
    mem.remember("test memory")
    result = handle_tool(mem, "smrti_reflect", {})
    assert "beliefs_updated" in result
    assert "atoms_pruned" in result


# ── smrti_believe ─────────────────────────────────────────────────────────────

def test_believe_returns_atom_id(mem):
    result = handle_tool(mem, "smrti_believe", {
        "statement": "The earth is round.",
        "probability": 0.99,
    })
    assert result["status"] == "ok"
    assert result["atom_id"]


def test_believe_with_evidence(mem):
    result = handle_tool(mem, "smrti_believe", {
        "statement": "Water boils at 100°C.",
        "probability": 0.95,
        "evidence": "Empirical observation",
    })
    assert result["status"] == "ok"


# ── smrti_forget ──────────────────────────────────────────────────────────────

def test_forget_softens_matching_atoms(mem):
    mem.remember("User hates meetings")
    result = handle_tool(mem, "smrti_forget", {"query": "meetings"})
    assert result["status"] == "ok"
    assert "softened" in result
    assert isinstance(result["softened"], list)


def test_forget_empty_when_no_match(mem):
    result = handle_tool(mem, "smrti_forget", {"query": "absolutely nothing here"})
    assert result["softened"] == []


# ── smrti_personality ─────────────────────────────────────────────────────────

def test_personality_get(mem):
    result = handle_tool(mem, "smrti_personality", {"action": "get"})
    assert isinstance(result, dict)
    assert "tenant_id" in result or result == {}


def test_personality_set_preset(mem):
    result = handle_tool(mem, "smrti_personality", {
        "action": "preset",
        "preset": "analytical",
    })
    assert result["status"] == "ok"
    assert result["preset"] == "analytical"


def test_personality_set_action(mem):
    result = handle_tool(mem, "smrti_personality", {
        "action": "set",
        "preset": "curious",
    })
    assert result["status"] == "ok"


# ── smrti_status ──────────────────────────────────────────────────────────────

def test_status_returns_counts(mem):
    mem.remember("some memory")
    result = handle_tool(mem, "smrti_status", {})
    assert "total_atoms" in result
    assert result["total_atoms"] >= 1


# ── unknown tool ─────────────────────────────────────────────────────────────

def test_unknown_tool_returns_error(mem):
    result = handle_tool(mem, "nonexistent_tool", {})
    assert "error" in result
