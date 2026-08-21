"""Tests for MCP server handle_tool dispatch (servers/mcp.py)."""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from smrti import Smrti
from smrti.core.provenance import VALENCE_STATED
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


def test_remember_explicit_zero_valence_skips_sentiment_estimate(mem):
    """Explicit neutral valence (0.0) is respected — only absence triggers estimation.

    Estimation lives on the engine now, so the patch targets it there; the
    server forwards the caller's value, or None, without reading it.
    """
    with patch("smrti.estimate_valence", return_value=0.3) as mock_est:
        result = handle_tool(mem, "smrti_remember", {"content": "test", "valence": 0.0})
    mock_est.assert_not_called()
    assert result["status"] == "ok"
    assert VALENCE_STATED in _metadata(mem, result["atom_id"])


def test_remember_none_valence_triggers_sentiment_estimate(mem):
    with patch("smrti.estimate_valence", return_value=-0.2) as mock_est:
        result = handle_tool(mem, "smrti_remember", {"content": "test"})
    mock_est.assert_called_once()
    assert VALENCE_STATED not in _metadata(mem, result["atom_id"])


def _metadata(mem, atom_id):
    return mem.db.fetchone("SELECT metadata FROM atoms WHERE id = ?", (atom_id,))["metadata"]


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


# ── source provenance ─────────────────────────────────────────────────────────

def test_remember_stamps_agent_source(mem):
    """MCP clients need a way to say a memory is the model's own output.

    Without it every call defaults to "user" and model chatter is stored with
    the same standing as what the user actually said.
    """
    result = handle_tool(
        mem, "smrti_remember", {"content": "Here are some ideas.", "source": "agent"}
    )
    row = mem.db.fetchone("SELECT metadata FROM atoms WHERE id = ?", (result["atom_id"],))
    assert json.loads(row["metadata"])["source"] == "agent"


def test_remember_defaults_to_user_source(mem):
    """Omitting source must behave exactly as before this parameter existed."""
    result = handle_tool(mem, "smrti_remember", {"content": "I use Kubernetes."})
    row = mem.db.fetchone("SELECT metadata FROM atoms WHERE id = ?", (result["atom_id"],))
    assert json.loads(row["metadata"]) == {}


def test_remember_source_is_advertised_in_the_tool_schema():
    """An undocumented parameter is one no MCP client will ever send."""
    from smrti.servers.tools import TOOLS

    schema = next(t for t in TOOLS if t["name"] == "smrti_remember")["inputSchema"]
    source = schema["properties"]["source"]
    assert source["enum"] == ["user", "agent"]
    assert source["default"] == "user"
    assert "source" not in schema["required"]
