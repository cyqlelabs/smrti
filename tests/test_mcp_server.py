"""Tests for the MCP space tools and the stdio server loop (servers/mcp.py).

`test_mcp.py` covers the remember/recall/personality handlers; this module
covers the space-set tools, the legacy aliases kept for REST callers, and
`run_mcp_server` itself, which is driven with a stand-in Server so the
registered handlers can be invoked without a real stdio transport.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from smrti import Smrti
from smrti.servers.mcp import handle_tool, run_mcp_server


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def mem(db_path):
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()


@pytest.fixture
def two_spaces(mem, db_path):
    """Two spaces sharing one near-identical atom so overlap is non-empty."""
    mem.remember("Kubernetes deployments roll out gradually", type="concept")
    mem.remember("Espresso machines need descaling", type="concept")
    other = Smrti(db_path=db_path, tenant_id="test", write_space="other")
    other.remember("Kubernetes deployments roll out gradually", type="concept")
    yield mem
    other.close()


# ── smrti_remember: belief branch ─────────────────────────────────────────────

def test_remember_with_belief_type_records_evidence(mem):
    result = handle_tool(mem, "smrti_remember", {
        "content": "Retries hide flaky tests.",
        "type": "belief",
        "probability": 0.9,
        "evidence": "observed twice in CI",
    })
    assert result["status"] == "ok"
    row = mem.db.fetchone("SELECT type FROM atoms WHERE id = ?", (result["atom_id"],))
    assert row["type"] == "belief"


# ── smrti_space_query ─────────────────────────────────────────────────────────

def test_space_query_overlap(two_spaces):
    result = handle_tool(two_spaces, "smrti_space_query", {
        "op": "overlap", "other_space": "other", "threshold": 0.7,
    })
    assert result["space_a"] == "default"
    assert result["space_b"] == "other"
    assert result["jaccard"] > 0
    pair = result["matched_pairs"][0]
    assert pair["atom_a"]["space"] == "default"
    assert pair["atom_b"]["space"] == "other"
    assert pair["similarity"] > 0


def test_space_query_intersection(two_spaces):
    result = handle_tool(two_spaces, "smrti_space_query", {
        "op": "intersection", "other_space": "other", "threshold": 0.7,
    })
    assert result["operation"] == "intersection"
    assert result["spaces"] == ["default", "other"]
    assert result["jaccard"] > 0
    assert all({"id", "label", "type", "space"} <= set(a) for a in result["atoms"])


def test_space_query_diff(two_spaces):
    result = handle_tool(two_spaces, "smrti_space_query", {
        "op": "diff", "other_space": "other", "threshold": 0.7,
    })
    assert result["operation"] == "difference"
    labels = [a["label"] for a in result["atoms"]]
    assert any("Espresso" in label for label in labels)


def test_space_query_rejects_unknown_op(mem):
    result = handle_tool(mem, "smrti_space_query", {"op": "nonsense", "other_space": "other"})
    assert result["error"] == "Unknown op: nonsense"


# ── legacy space handlers ─────────────────────────────────────────────────────

def test_legacy_space_overlap(two_spaces):
    result = handle_tool(two_spaces, "smrti_space_overlap", {
        "other_space": "other", "threshold": 0.7,
    })
    assert result["jaccard"] > 0
    assert result["matched_pairs"]


def test_legacy_space_intersection(two_spaces):
    result = handle_tool(two_spaces, "smrti_space_intersection", {"other_space": "other"})
    assert result["operation"] == "intersection"
    assert "jaccard" in result


def test_legacy_space_diff(two_spaces):
    result = handle_tool(two_spaces, "smrti_space_diff", {"other_space": "other"})
    assert result["operation"] == "difference"


def test_space_merge_reports_the_bridge_space(two_spaces):
    result = handle_tool(two_spaces, "smrti_space_merge", {
        "other_space": "other", "threshold": 0.7, "min_jaccard": 0.01,
    })
    assert result["status"] == "ok"
    assert result["bridge_space"] == "default_x_other"
    assert result["bridges_created"] >= 1


def test_list_spaces(two_spaces):
    result = handle_tool(two_spaces, "smrti_list_spaces", {})
    assert "default" in result["spaces"]
    assert "other" in result["spaces"]


def test_unknown_tool_returns_error(mem):
    assert handle_tool(mem, "smrti_nope", {}) == {"error": "Unknown tool: smrti_nope"}


# ── run_mcp_server ────────────────────────────────────────────────────────────

class _FakeServer:
    """Captures the handlers `run_mcp_server` registers and replays them."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.handlers: dict = {}
        self.calls: list = []

    def _register(self, key):
        def decorator(fn):
            self.handlers[key] = fn
            return fn
        return decorator

    def list_tools(self):
        return self._register("list_tools")

    def call_tool(self):
        return self._register("call_tool")

    def create_initialization_options(self):
        return {"init": True}

    async def run(self, read_stream, write_stream, options):
        self.calls.append((read_stream, write_stream, options))
        self.listed = await self.handlers["list_tools"]()
        self.called = await self.handlers["call_tool"](
            "smrti_remember", {"content": "Deployed the new indexer."}
        )


@contextlib.asynccontextmanager
async def _fake_stdio():
    yield ("read-stream", "write-stream")


async def _noop_reflect(*_args, **_kwargs):
    return


def _drive_server(mem, extract: bool):
    server = _FakeServer("smrti")
    with patch("smrti.servers.mcp.Server", return_value=server), \
            patch("smrti.servers.mcp.create_smrti", return_value=mem), \
            patch("smrti.servers.mcp.stdio_server", _fake_stdio), \
            patch("smrti.servers.mcp.run_reflect_loop", new=_noop_reflect), \
            patch("smrti.servers.config.EXTRACT", extract):
        run_mcp_server()
    return server


def test_run_mcp_server_advertises_tools_and_dispatches(mem):
    server = _drive_server(mem, extract=False)
    assert server.calls == [("read-stream", "write-stream", {"init": True})]
    names = {t.name for t in server.listed}
    assert "smrti_remember" in names and "smrti_recall" in names
    assert all(t.description for t in server.listed)

    payload = json.loads(server.called[0].text)
    assert payload["status"] == "ok"
    assert mem.db.fetchone(
        "SELECT id FROM atoms WHERE id = ?", (payload["atom_id"],)
    ) is not None


def test_run_mcp_server_extracts_in_the_background_when_enabled(mem):
    seen = {}

    async def _fake_extract(episode_id, content, *args, **kwargs):
        seen["episode_id"] = episode_id
        seen["content"] = content

    with patch("smrti.extraction.extract.extract_and_link_serialized", new=_fake_extract):
        server = _drive_server(mem, extract=True)

    payload = json.loads(server.called[0].text)
    assert seen["episode_id"] == payload["atom_id"]
    assert seen["content"] == "Deployed the new indexer."


def test_run_mcp_server_skips_extraction_for_ignored_content(mem):
    calls = []

    async def _fake_extract(*args, **kwargs):
        calls.append(args)

    server = _FakeServer("smrti")

    async def _run(read_stream, write_stream, options):
        server.called = await server.handlers["call_tool"]("smrti_status", {})

    server.run = _run
    with patch("smrti.servers.mcp.Server", return_value=server), \
            patch("smrti.servers.mcp.create_smrti", return_value=mem), \
            patch("smrti.servers.mcp.stdio_server", _fake_stdio), \
            patch("smrti.servers.mcp.run_reflect_loop", new=_noop_reflect), \
            patch("smrti.extraction.extract.extract_and_link_serialized", new=_fake_extract), \
            patch("smrti.servers.config.EXTRACT", True):
        run_mcp_server()

    assert calls == []
    assert "spaces" in json.loads(server.called[0].text)


def test_run_mcp_server_cancels_the_reflect_loop_on_exit(mem):
    started = asyncio.Event()

    async def _blocking_reflect(*_args, **_kwargs):
        started.set()
        await asyncio.sleep(3600)

    server = _FakeServer("smrti")

    async def _run(read_stream, write_stream, options):
        await started.wait()

    server.run = _run
    with patch("smrti.servers.mcp.Server", return_value=server), \
            patch("smrti.servers.mcp.create_smrti", return_value=mem), \
            patch("smrti.servers.mcp.stdio_server", _fake_stdio), \
            patch("smrti.servers.mcp.run_reflect_loop", new=_blocking_reflect), \
            patch("smrti.servers.config.EXTRACT", False):
        run_mcp_server()  # returns only once the reflect task is cancelled
