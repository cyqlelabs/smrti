"""Coverage tests for viz_routes endpoints: llm-calls, clear, graph edges."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from smrti import Smrti
import smrti.call_log as call_log


@pytest.fixture(scope="module")
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def mem_instance(db_path):
    engine = Smrti(db_path=db_path, tenant_id="default", write_space="default")
    yield engine
    engine.close()


async def _noop_reflect(*_args, **_kwargs):
    return


@pytest.fixture(scope="module")
def client(mem_instance):
    from smrti.servers import rest as rest_mod
    with patch.object(rest_mod, "get_mem", return_value=mem_instance):
        with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
            with TestClient(rest_mod.app, raise_server_exceptions=True) as c:
                yield c


@pytest.fixture(autouse=True)
def reset_call_log():
    call_log._CALL_LOG.clear()
    yield
    call_log._CALL_LOG.clear()


# ── /llm-calls GET ─────────────────────────────────────────────────────────

def test_get_llm_calls_empty(client):
    resp = client.get("/llm-calls")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_llm_calls_returns_entries(client):
    call_log.append({"kind": "extraction", "model": "gpt-4o"})
    resp = client.get("/llm-calls")
    assert resp.status_code == 200
    entries = resp.json()
    assert any(e.get("kind") == "extraction" for e in entries)


# ── /llm-calls DELETE ──────────────────────────────────────────────────────

def test_clear_llm_calls(client):
    call_log.append({"kind": "proxy"})
    resp = client.delete("/llm-calls")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert call_log.get_all() == []


# ── /graph with edges ──────────────────────────────────────────────────────

def test_get_graph_with_edges(client, mem_instance):
    """Graph endpoint must include relation edges when linked atoms are present."""
    a = mem_instance.remember("Edge source atom")
    b = mem_instance.remember("Edge target atom")
    mem_instance.atomspace.link_atoms(a, b, "linked_to", mem_instance.tenant_id, mem_instance.write_space)

    resp = client.get("/graph?tenant_id=default&space=default&limit=100")
    assert resp.status_code == 200
    data = resp.json()
    assert "edges" in data
    # At least one edge should be present since we linked two atoms
    edge_relations = [e.get("relation") for e in data["edges"]]
    assert "linked_to" in edge_relations


# ── /atoms/{atom_id} 404 ──────────────────────────────────────────────────

def test_get_atom_found_and_404(client, mem_instance):
    atom_id = mem_instance.remember("Findable viz atom")
    resp = client.get(f"/atoms/{atom_id}")
    assert resp.status_code == 200

    resp404 = client.get("/atoms/nonexistent-00000000")
    assert resp404.status_code == 404
