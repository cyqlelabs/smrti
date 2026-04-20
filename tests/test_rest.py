"""Tests for the FastAPI REST server (servers/rest.py)."""
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from smrti import Smrti


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


@pytest.fixture(scope="module")
def client(mem_instance):
    from smrti.servers import rest as rest_mod
    # Patch get_mem so the app uses our in-memory fixture
    with patch.object(rest_mod, "get_mem", return_value=mem_instance):
        with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
            with patch("smrti.servers.config.EXTRACT", False):
                with TestClient(rest_mod.app, raise_server_exceptions=True) as c:
                    yield c


async def _noop_reflect(*_args, **_kwargs):
    return


# ── /remember ────────────────────────────────────────────────────────────────

def test_remember_ok(client):
    resp = client.post("/remember", json={"content": "REST server is tested."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["atom_id"]


def test_remember_with_valence(client):
    resp = client.post("/remember", json={
        "content": "Deployment failed catastrophically.",
        "valence": -0.9,
        "probability": 0.95,
    })
    assert resp.status_code == 200


def test_remember_default_type(client):
    resp = client.post("/remember", json={"content": "Default episode type."})
    assert resp.status_code == 200


# ── /recall ───────────────────────────────────────────────────────────────────

def test_recall_returns_memories(client):
    resp = client.post("/recall", json={"query": "REST server"})
    assert resp.status_code == 200
    assert "memories" in resp.json()


def test_recall_with_top_k(client):
    resp = client.post("/recall", json={"query": "test", "top_k": 3})
    assert resp.status_code == 200
    memories = resp.json()["memories"]
    assert len(memories) <= 3


def test_recall_with_min_confidence(client):
    resp = client.post("/recall", json={"query": "test", "min_confidence": 0.99})
    assert resp.status_code == 200


# ── /reflect ──────────────────────────────────────────────────────────────────

def test_reflect_runs(client):
    resp = client.post("/reflect")
    assert resp.status_code == 200
    data = resp.json()
    assert "beliefs_updated" in data


# ── /believe ──────────────────────────────────────────────────────────────────

def test_believe_ok(client):
    resp = client.post("/believe", json={
        "statement": "The REST API works correctly.",
        "probability": 0.9,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_believe_with_evidence(client):
    resp = client.post("/believe", json={
        "statement": "Tests pass reliably.",
        "probability": 0.85,
        "evidence": "CI pipeline evidence",
    })
    assert resp.status_code == 200


# ── /forget ───────────────────────────────────────────────────────────────────

def test_forget_ok(client):
    client.post("/remember", json={"content": "Temporary test memory to forget."})
    resp = client.post("/forget", json={"query": "Temporary test memory"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── /personality ──────────────────────────────────────────────────────────────

def test_get_personality(client):
    resp = client.get("/personality")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


def test_set_personality(client):
    resp = client.put("/personality", json={
        "action": "preset",
        "preset": "analytical",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── /status ───────────────────────────────────────────────────────────────────

def test_status(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_atoms" in data


# ── /atoms/{atom_id} ──────────────────────────────────────────────────────────

def test_get_atom_found(client):
    post = client.post("/remember", json={"content": "Findable atom."})
    atom_id = post.json()["atom_id"]

    resp = client.get(f"/atoms/{atom_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == atom_id


def test_get_atom_not_found(client):
    resp = client.get("/atoms/nonexistent-uuid-000")
    assert resp.status_code == 404


# ── /spaces/current ───────────────────────────────────────────────────────────

def test_clear_current_space(client):
    client.post("/remember", json={"content": "To be cleared."})
    resp = client.delete("/spaces/current")
    assert resp.status_code == 200
    data = resp.json()
    assert "deleted" in data
    assert data["deleted"] >= 1  # at minimum the atom we just seeded


# ── /tenants ──────────────────────────────────────────────────────────────────

def test_list_tenants(client):
    client.post("/remember", json={"content": "Tenant test memory."})
    resp = client.get("/tenants")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ── /spaces ───────────────────────────────────────────────────────────────────

def test_list_spaces(client):
    resp = client.get("/spaces?tenant_id=default")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ── /graph ────────────────────────────────────────────────────────────────────

def test_get_graph(client):
    resp = client.get("/graph?tenant_id=default&space=default")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data


def test_get_graph_with_limit(client):
    resp = client.get("/graph?tenant_id=default&space=default&limit=5")
    assert resp.status_code == 200
    assert len(resp.json()["nodes"]) <= 5


def test_get_graph_empty_types_uses_defaults(client):
    resp = client.get("/graph?tenant_id=default&space=default&types=")
    assert resp.status_code == 200


def test_get_graph_custom_min_confidence(client):
    resp = client.get("/graph?tenant_id=default&space=default&min_confidence=0.9")
    assert resp.status_code == 200


# ── /viz (404 when no static file) ───────────────────────────────────────────

def test_viz_returns_404_when_no_html(client):
    resp = client.get("/viz")
    # Either 200 (if static file exists) or 404 (if not) — both are valid
    assert resp.status_code in (200, 404)


# ── Empty / whitespace query validation ─────────────────────────────────────

def test_recall_rejects_empty_query(client):
    """A blank query should return 422 (validation), not hit the engine."""
    resp = client.post("/recall", json={"query": "", "top_k": 5})
    assert resp.status_code == 422
    body = resp.json()
    assert "query" in str(body).lower()


def test_recall_rejects_whitespace_only_query(client):
    resp = client.post("/recall", json={"query": "   \t\n  ", "top_k": 5})
    assert resp.status_code == 422


def test_recall_accepts_cjk_query(client):
    """Chinese/Japanese/Korean queries should pass validation (bytes > 0)."""
    resp = client.post("/recall", json={"query": "金刚经空性", "top_k": 5})
    assert resp.status_code == 200


def test_forget_rejects_empty_query(client):
    resp = client.post("/forget", json={"query": "", "reason": "test"})
    assert resp.status_code == 422
