"""Per-request space routing on the REST server.

The REST server historically served exactly one (tenant, write_space) pair
fixed at process start by env vars. These tests pin the per-request overlay:
an optional ``space`` field on the write/read endpoints routes the call to a
cached per-space Smrti instance, while requests without the field keep the
env-configured behavior byte-for-byte.
"""
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient


async def _noop_reflect(*_args, **_kwargs):
    return


def _reset(rest_mod):
    rest_mod._mem = None
    cache = getattr(rest_mod, "_space_mems", None)
    if cache is not None:
        cache.clear()


@pytest.fixture()
def rest_mod():
    from smrti.servers import rest
    return rest


@pytest.fixture()
def client(tmp_path, monkeypatch, rest_mod):
    from smrti.servers import config as cfg
    monkeypatch.setattr(cfg, "DB", str(tmp_path / "mem.db"))
    monkeypatch.setattr(cfg, "TENANT_ID", "default")
    monkeypatch.setattr(cfg, "SPACE", "main")
    monkeypatch.setattr(cfg, "READ_SPACES", None)
    monkeypatch.setattr(cfg, "EXTRACT", False)
    monkeypatch.setattr(cfg, "IGNORE_PATTERNS", [])
    _reset(rest_mod)
    with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
        with TestClient(rest_mod.app, raise_server_exceptions=True) as c:
            yield c
    _reset(rest_mod)


def _atom_space(rest_mod, atom_id):
    row = rest_mod.get_mem().db.fetchone(
        "SELECT space FROM atoms WHERE id = ?", (atom_id,)
    )
    return row["space"]


def _atom_confidence(rest_mod, atom_id):
    row = rest_mod.get_mem().db.fetchone(
        "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
    )
    return row["confidence"]


# ── /remember ────────────────────────────────────────────────────────────────

def test_remember_routes_to_requested_space(client, rest_mod):
    resp = client.post(
        "/remember", json={"content": "Deploy notes for work.", "space": "work"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["space"] == "work"
    assert _atom_space(rest_mod, data["atom_id"]) == "work"


def test_remember_without_space_lands_in_configured_space(client, rest_mod):
    resp = client.post("/remember", json={"content": "Plain default write."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["space"] == "main"
    assert _atom_space(rest_mod, data["atom_id"]) == "main"


def test_remember_empty_space_means_configured_space(client, rest_mod):
    resp = client.post("/remember", json={"content": "Blank space string.", "space": ""})
    assert resp.status_code == 200
    assert _atom_space(rest_mod, resp.json()["atom_id"]) == "main"


# ── /believe ─────────────────────────────────────────────────────────────────

def test_believe_routes_to_requested_space(client, rest_mod):
    resp = client.post(
        "/believe",
        json={"statement": "Work believes this.", "probability": 0.9, "space": "work"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["space"] == "work"
    assert _atom_space(rest_mod, data["atom_id"]) == "work"


# ── /recall ──────────────────────────────────────────────────────────────────

def test_recall_defaults_to_the_requested_space_only(client):
    client.post(
        "/remember", json={"content": "The secret project is Apollo.", "space": "work"}
    )
    client.post("/remember", json={"content": "The secret recipe is carbonara."})

    resp = client.post("/recall", json={"query": "secret", "space": "work"})
    assert resp.status_code == 200
    memories = resp.json()["memories"]
    assert memories
    assert {m["space"] for m in memories} == {"work"}


def test_recall_read_spaces_overlay_spans_spaces(client):
    client.post(
        "/remember", json={"content": "The secret project is Apollo.", "space": "work"}
    )
    client.post("/remember", json={"content": "The secret recipe is carbonara."})

    resp = client.post(
        "/recall",
        json={"query": "secret", "space": "work", "read_spaces": ["work", "main"]},
    )
    assert resp.status_code == 200
    assert {m["space"] for m in resp.json()["memories"]} == {"work", "main"}


def test_recall_without_space_reads_the_configured_space(client):
    client.post(
        "/remember", json={"content": "The secret project is Apollo.", "space": "work"}
    )
    client.post("/remember", json={"content": "The secret recipe is carbonara."})

    resp = client.post("/recall", json={"query": "secret"})
    assert resp.status_code == 200
    memories = resp.json()["memories"]
    assert memories
    assert {m["space"] for m in memories} == {"main"}


# ── /forget ──────────────────────────────────────────────────────────────────

def test_forget_softens_only_the_requested_space(client, rest_mod):
    in_work = client.post(
        "/remember", json={"content": "Ephemeral fact to forget.", "space": "work"}
    ).json()["atom_id"]
    in_main = client.post(
        "/remember", json={"content": "Ephemeral fact to forget."}
    ).json()["atom_id"]

    resp = client.post("/forget", json={"query": "Ephemeral fact", "space": "work"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    assert _atom_confidence(rest_mod, in_work) < 0.5
    assert _atom_confidence(rest_mod, in_main) == pytest.approx(0.5)


# ── instance cache ───────────────────────────────────────────────────────────

def test_get_mem_reuses_per_space_instances(client, rest_mod):
    default = rest_mod.get_mem()
    assert rest_mod.get_mem() is default
    assert rest_mod.get_mem("") is default
    assert rest_mod.get_mem("main") is default  # configured name → env instance

    work = rest_mod.get_mem("work")
    assert work is not default
    assert rest_mod.get_mem("work") is work
    assert work.tenant_id == "default"
    assert work.write_space == "work"
    assert work.read_spaces == ["work"]


def test_space_instances_are_bounded(client, rest_mod):
    for i in range(rest_mod._SPACE_MEMS_MAX + 1):
        rest_mod.get_mem(f"space-{i}")
    assert len(rest_mod._space_mems) == rest_mod._SPACE_MEMS_MAX
    assert "space-0" not in rest_mod._space_mems
    assert f"space-{rest_mod._SPACE_MEMS_MAX}" in rest_mod._space_mems


# ── extraction follows the routed instance ───────────────────────────────────

def test_extraction_uses_the_requested_space_instance(client):
    captured = {}

    async def _capture(episode_id, content, mem, auth, model, upstream, source="user", **kw):
        captured["mem"] = mem

    with patch("smrti.servers.config.EXTRACT", True), \
         patch("smrti.extraction.extract.extract_and_link_serialized", new=_capture):
        resp = client.post(
            "/remember", json={"content": "Extract me please.", "space": "work"}
        )

    assert resp.status_code == 200
    assert captured["mem"].write_space == "work"
