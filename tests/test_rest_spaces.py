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


def _configure(tmp_path, monkeypatch, rest_mod, tenant_id="default"):
    from smrti.servers import config as cfg
    monkeypatch.setattr(cfg, "DB", str(tmp_path / "mem.db"))
    monkeypatch.setattr(cfg, "TENANT_ID", tenant_id)
    monkeypatch.setattr(cfg, "SPACE", "main")
    monkeypatch.setattr(cfg, "READ_SPACES", None)
    monkeypatch.setattr(cfg, "EXTRACT", False)
    monkeypatch.setattr(cfg, "IGNORE_PATTERNS", [])
    _reset(rest_mod)


@pytest.fixture()
def client(tmp_path, monkeypatch, rest_mod):
    _configure(tmp_path, monkeypatch, rest_mod)
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


# ── /reflect ─────────────────────────────────────────────────────────────────

def _epoch_count(rest_mod, space):
    row = rest_mod.get_mem().db.fetchone(
        "SELECT epoch_count FROM personality WHERE tenant_id = 'default' AND space = ?",
        (space,),
    )
    return row["epoch_count"] if row else None


def test_reflect_targets_the_requested_space(client, rest_mod):
    client.post("/remember", json={"content": "Work epoch fodder.", "space": "work"})

    resp = client.post("/reflect", json={"space": "work"})
    assert resp.status_code == 200
    assert "beliefs_updated" in resp.json()
    assert _epoch_count(rest_mod, "work") == 1
    assert _epoch_count(rest_mod, "main") == 0


def test_reflect_without_body_stays_backward_compatible(client, rest_mod):
    resp = client.post("/reflect")
    assert resp.status_code == 200
    assert "beliefs_updated" in resp.json()
    assert _epoch_count(rest_mod, "main") == 1


def test_reflect_with_empty_body_reflects_the_configured_space(client, rest_mod):
    resp = client.post("/reflect", json={})
    assert resp.status_code == 200
    assert _epoch_count(rest_mod, "main") == 1


# ── /personality ─────────────────────────────────────────────────────────────

def test_personality_routes_to_the_requested_space(client):
    resp = client.put(
        "/personality",
        json={"action": "preset", "preset": "analytical", "space": "work"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    got = client.get("/personality", params={"space": "work"})
    assert got.json()["preset_name"] == "analytical"

    main = client.get("/personality")
    assert main.json()["preset_name"] == "balanced"


# ── DELETE /spaces/current ───────────────────────────────────────────────────

def test_clear_space_param_deletes_only_that_space(client, rest_mod):
    client.post("/remember", json={"content": "Work atom to clear.", "space": "work"})
    kept = client.post("/remember", json={"content": "Main atom to keep."}).json()["atom_id"]

    resp = client.delete("/spaces/current", params={"space": "work"})
    assert resp.status_code == 200
    assert resp.json()["deleted"] >= 1

    db = rest_mod.get_mem().db
    left = db.fetchone("SELECT COUNT(*) AS n FROM atoms WHERE space = 'work'", ())
    assert left["n"] == 0
    assert _atom_space(rest_mod, kept) == "main"


# ── reflect loop coverage ────────────────────────────────────────────────────

def test_reflect_loop_covers_every_touched_space(tmp_path, monkeypatch, rest_mod):
    _configure(tmp_path, monkeypatch, rest_mod)
    captured = {}

    async def _capture(get_instances):
        captured["get"] = get_instances

    with patch("smrti.servers.rest.run_reflect_loop", new=_capture):
        with TestClient(rest_mod.app, raise_server_exceptions=True) as c:
            c.post("/remember", json={"content": "Work atom.", "space": "work"})
            mems = captured["get"]()
            assert {m.write_space for m in mems} == {"main", "work"}
    _reset(rest_mod)


# ── /status capability signal ────────────────────────────────────────────────

def test_status_reports_spaces_and_version(client):
    """The `spaces` key is how clients detect per-request space support."""
    client.post("/remember", json={"content": "Work atom.", "space": "work"})
    client.post("/remember", json={"content": "Main atom."})

    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["spaces"]) >= {"main", "work"}
    assert data["version"]


def test_status_reports_the_configured_write_space(client):
    """Clients need this to tell whether their own space config agrees with the
    server's — a mismatch would send writes to a space the graph is not in."""
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["space"] == "main"


def test_mcp_status_reports_version(client, rest_mod):
    from smrti.servers.mcp import handle_tool
    result = handle_tool(rest_mod.get_mem(), "smrti_status", {})
    assert result["version"]
    assert "spaces" in result


# ── GET /spaces tenant default ───────────────────────────────────────────────

def test_list_spaces_defaults_to_the_configured_tenant(tmp_path, monkeypatch, rest_mod):
    """A server started for tenant `acme` must list acme's spaces, not `default`'s."""
    _configure(tmp_path, monkeypatch, rest_mod, tenant_id="acme")
    with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
        with TestClient(rest_mod.app, raise_server_exceptions=True) as c:
            c.post("/remember", json={"content": "Acme memory."})
            resp = c.get("/spaces")
            assert resp.status_code == 200
            assert resp.json() == ["main"]
    _reset(rest_mod)


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
