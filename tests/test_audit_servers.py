"""Regressions from the servers audit: tenant-bound keys, the proxy, shutdown and limits."""
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from smrti import Smrti, call_log

ADMIN = {"X-Api-Key": "admin-key"}
ACME = {"X-Api-Key": "acme-key"}
MULTI = {"X-Api-Key": "multi-key"}


async def _noop_reflect(*_args, **_kwargs):
    return


def _reset(rest_mod):
    rest_mod._mem = None
    rest_mod._space_mems.clear()
    rest_mod._tenant_mems.clear()


@pytest.fixture()
def keyed(tmp_path, monkeypatch):
    """A REST server on its own database with an all-tenants key and two bound keys."""
    from smrti.servers import config as cfg
    from smrti.servers import rest as rest_mod

    db = str(tmp_path / "mem.db")
    for name, value in {
        "DB": db, "TENANT_ID": "default", "SPACE": "default", "READ_SPACES": None,
        "EXTRACT": False, "IGNORE_PATTERNS": [], "API_KEY": "admin-key",
        "API_KEYS": {"acme-key": ("acme",), "multi-key": ("beta", "gamma")},
    }.items():
        monkeypatch.setattr(cfg, name, value)
    _reset(rest_mod)
    Smrti(db_path=db, tenant_id="default").remember("default tenant: the launch codes are 1234")
    Smrti(db_path=db, tenant_id="acme").remember("acme tenant: payroll runs on Fridays")
    call_log._CALL_LOG.clear()
    with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
        with TestClient(rest_mod.app, raise_server_exceptions=True) as client:
            client.db = db
            yield client
    _reset(rest_mod)
    call_log._CALL_LOG.clear()


def _contents(response):
    return [n["content"] for n in response.json()["nodes"]]


def test_a_bound_key_cannot_read_another_tenants_graph_or_atoms(keyed):
    """One key used to read any tenant: /graph and /atoms took the tenant from the query."""
    other = Smrti(db_path=keyed.db, tenant_id="default").db.fetchone(
        "SELECT id FROM atoms WHERE tenant_id = 'default'"
    )["id"]

    assert keyed.get("/graph", params={"tenant_id": "default"}, headers=ACME).status_code == 403
    assert keyed.get(f"/atoms/{other}", params={"tenant_id": "default"}, headers=ACME).status_code == 403
    assert _contents(keyed.get("/graph", headers=ACME)) == ["acme tenant: payroll runs on Fridays"]
    assert _contents(keyed.get("/graph", params={"tenant_id": "acme"}, headers=ADMIN)) == [
        "acme tenant: payroll runs on Fridays"
    ]


def test_a_bound_key_reads_and_writes_its_own_tenant_when_it_names_none(keyed):
    """The default tenant of a bound key is its first, on every route that names none."""
    written = keyed.post("/remember", json={"content": "acme: invoices go out monthly"}, headers=ACME)
    tenant = Smrti(db_path=keyed.db, tenant_id="default").db.fetchone(
        "SELECT tenant_id FROM atoms WHERE id = ?", (written.json()["atom_id"],)
    )["tenant_id"]
    assert tenant == "acme"

    assert keyed.get("/status", headers=ACME).json()["total_atoms"] == 2
    assert keyed.get("/status", headers=ADMIN).json()["total_atoms"] == 1
    assert keyed.get("/tenants", headers=ACME).json() == ["acme"]
    assert keyed.get("/tenants", headers=ADMIN).json() == ["acme", "default"]
    assert 'tenant="acme"' in keyed.get("/metrics", headers=ACME).text


def test_shared_logs_are_filtered_by_tenant_and_only_the_admin_key_clears_them(keyed):
    """/llm-calls and /decisions held every tenant's conversations and queries for any key."""
    call_log.append({"kind": "proxy", "tenant_id": "acme", "original_messages": ["acme says hi"]})
    call_log.append({"kind": "proxy", "tenant_id": "default", "original_messages": ["secret"]})

    assert [e["tenant_id"] for e in keyed.get("/llm-calls", headers=ACME).json()] == ["acme"]
    assert len(keyed.get("/llm-calls", headers=ADMIN).json()) == 2
    assert keyed.delete("/llm-calls", headers=ACME).status_code == 403
    assert keyed.delete("/decisions", headers=MULTI).status_code == 403
    assert keyed.delete("/llm-calls", headers=ADMIN).status_code == 200


def test_a_bound_key_may_not_open_another_database(keyed):
    """An alternate ?db= has tenants of its own, which no key was bound to."""
    response = keyed.get("/tenants", params={"db": keyed.db}, headers=ACME)
    assert response.status_code == 403
    assert keyed.get("/tenants", params={"db": keyed.db}, headers=ADMIN).status_code == 200


def test_the_proxy_refuses_a_tenant_header_outside_the_key_and_caps_its_length(tmp_path, monkeypatch):
    """X-Smrti-Tenant-Id selected any tenant, and accepted names of any length."""
    from smrti.servers import config as cfg
    from smrti.servers import proxy

    monkeypatch.setattr(cfg, "API_KEYS", {"acme-key": ("acme",)})
    with patch("smrti.servers.proxy._bootstrap", return_value=("default", "default", str(tmp_path / "p.db"))):
        with TestClient(proxy.app, raise_server_exceptions=True) as client:
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            refused = client.post(
                "/v1/chat/completions", json=body,
                headers={**ACME, "X-Smrti-Tenant-Id": "victim"},
            )
            assert refused.status_code == 403
            assert refused.json()["error"]["code"] == "tenant_not_allowed"

            monkeypatch.setattr(cfg, "API_KEYS", {})
            too_long = client.post(
                "/v1/chat/completions", json=body, headers={"X-Smrti-Tenant-Id": "T" * 129},
            )
            assert too_long.status_code == 400


def test_a_malformed_api_keys_entry_stops_the_server_without_echoing_the_key():
    """A bound key that grants nothing must not start silently, nor leak into the log."""
    from smrti.servers.config import _parse_api_keys

    assert _parse_api_keys(" k1:acme ; k2:beta|gamma,") == {"k1": ("acme",), "k2": ("beta", "gamma")}
    for raw in ("sekrit-without-tenants", "sekrit:", "k1:a;k1:b"):
        with pytest.raises(ValueError) as exc:
            _parse_api_keys(raw)
        assert "sekrit" not in str(exc.value)
