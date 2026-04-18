"""Tests for /metrics Prometheus exposition endpoint."""
import os
import tempfile
from unittest.mock import patch

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

    async def _noop_reflect(*_args, **_kwargs):
        return

    with patch.object(rest_mod, "get_mem", return_value=mem_instance):
        with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
            with TestClient(rest_mod.app, raise_server_exceptions=True) as c:
                yield c


def test_metrics_returns_plaintext(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")


def test_metrics_contains_atoms_total(client):
    # Store something so the gauge is > 0
    client.post("/remember", json={"content": "metric check fixture"})

    resp = client.get("/metrics")
    body = resp.text
    assert "# TYPE smrti_atoms_total gauge" in body
    assert "smrti_atoms_total{" in body


def test_metrics_valid_prometheus_format(client):
    """Each metric line must match Prometheus text format roughly."""
    resp = client.get("/metrics")
    lines = [l for l in resp.text.splitlines() if l and not l.startswith("#")]
    for line in lines:
        # Prom format: metric_name{labels} value
        assert " " in line, f"malformed: {line}"
        name_labels, value = line.rsplit(" ", 1)
        # value must parse as number
        float(value)
