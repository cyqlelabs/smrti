"""Tests for the `smrti` command line interface (cli.py).

The serve commands are exercised with their server entry points patched out:
what is under test is the CLI wiring — option parsing, the exposure warning,
and the pidfile lifecycle — not uvicorn itself.
"""
from __future__ import annotations

import os
import runpy
import signal
import tempfile
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from smrti import cli
from smrti.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def run_dir(tmp_path, monkeypatch):
    """Redirect the pidfile directory so tests never touch ~/.smrti/run."""
    d = tmp_path / "run"
    monkeypatch.setattr(cli, "_RUN_DIR", d)
    return d


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
        except OSError:
            pass


# ── init / status ─────────────────────────────────────────────────────────────

def test_init_creates_store(db_path):
    result = runner.invoke(app, ["init", "--db", db_path, "--tenant-id", "t1", "--space", "s1"])
    assert result.exit_code == 0, result.output
    assert "Initialized Smrti at" in result.output
    assert "Tenant: t1 | Space: s1" in result.output
    assert "Total atoms: 0" in result.output


def test_status_reports_atom_counts(db_path):
    runner.invoke(app, ["init", "--db", db_path])
    result = runner.invoke(app, ["status", "--db", db_path])
    assert result.exit_code == 0, result.output
    assert "Total atoms:" in result.output
    assert "By type:" in result.output
    assert "Personality: balanced" in result.output


def test_status_omits_personality_line_when_absent(db_path):
    runner.invoke(app, ["init", "--db", db_path])
    with patch("smrti.Smrti.status", return_value={"total_atoms": 0, "by_type": {}}):
        result = runner.invoke(app, ["status", "--db", db_path])
    assert result.exit_code == 0, result.output
    assert "Personality:" not in result.output


# ── _warn_if_exposed ──────────────────────────────────────────────────────────

def test_warn_if_exposed_is_silent_on_loopback(capsys):
    cli._warn_if_exposed("127.0.0.1")
    assert capsys.readouterr().err == ""


def test_warn_if_exposed_is_silent_with_api_key(monkeypatch, capsys):
    monkeypatch.setenv("SMRTI_API_KEY", "secret")
    cli._warn_if_exposed("0.0.0.0")
    assert capsys.readouterr().err == ""


def test_warn_if_exposed_warns_on_public_bind(monkeypatch, capsys):
    monkeypatch.delenv("SMRTI_API_KEY", raising=False)
    cli._warn_if_exposed("0.0.0.0")
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "SMRTI_API_KEY" in err


# ── _pidfile / _held / _signal ────────────────────────────────────────────────

def test_pidfile_writes_pid_and_removes_on_exit(run_dir):
    path = run_dir / "rest-8420.pid"
    with cli._pidfile("rest", 8420):
        assert path.read_text() == str(os.getpid())
    assert not path.exists()


def test_pidfile_rejects_a_second_server_on_the_same_port(run_dir):
    with cli._pidfile("rest", 8420):
        with pytest.raises(typer.BadParameter, match="already running on port 8420"):
            with cli._pidfile("rest", 8420):
                pass


def test_held_is_true_while_the_pidfile_is_locked(run_dir):
    path = run_dir / "rest-8420.pid"
    with cli._pidfile("rest", 8420):
        assert cli._held(path) is True


def test_held_is_false_for_an_unlocked_pidfile(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "rest-8420.pid"
    path.write_text("12345")
    assert cli._held(path) is False


def test_held_is_false_for_a_missing_pidfile(run_dir):
    assert cli._held(run_dir / "does-not-exist.pid") is False


def test_signal_delivers_the_signal():
    with patch("smrti.cli.os.kill") as kill:
        cli._signal(4242, signal.SIGTERM)
    kill.assert_called_once_with(4242, signal.SIGTERM)


def test_signal_ignores_a_dead_process():
    with patch("smrti.cli.os.kill", side_effect=ProcessLookupError):
        cli._signal(4242, signal.SIGTERM)  # must not raise


# ── stop ──────────────────────────────────────────────────────────────────────

def test_stop_rejects_unknown_mode():
    result = runner.invoke(app, ["stop", "bogus"])
    assert result.exit_code != 0
    assert "unknown mode" in result.output


def test_stop_reports_when_nothing_is_running():
    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0, result.output
    assert "No running Smrti servers found." in result.output


def test_stop_clears_a_stale_pidfile(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "rest-8420.pid"
    path.write_text("999999")
    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0, result.output
    assert "not running (cleared stale pidfile)" in result.output
    assert not path.exists()


def test_stop_skips_an_unreadable_pidfile(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "rest-8420.pid"
    path.write_text("not-a-pid")
    with patch("smrti.cli._held", return_value=True):
        result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0, result.output
    assert "unreadable pidfile, skipping" in result.output
    assert path.exists()


def test_stop_sends_sigterm_and_reports_a_graceful_exit(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rest-8420.pid").write_text("4242")
    # Held on the first check, released once SIGTERM lands.
    with patch("smrti.cli._held", side_effect=[True, False, False]):
        with patch("smrti.cli._signal") as sig:
            result = runner.invoke(app, ["stop", "rest"])
    assert result.exit_code == 0, result.output
    assert "stopped (pid 4242)" in result.output
    sig.assert_called_once_with(4242, signal.SIGTERM)
    assert not (run_dir / "rest-8420.pid").exists()


def test_stop_escalates_to_sigkill_after_the_timeout(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "proxy-8421.pid").write_text("4242")
    with patch("smrti.cli._held", return_value=True):
        with patch("smrti.cli._signal") as sig:
            result = runner.invoke(app, ["stop", "--timeout", "0"])
    assert result.exit_code == 0, result.output
    assert "killed (pid 4242)" in result.output
    assert sig.call_args_list[0][0] == (4242, signal.SIGTERM)
    assert sig.call_args_list[-1][0] == (4242, signal.SIGKILL)


def test_stop_waits_for_a_slow_shutdown(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rest-8420.pid").write_text("4242")
    # Still held on the first poll, released on the second.
    with patch("smrti.cli._held", side_effect=[True, True, False, False]):
        with patch("smrti.cli._signal"):
            with patch("smrti.cli.time.sleep") as sleep:
                result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0, result.output
    assert "stopped (pid 4242)" in result.output
    sleep.assert_called_with(0.1)


def test_stop_filters_by_port(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rest-8420.pid").write_text("1")
    (run_dir / "rest-9999.pid").write_text("2")
    result = runner.invoke(app, ["stop", "rest", "--port", "9999"])
    assert result.exit_code == 0, result.output
    assert "rest-9999" in result.output
    assert "rest-8420" not in result.output
    assert (run_dir / "rest-8420.pid").exists()


# ── serve ─────────────────────────────────────────────────────────────────────

def test_serve_mcp_starts_the_stdio_server():
    with patch("smrti.servers.mcp.run_mcp_server") as run:
        result = runner.invoke(app, ["serve", "mcp"])
    assert result.exit_code == 0, result.output
    run.assert_called_once_with()


def test_serve_rest_binds_and_holds_a_pidfile(run_dir):
    seen = {}

    def _run(host, port):
        seen["locked"] = (run_dir / f"rest-{port}.pid").exists()

    with patch("smrti.servers.rest.run_rest_server", side_effect=_run) as run:
        result = runner.invoke(app, ["serve", "rest", "--host", "127.0.0.1", "--port", "8500"])
    assert result.exit_code == 0, result.output
    assert "Starting Smrti REST API on http://127.0.0.1:8500" in result.output
    run.assert_called_once_with(host="127.0.0.1", port=8500)
    assert seen["locked"] is True
    assert not (run_dir / "rest-8500.pid").exists()


def test_serve_rest_warns_when_binding_publicly(monkeypatch):
    monkeypatch.delenv("SMRTI_API_KEY", raising=False)
    with patch("smrti.servers.rest.run_rest_server"):
        result = runner.invoke(app, ["serve", "rest", "--host", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output


def test_serve_viz_opens_the_browser():
    with patch("smrti.servers.rest.run_rest_server"):
        with patch("threading.Thread") as thread:
            result = runner.invoke(app, ["serve", "viz", "--port", "8501"])
    assert result.exit_code == 0, result.output
    assert "Starting Smrti visualizer on http://127.0.0.1:8501/viz" in result.output
    thread.assert_called_once()
    assert thread.call_args.kwargs["daemon"] is True
    # Run the target the CLI handed to the thread; it must open exactly that URL.
    with patch("time.sleep"):
        with patch("webbrowser.open") as open_url:
            thread.call_args.kwargs["target"]()
    open_url.assert_called_once_with("http://127.0.0.1:8501/viz")


def test_serve_viz_respects_no_browser():
    with patch("smrti.servers.rest.run_rest_server") as run:
        with patch("threading.Thread") as thread:
            result = runner.invoke(app, ["serve", "viz", "--no-browser", "--port", "8502"])
    assert result.exit_code == 0, result.output
    thread.assert_not_called()
    run.assert_called_once_with(host="127.0.0.1", port=8502)


def test_serve_proxy_reports_the_default_upstream(monkeypatch):
    monkeypatch.delenv("SMRTI_UPSTREAM_URL", raising=False)
    with patch("smrti.servers.proxy.run_proxy_server") as run:
        result = runner.invoke(app, ["serve", "proxy", "--host", "127.0.0.1", "--port", "8503"])
    assert result.exit_code == 0, result.output
    assert "Upstream: https://api.openai.com" in result.output
    assert "Visualizer:  http://127.0.0.1:8503/viz" in result.output
    run.assert_called_once_with(host="127.0.0.1", port=8503)


def test_serve_proxy_upstream_option_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("SMRTI_UPSTREAM_URL", "https://env.example")
    with patch("smrti.servers.proxy.run_proxy_server"):
        result = runner.invoke(
            app, ["serve", "proxy", "--upstream", "https://flag.example/", "--port", "8504"]
        )
    assert result.exit_code == 0, result.output
    # The trailing slash is stripped before it reaches the environment.
    assert os.environ["SMRTI_UPSTREAM_URL"] == "https://flag.example"
    assert "Upstream: https://flag.example" in result.output
    # A 0.0.0.0 bind advertises the loopback address for the visualizer.
    assert "Visualizer:  http://127.0.0.1:8504/viz" in result.output


def test_serve_town_sets_env_and_starts_the_simulation(monkeypatch, tmp_path):
    town_db = str(tmp_path / "town.db")
    with patch("smrti_town.server.serve") as serve:
        result = runner.invoke(
            app,
            ["serve", "town", "--db", town_db, "--tenant-id", "riverton",
             "--no-browser", "--port", "8505"],
        )
    assert result.exit_code == 0, result.output
    assert os.environ["SMRTI_TOWN_DB"] == town_db
    assert os.environ["SMRTI_TOWN_TENANT"] == "riverton"
    assert "Tenant: riverton" in result.output
    serve.assert_called_once_with(host="127.0.0.1", port=8505)


def test_serve_town_opens_the_browser(tmp_path):
    with patch("smrti_town.server.serve"):
        with patch("threading.Thread") as thread:
            result = runner.invoke(
                app, ["serve", "town", "--db", str(tmp_path / "t.db"), "--port", "8506"]
            )
    assert result.exit_code == 0, result.output
    thread.assert_called_once()
    with patch("time.sleep"):
        with patch("webbrowser.open") as open_url:
            thread.call_args.kwargs["target"]()
    open_url.assert_called_once_with("http://127.0.0.1:8506")


# ── python -m smrti ───────────────────────────────────────────────────────────

def test_module_entrypoint_invokes_the_app():
    with patch("smrti.cli.app") as mock_app:
        runpy.run_module("smrti.__main__", run_name="__main__")
    mock_app.assert_called_once_with()
