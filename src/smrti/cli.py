"""CLI for smrti: smrti serve mcp|rest|viz|proxy|town, smrti stop, smrti init, smrti status."""
from __future__ import annotations

import contextlib
import os
import signal
import time
from pathlib import Path
from typing import IO, Iterator, Optional

import typer

# The pidfile lock rides whichever advisory-lock primitive this OS has: flock
# where it exists, msvcrt region locks on Windows — importing fcntl at module
# level made every smrti command a Unix-only program. Windows region locks are
# mandatory, so the locked byte sits far past any pid text: `stop` reads the
# pid of a live, still-locked server, and a lock over those bytes would turn
# that read into a permission error instead of an answer.
if os.name == "nt":
    import msvcrt

    _LOCK_OFFSET = 4096

    def _try_lock(handle: IO[str]) -> None:
        handle.seek(_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle: IO[str]) -> None:
        handle.seek(_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(handle: IO[str]) -> None:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle: IO[str]) -> None:
        fcntl.flock(handle, fcntl.LOCK_UN)

app = typer.Typer(help="Smrti memory engine CLI")

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

_RUN_DIR = Path(os.path.expanduser(os.environ.get("SMRTI_RUN_DIR", "~/.smrti/run")))

_SERVER_MODES = ("rest", "viz", "proxy", "town")


@contextlib.contextmanager
def _pidfile(mode: str, port: int) -> Iterator[None]:
    """Record this process so `smrti stop` can find it.

    The file is held under an exclusive advisory lock for the lifetime of the
    server. The kernel drops that lock however the process dies, so a lock left
    unheld marks the entry stale even after a SIGKILL or a power loss — which
    keeps `smrti stop` from ever signalling a recycled PID.
    """
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = _RUN_DIR / f"{mode}-{port}.pid"
    handle = path.open("a+")
    try:
        _try_lock(handle)
    except OSError:
        handle.close()
        raise typer.BadParameter(f"a {mode} server is already running on port {port}")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    try:
        yield
    finally:
        if os.name == "nt":
            # Windows refuses to delete an open file, so the handle goes
            # first; closing it is also what releases the region lock.
            handle.close()
            path.unlink(missing_ok=True)
        else:
            path.unlink(missing_ok=True)
            handle.close()


def _held(path: Path) -> bool:
    """True while the process that wrote this pidfile is still alive."""
    try:
        handle = path.open("r+")
    except OSError:
        return False
    try:
        _try_lock(handle)
    except OSError:
        return True
    else:
        _unlock(handle)
        return False
    finally:
        handle.close()


def _signal(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, sig)


def _warn_if_exposed(host: str) -> None:
    """Warn once at startup when binding beyond loopback without an API key."""
    if host in _LOOPBACK_HOSTS or os.environ.get("SMRTI_API_KEY"):
        return
    typer.secho(
        f"WARNING: binding to {host} without SMRTI_API_KEY set — the API is "
        "reachable from other machines with no authentication. "
        "Set SMRTI_API_KEY to require a key on every request.",
        fg=typer.colors.YELLOW,
        err=True,
        bold=True,
    )


@app.command()
def init(
    db: Optional[str] = typer.Option(None, help="Path to SQLite database  [env: SMRTI_DB]"),
    personality: Optional[str] = typer.Option(None, help="Personality preset  [env: SMRTI_PERSONALITY]"),
    tenant_id: Optional[str] = typer.Option(None, help="Tenant ID  [env: SMRTI_TENANT_ID]"),
    space: Optional[str] = typer.Option(None, help="Memory space  [env: SMRTI_SPACE]"),
) -> None:
    """Initialize a new Smrti memory store."""
    from smrti import Smrti
    from smrti.servers import config as cfg

    # Unset options fall back to the same env vars the servers read, so the CLI
    # and `smrti serve` always address the same store.
    db = db or cfg.DB
    personality = personality or cfg.PERSONALITY
    tenant_id = tenant_id or cfg.TENANT_ID
    space = space or cfg.SPACE

    mem = Smrti(db_path=db, personality=personality, tenant_id=tenant_id, write_space=space)
    s = mem.status()
    typer.echo(f"Initialized Smrti at {os.path.expanduser(db)}")
    typer.echo(f"Tenant: {tenant_id} | Space: {space} | Personality: {personality}")
    typer.echo(f"Total atoms: {s['total_atoms']}")
    mem.close()


@app.command()
def status(
    db: Optional[str] = typer.Option(None, help="Path to SQLite database  [env: SMRTI_DB]"),
    tenant_id: Optional[str] = typer.Option(None, help="Tenant ID  [env: SMRTI_TENANT_ID]"),
    space: Optional[str] = typer.Option(None, help="Memory space  [env: SMRTI_SPACE]"),
) -> None:
    """Show memory statistics."""
    from smrti import Smrti
    from smrti.servers import config as cfg

    db = db or cfg.DB
    tenant_id = tenant_id or cfg.TENANT_ID
    space = space or cfg.SPACE

    mem = Smrti(db_path=db, personality=cfg.PERSONALITY, tenant_id=tenant_id, write_space=space)
    s = mem.status()
    typer.echo(f"Total atoms: {s['total_atoms']}")
    typer.echo(f"By type: {s['by_type']}")
    p = s.get("personality", {})
    if p:
        typer.echo(f"Personality: {p.get('preset_name', 'custom')}")
    mem.close()


@app.command()
def stop(
    mode: str = typer.Argument("", help="Mode to stop: rest, viz, proxy or town. All if omitted."),
    port: int = typer.Option(0, help="Only stop the server bound to this port"),
    timeout: float = typer.Option(10.0, help="Seconds to wait for a graceful shutdown"),
) -> None:
    """Stop servers started by `smrti serve`.

    Sends SIGTERM so uvicorn drains connections and checkpoints the WAL, then
    escalates to SIGKILL if the process outlives the timeout.
    """
    if mode and mode not in _SERVER_MODES:
        raise typer.BadParameter(f"unknown mode {mode!r}, expected one of {', '.join(_SERVER_MODES)}")

    pattern = f"{mode or '*'}-{port or '*'}.pid"
    entries = sorted(_RUN_DIR.glob(pattern)) if _RUN_DIR.is_dir() else []
    if not entries:
        typer.echo("No running Smrti servers found.")
        return

    for path in entries:
        if not _held(path):
            path.unlink(missing_ok=True)
            typer.echo(f"{path.stem}: not running (cleared stale pidfile)")
            continue
        try:
            pid = int(path.read_text().strip())
        except (ValueError, OSError):
            typer.secho(f"{path.stem}: unreadable pidfile, skipping", fg=typer.colors.YELLOW)
            continue

        _signal(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while _held(path) and time.monotonic() < deadline:
            time.sleep(0.1)

        if _held(path):
            # Windows has no SIGKILL; os.kill with any non-console signal is
            # TerminateProcess there, so SIGTERM already hits as hard.
            _signal(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            typer.secho(f"{path.stem}: killed (pid {pid})", fg=typer.colors.YELLOW)
        else:
            typer.echo(f"{path.stem}: stopped (pid {pid})")
        path.unlink(missing_ok=True)


serve_app = typer.Typer(help="Start a server")
app.add_typer(serve_app, name="serve")


@serve_app.command("mcp")
def serve_mcp() -> None:
    """Start MCP stdio server."""
    from smrti.servers.mcp import run_mcp_server

    run_mcp_server()


@serve_app.command("rest")
def serve_rest(
    host: str = typer.Option("0.0.0.0", help="Host"),
    port: int = typer.Option(8420, help="Port"),
) -> None:
    """Start REST API server."""
    from smrti.servers.rest import run_rest_server

    _warn_if_exposed(host)
    typer.echo(f"Starting Smrti REST API on http://{host}:{port}")
    with _pidfile("rest", port):
        run_rest_server(host=host, port=port)


@serve_app.command("viz")
def serve_viz(
    host: str = typer.Option("127.0.0.1", help="Host"),
    port: int = typer.Option(8420, help="Port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically"),
) -> None:
    """Start the REST API and open the memory visualizer in a browser."""
    import threading
    import time
    import webbrowser

    from smrti.servers.rest import run_rest_server

    _warn_if_exposed(host)
    url = f"http://{host}:{port}/viz"
    typer.echo(f"Starting Smrti visualizer on {url}")

    if not no_browser:
        def _open():
            time.sleep(1.2)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    with _pidfile("viz", port):
        run_rest_server(host=host, port=port)


@serve_app.command("proxy")
def serve_proxy(
    host: str = typer.Option("0.0.0.0", help="Host"),
    port: int = typer.Option(8421, help="Port"),
    upstream: str = typer.Option("", help="Upstream LLM base URL (overrides SMRTI_UPSTREAM_URL)"),
) -> None:
    """Start OpenAI-compatible proxy with transparent memory injection.

    Per-request headers:
      X-Smrti-Tenant-Id    Hard isolation boundary (the human user).
      X-Smrti-Read-Spaces  Comma-separated ordered list of spaces to read from.
      X-Smrti-Write-Space  Space where new memories are stored.
    """
    if upstream:
        os.environ["SMRTI_UPSTREAM_URL"] = upstream.rstrip("/")

    from smrti.servers.proxy import run_proxy_server

    _warn_if_exposed(host)
    effective_upstream = os.environ.get("SMRTI_UPSTREAM_URL", "https://api.openai.com")
    typer.echo(f"Starting Smrti proxy on http://{host}:{port}/v1")
    typer.echo(f"Visualizer:  http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/viz")
    typer.echo(f"Upstream: {effective_upstream}")
    with _pidfile("proxy", port):
        run_proxy_server(host=host, port=port)


@serve_app.command("town")
def serve_town(
    host: str = typer.Option("127.0.0.1", help="Host"),
    port: int = typer.Option(8430, help="Port"),
    db: Optional[str] = typer.Option(None, help="Path to town SQLite database  [env: SMRTI_TOWN_DB]"),
    tenant_id: Optional[str] = typer.Option(None, help="Tenant ID for the town  [env: SMRTI_TOWN_TENANT]"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically"),
) -> None:
    """Start the smrti-town isometric city-builder and open it in a browser.

    An empty field appears. Place the Town Hall, choose a mayor, and watch
    your council debate what to build next.  The simulation uses its own
    Smrti instances with the given DB path.
    """
    import threading
    import time
    import webbrowser

    # Only an explicit option overrides the env var — assigning unconditionally
    # would overwrite SMRTI_TOWN_DB with this command's own default.
    db = db or os.environ.get("SMRTI_TOWN_DB", "~/.smrti/town.db")
    tenant_id = tenant_id or os.environ.get("SMRTI_TOWN_TENANT", "millbrook")
    os.environ["SMRTI_TOWN_DB"] = db
    os.environ["SMRTI_TOWN_TENANT"] = tenant_id

    from smrti_town.server import serve

    url = f"http://{host}:{port}"
    typer.echo(f"Starting smrti-town simulation on {url}")
    typer.echo(f"DB: {os.path.expanduser(db)} | Tenant: {tenant_id}")
    typer.echo(f"Simulation auto-starts when browser connects.")

    if not no_browser:
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    with _pidfile("town", port):
        serve(host=host, port=port)


if __name__ == "__main__":
    app()
