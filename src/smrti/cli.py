"""CLI for smrti: smrti serve mcp|rest|proxy, smrti init, smrti status."""
from __future__ import annotations

import os

import typer

app = typer.Typer(help="Smrti memory engine CLI")

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


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
    db: str = typer.Option("~/.smrti/memory.db", help="Path to SQLite database"),
    personality: str = typer.Option("balanced", help="Personality preset"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
    space: str = typer.Option("default", help="Memory space"),
) -> None:
    """Initialize a new Smrti memory store."""
    from smrti import Smrti

    mem = Smrti(db_path=db, personality=personality, tenant_id=tenant_id, write_space=space)
    s = mem.status()
    typer.echo(f"Initialized Smrti at {os.path.expanduser(db)}")
    typer.echo(f"Tenant: {tenant_id} | Space: {space} | Personality: {personality}")
    typer.echo(f"Total atoms: {s['total_atoms']}")
    mem.close()


@app.command()
def status(
    db: str = typer.Option("~/.smrti/memory.db", help="Path to SQLite database"),
    tenant_id: str = typer.Option("default", help="Tenant ID"),
    space: str = typer.Option("default", help="Memory space"),
) -> None:
    """Show memory statistics."""
    from smrti import Smrti
    from smrti.servers import config as cfg

    mem = Smrti(db_path=db, personality=cfg.PERSONALITY, tenant_id=tenant_id, write_space=space)
    s = mem.status()
    typer.echo(f"Total atoms: {s['total_atoms']}")
    typer.echo(f"By type: {s['by_type']}")
    p = s.get("personality", {})
    if p:
        typer.echo(f"Personality: {p.get('preset_name', 'custom')}")
    mem.close()


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
    run_proxy_server(host=host, port=port)


@serve_app.command("town")
def serve_town(
    host: str = typer.Option("127.0.0.1", help="Host"),
    port: int = typer.Option(8430, help="Port"),
    db: str = typer.Option("~/.smrti/town.db", help="Path to town SQLite database"),
    tenant_id: str = typer.Option("millbrook", help="Tenant ID for the town"),
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

    serve(host=host, port=port)


if __name__ == "__main__":
    app()
