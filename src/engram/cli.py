"""CLI for engram: engram serve mcp|rest, engram init, engram status."""
from __future__ import annotations

import os

import typer

app = typer.Typer(help="Engram memory engine CLI")


@app.command()
def init(
    db: str = typer.Option("~/.engram/memory.db", help="Path to SQLite database"),
    personality: str = typer.Option("balanced", help="Personality preset"),
    agent_id: str = typer.Option("default", help="Agent ID"),
) -> None:
    """Initialize a new Engram memory store."""
    from engram import Engram

    mem = Engram(db_path=db, personality=personality, agent_id=agent_id)
    s = mem.status()
    typer.echo(f"Initialized Engram at {os.path.expanduser(db)}")
    typer.echo(f"Agent: {agent_id} | Personality: {personality}")
    typer.echo(f"Total atoms: {s['total_atoms']}")
    mem.close()


@app.command()
def status(
    db: str = typer.Option("~/.engram/memory.db", help="Path to SQLite database"),
    agent_id: str = typer.Option("default", help="Agent ID"),
) -> None:
    """Show memory statistics."""
    from engram import Engram

    mem = Engram(db_path=db, agent_id=agent_id)
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
    from engram.servers.mcp import run_mcp_server

    run_mcp_server()


@serve_app.command("rest")
def serve_rest(
    host: str = typer.Option("0.0.0.0", help="Host"),
    port: int = typer.Option(8420, help="Port"),
) -> None:
    """Start REST API server."""
    from engram.servers.rest import run_rest_server

    typer.echo(f"Starting Engram REST API on http://{host}:{port}")
    run_rest_server(host=host, port=port)


@serve_app.command("proxy")
def serve_proxy(
    host: str = typer.Option("0.0.0.0", help="Host"),
    port: int = typer.Option(8421, help="Port"),
    upstream: str = typer.Option("", help="Upstream LLM base URL (overrides ENGRAM_UPSTREAM_URL)"),
) -> None:
    """Start OpenAI-compatible proxy server with transparent memory injection."""
    if upstream:
        os.environ["ENGRAM_UPSTREAM_URL"] = upstream.rstrip("/")

    from engram.servers.proxy import run_proxy_server

    effective_upstream = os.environ.get("ENGRAM_UPSTREAM_URL", "https://api.openai.com")
    typer.echo(f"Starting Engram proxy on http://{host}:{port}/v1")
    typer.echo(f"Upstream: {effective_upstream}")
    run_proxy_server(host=host, port=port)


if __name__ == "__main__":
    app()
