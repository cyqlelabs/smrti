"""MCP server for smrti (stdio transport)."""
from __future__ import annotations

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from smrti import Smrti
from smrti.extraction.sentiment import estimate_valence
from smrti.retrieval.classify import classify_memory
from smrti.servers import config as cfg
from smrti.servers.tools import TOOLS
from smrti.servers.reflect_loop import run_reflect_loop


def create_smrti() -> Smrti:
    return Smrti(
        db_path=cfg.DB,
        personality=cfg.PERSONALITY,
        tenant_id=cfg.TENANT_ID,
        write_space=cfg.SPACE,
        read_spaces=cfg.READ_SPACES,
        ignore_patterns=cfg.IGNORE_PATTERNS or None,
    )


def handle_tool(mem: Smrti, name: str, args: dict) -> dict:
    if name == "smrti_remember":
        content = args["content"]
        if mem.is_ignored(content):
            return {"status": "ignored", "atom_id": ""}
        valence = args.get("valence")
        if valence is None or valence == 0.0:
            valence = estimate_valence(content, mem.embed)
        atom_id = mem.remember(
            content=content,
            type=args.get("type", "episode"),
            probability=args.get("probability", 0.8),
            valence=valence,
        )
        return {"status": "ok", "atom_id": atom_id}

    elif name == "smrti_recall":
        results = mem.recall(
            query=args["query"],
            top_k=args.get("top_k", 10),
            min_confidence=args.get("min_confidence", 0.1),
        )
        return {
            "memories": [
                {
                    "id": r.atom.id,
                    "label": r.atom.label,
                    "content": r.atom.content,
                    "type": r.atom.type.value,
                    "probability": r.atom.truth.probability,
                    "confidence": r.atom.truth.confidence,
                    "sti": r.atom.attention.sti,
                    "lti": r.atom.attention.lti,
                    "valence": r.atom.valence.valence,
                    "intensity": r.atom.valence.intensity,
                    "severity": classify_memory(r),
                    "salience": r.salience,
                    "similarity": r.similarity,
                    "space": r.atom.space,
                }
                for r in results
            ]
        }

    elif name == "smrti_reflect":
        result = mem.reflect()
        return result.model_dump()

    elif name == "smrti_believe":
        atom_id = mem.believe(
            statement=args["statement"],
            probability=args["probability"],
            evidence=args.get("evidence"),
        )
        return {"status": "ok", "atom_id": atom_id}

    elif name == "smrti_forget":
        forgotten = mem.forget(query=args["query"], top_k=5)
        return {"status": "ok", "softened": forgotten}

    elif name == "smrti_personality":
        action = args["action"]
        if action == "get":
            row = mem.db.fetchone(
                "SELECT * FROM personality WHERE tenant_id = ? AND space = ?",
                (mem.tenant_id, mem.write_space),
            )
            return dict(row) if row else {}
        elif action in ("set", "preset"):
            preset = args.get("preset") or "balanced"
            mem.set_personality(preset)
            return {"status": "ok", "preset": preset}

    elif name == "smrti_status":
        return mem.status()

    return {"error": f"Unknown tool: {name}"}


def run_mcp_server() -> None:
    """Run the MCP server over stdio."""
    server = Server("smrti")
    mem = create_smrti()

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        result = handle_tool(mem, name, arguments)
        if name == "smrti_remember" and cfg.EXTRACT:
            episode_id = result.get("atom_id", "")
            content = arguments.get("content", "")
            if episode_id and content:
                from smrti.extraction.extract import extract_and_link_hybrid
                asyncio.create_task(
                    extract_and_link_hybrid(episode_id, content, mem, "", cfg.EXTRACT_MODEL, cfg.EXTRACT_URL, mode=cfg.EXTRACT_MODE)
                )
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    async def _main() -> None:
        task = asyncio.create_task(run_reflect_loop(lambda: [mem]))
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )
        finally:
            task.cancel()

    asyncio.run(_main())
