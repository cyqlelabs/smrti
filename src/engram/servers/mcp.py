"""MCP server for engram (stdio transport)."""
from __future__ import annotations

import asyncio
import json
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from engram import Engram
from engram.servers.tools import TOOLS


def create_engram() -> Engram:
    db_path = os.environ.get("ENGRAM_DB", "~/.engram/memory.db")
    personality = os.environ.get("ENGRAM_PERSONALITY", "balanced")
    tenant_id = os.environ.get("ENGRAM_TENANT_ID", "default")
    write_space = os.environ.get("ENGRAM_SPACE", "default")
    read_spaces_raw = os.environ.get("ENGRAM_READ_SPACES", "")
    read_spaces = [s.strip() for s in read_spaces_raw.split(",") if s.strip()] or None
    return Engram(
        db_path=db_path,
        personality=personality,
        tenant_id=tenant_id,
        write_space=write_space,
        read_spaces=read_spaces,
    )


def handle_tool(mem: Engram, name: str, args: dict) -> dict:
    if name == "engram_remember":
        atom_id = mem.remember(
            content=args["content"],
            type=args.get("type", "episode"),
            probability=args.get("probability", 0.8),
            valence=args.get("valence", 0.0),
        )
        return {"status": "ok", "atom_id": atom_id}

    elif name == "engram_recall":
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
                    "salience": r.salience,
                    "similarity": r.similarity,
                    "space": r.atom.space,
                }
                for r in results
            ]
        }

    elif name == "engram_reflect":
        result = mem.reflect()
        return result.model_dump()

    elif name == "engram_believe":
        atom_id = mem.believe(
            statement=args["statement"],
            probability=args["probability"],
            evidence=args.get("evidence"),
        )
        return {"status": "ok", "atom_id": atom_id}

    elif name == "engram_forget":
        results = mem.recall(query=args["query"], top_k=5)
        forgotten = []
        for r in results:
            mem.db.execute(
                "UPDATE atoms SET confidence = confidence * 0.3 WHERE id = ?",
                (r.atom.id,),
            )
            forgotten.append(r.atom.label)
        return {"status": "ok", "softened": forgotten}

    elif name == "engram_personality":
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

    elif name == "engram_status":
        return mem.status()

    return {"error": f"Unknown tool: {name}"}


def run_mcp_server() -> None:
    """Run the MCP server over stdio."""
    server = Server("engram")
    mem = create_engram()

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
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_main())
