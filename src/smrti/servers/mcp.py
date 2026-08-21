"""MCP server for smrti (stdio transport)."""
from __future__ import annotations

import asyncio
import dataclasses
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from smrti import Smrti
from smrti.extraction.sentiment import estimate_valence
from smrti.personality.params import PersonalityProfile, load_preset
from smrti.retrieval.classify import classify_memory
from smrti.servers import config as cfg
from smrti.servers.tools import TOOLS
from smrti.servers.reflect_loop import run_reflect_loop

# Strong references to fire-and-forget tasks so the GC cannot cancel them mid-flight
_background_tasks: set[asyncio.Task] = set()


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
            return {"status": "ignored", "atom_id": "", "space": mem.write_space}
        valence = args.get("valence")
        if valence is None:
            valence = estimate_valence(content, mem.embed)
        atom_type = args.get("type", "episode")
        # Provenance and valence are read once and handed to both writers. The
        # belief branch used to drop them, so every belief in the graph read as
        # source-less and unemotional however it was stored — which cost
        # beliefs the agent-source discount at ranking and the faster decay
        # that goes with it.
        source = args.get("source", "user")
        if atom_type == "belief":
            atom_id = mem.believe(
                statement=content,
                probability=args.get("probability", 0.8),
                evidence=args.get("evidence"),
                valence=valence,
                source=source,
            )
        else:
            atom_id = mem.remember(
                content=content,
                type=atom_type,
                probability=args.get("probability", 0.8),
                valence=valence,
                metadata={"source": "agent"} if source == "agent" else None,
            )
        return {"status": "ok", "atom_id": atom_id, "space": mem.write_space}

    elif name == "smrti_recall":
        results = mem.recall(
            query=args["query"],
            top_k=args.get("top_k", 10),
            min_confidence=args.get("min_confidence", 0.1),
            read_spaces=args.get("read_spaces") or None,
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
        return {"status": "ok", "atom_id": atom_id, "space": mem.write_space}

    elif name == "smrti_forget":
        forgotten = mem.forget(query=args["query"], top_k=5)
        result = {"status": "ok", "softened": forgotten}
        if args.get("reason"):
            result["reason"] = args["reason"]
        return result

    elif name == "smrti_personality":
        action = args["action"]
        if action == "get":
            row = mem.db.fetchone(
                "SELECT * FROM personality WHERE tenant_id = ? AND space = ?",
                (mem.tenant_id, mem.write_space),
            )
            return dict(row) if row else {}
        elif action in ("set", "preset"):
            preset = args.get("preset")
            params = args.get("params")
            if params:
                try:
                    base = load_preset(preset) if preset else PersonalityProfile()
                    profile = dataclasses.replace(base, **params)
                except (TypeError, ValueError) as exc:
                    return {"error": f"Invalid personality params: {exc}"}
                mem.set_personality_profile(profile, preset or "custom")
                return {"status": "ok", "preset": preset or "custom"}
            if not preset:
                return {"error": "action requires a 'preset' or 'params' argument"}
            mem.set_personality(preset)
            return {"status": "ok", "preset": preset}
        return {"error": f"Unknown action: {action}"}

    elif name == "smrti_status":
        from smrti import __version__
        result = mem.status()
        result["spaces"] = mem.list_spaces()
        result["version"] = __version__
        return result

    elif name == "smrti_space_query":
        op = args["op"]
        other_space = args["other_space"]
        threshold = args.get("threshold", 0.85)
        if op == "overlap":
            result = mem.space_overlap(other_space=other_space, threshold=threshold)
            return {
                "space_a": result.space_a,
                "space_b": result.space_b,
                "jaccard": result.jaccard,
                "matched_pairs": [
                    {
                        "atom_a": {"id": p.atom_a.id, "label": p.atom_a.label, "space": p.atom_a.space},
                        "atom_b": {"id": p.atom_b.id, "label": p.atom_b.label, "space": p.atom_b.space},
                        "similarity": p.similarity,
                    }
                    for p in result.pairs
                ],
            }
        elif op == "intersection":
            result = mem.space_intersection(other_space=other_space, threshold=threshold)
            return {
                "operation": result.operation,
                "spaces": result.spaces,
                "atoms": [
                    {"id": a.id, "label": a.label, "type": a.type.value, "space": a.space}
                    for a in result.atoms
                ],
                "jaccard": result.overlap.jaccard if result.overlap else 0.0,
            }
        elif op == "diff":
            result = mem.space_difference(other_space=other_space, threshold=threshold)
            return {
                "operation": result.operation,
                "spaces": result.spaces,
                "atoms": [
                    {"id": a.id, "label": a.label, "type": a.type.value, "space": a.space}
                    for a in result.atoms
                ],
            }
        return {"error": f"Unknown op: {op}"}

    # Legacy handlers retained for backward compatibility (REST routes, direct callers)
    elif name == "smrti_space_overlap":
        result = mem.space_overlap(
            other_space=args["other_space"],
            threshold=args.get("threshold", 0.85),
        )
        return {
            "space_a": result.space_a,
            "space_b": result.space_b,
            "jaccard": result.jaccard,
            "matched_pairs": [
                {
                    "atom_a": {"id": p.atom_a.id, "label": p.atom_a.label, "space": p.atom_a.space},
                    "atom_b": {"id": p.atom_b.id, "label": p.atom_b.label, "space": p.atom_b.space},
                    "similarity": p.similarity,
                }
                for p in result.pairs
            ],
        }

    elif name == "smrti_space_intersection":
        result = mem.space_intersection(
            other_space=args["other_space"],
            threshold=args.get("threshold", 0.85),
        )
        return {
            "operation": result.operation,
            "spaces": result.spaces,
            "atoms": [
                {"id": a.id, "label": a.label, "type": a.type.value, "space": a.space}
                for a in result.atoms
            ],
            "jaccard": result.overlap.jaccard if result.overlap else 0.0,
        }

    elif name == "smrti_space_diff":
        result = mem.space_difference(
            other_space=args["other_space"],
            threshold=args.get("threshold", 0.85),
        )
        return {
            "operation": result.operation,
            "spaces": result.spaces,
            "atoms": [
                {"id": a.id, "label": a.label, "type": a.type.value, "space": a.space}
                for a in result.atoms
            ],
        }

    elif name == "smrti_space_merge":
        count = mem.materialize_bridge(
            other_space=args["other_space"],
            threshold=args.get("threshold", 0.85),
            min_jaccard=args.get("min_jaccard", 0.1),
        )
        return {
            "status": "ok",
            "bridges_created": count,
            "bridge_space": f"{'_x_'.join(sorted([mem.write_space, args['other_space']]))}",
        }

    elif name == "smrti_list_spaces":
        return {"spaces": mem.list_spaces()}

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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, handle_tool, mem, name, arguments)
        if name == "smrti_remember" and cfg.EXTRACT:
            episode_id = result.get("atom_id", "")
            content = arguments.get("content", "")
            if episode_id and content:
                from smrti.extraction.extract import extract_and_link_serialized
                task = asyncio.create_task(
                    extract_and_link_serialized(
                        episode_id, content, mem, "", cfg.EXTRACT_MODEL, cfg.EXTRACT_URL,
                        arguments.get("source", "user"), mode=cfg.EXTRACT_MODE,
                    )
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
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
