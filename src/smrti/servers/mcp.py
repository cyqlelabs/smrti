"""MCP server for smrti (stdio transport)."""
from __future__ import annotations

import asyncio
import dataclasses
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from smrti import Smrti, _write_metadata
from smrti.personality.params import PersonalityProfile, load_preset
from smrti.retrieval.classify import classify_memory
from smrti.servers import config as cfg
from smrti.servers.common import api_root, drain
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
        temporal=cfg.TEMPORAL,
    )


def _coverage(overlap) -> dict:
    """How much of each space an operation's figures describe."""
    return {
        "sampled_a": overlap.sampled_a,
        "size_a": overlap.size_a,
        "sampled_b": overlap.sampled_b,
        "size_b": overlap.size_b,
        "complete": overlap.complete,
    }



def _memory_json(r) -> dict:
    """One recalled memory as the recall tool and route return it."""
    return {
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
        # How useful the memory was judged as evidence for the
        # query when the rerank decision task ran; null when
        # it did not.
        "evidence": r.evidence,
        "space": r.atom.space,
        # Dates the extraction model pinned down for relative
        # expressions in this memory. They live in metadata rather
        # than in the text, which was embedded before the model
        # ever saw it, so this is where the reader meets them.
        "temporal": r.atom.metadata.get("temporal") or [],
    }


def _superseded_json(replaced: dict) -> list[dict]:
    """The superseded claims about the recalled memories, each once."""
    seen: set[str] = set()
    out: list[dict] = []
    for atom_id, claims in replaced.items():
        for claim in claims:
            if claim.edge_id in seen:
                continue
            seen.add(claim.edge_id)
            out.append({
                "about": atom_id,
                "subject": claim.subject,
                "predicate": claim.predicate,
                "old_object": claim.old_object,
                "new_object": claim.new_object,
            })
    return out

def _pairs_json(pairs) -> list[dict]:
    """The matched atom pairs of a space overlap."""
    return [
        {
            "atom_a": {"id": p.atom_a.id, "label": p.atom_a.label, "space": p.atom_a.space},
            "atom_b": {"id": p.atom_b.id, "label": p.atom_b.label, "space": p.atom_b.space},
            "similarity": p.similarity,
        }
        for p in pairs
    ]


def _atoms_json(atoms) -> list[dict]:
    """The atoms a space intersection or difference returns."""
    return [{"id": a.id, "label": a.label, "type": a.type.value, "space": a.space} for a in atoms]


def _remember(mem: Smrti, args: dict) -> dict:
    content = args["content"]
    if mem.is_ignored(content):
        return {"status": "ignored", "atom_id": "", "space": mem.write_space}
    # An absent valence is forwarded as None: the engine reads the tone
    # from the text and records that nobody stated it. That distinction is
    # what keeps stored conversation from minting behavioral constraints —
    # an estimate is a reading of the speaker's mood, not a report of a
    # mistake — so it is resolved in one place rather than here.
    valence = args.get("valence")
    atom_type = args.get("type", "episode")
    # Provenance and valence are read once and handed to both writers. The
    # belief branch used to drop them, so every belief in the graph read as
    # source-less and unemotional however it was stored — which cost
    # beliefs the agent-source discount at ranking and the faster decay
    # that goes with it.
    source = args.get("source", "user")
    intensity = args.get("intensity")
    if atom_type == "belief":
        atom_id = mem.believe(
            statement=content,
            probability=args.get("probability", 0.8),
            evidence=args.get("evidence"),
            valence=valence,
            intensity=intensity,
            source=source,
        )
    else:
        atom_id = mem.remember(
            content=content,
            type=atom_type,
            probability=args.get("probability", 0.8),
            valence=valence,
            intensity=intensity,
            metadata=_write_metadata(source, False) or None,
        )
    return {"status": "ok", "atom_id": atom_id, "space": mem.write_space}


def _recall(mem: Smrti, args: dict) -> dict:
    # An absent floor is the personality's: min_confidence_to_surface is a
    # promise the preset makes, and it is kept here rather than replaced
    # by a literal the tool schema happened to carry.
    results = mem.recall(
        query=args["query"],
        top_k=args.get("top_k", 10),
        min_confidence=args.get("min_confidence"),
        read_spaces=args.get("read_spaces") or None,
        boost=args.get("boost", True),
        rerank=args.get("rerank", True),
    )
    shown = {r.atom.id for r in results}
    # Stated warnings the recall did not return — outranked in the
    # candidate pool or cut by top_k — come back apart from it, so a
    # warning reaches the caller whatever top_k it asked for.
    warnings = [
        w for w in mem.find_warnings(args["query"], read_spaces=args.get("read_spaces") or None)
        if w.atom.id not in shown
    ]
    replaced = mem.superseded_claims(
        [r.atom.id for r in results], read_spaces=args.get("read_spaces") or None
    )
    return {
        "memories": [_memory_json(r) for r in results],
        "warnings": [_memory_json(w) for w in warnings],
        "superseded": _superseded_json(replaced),
    }


def _attend(mem: Smrti, args: dict) -> dict:
    # The access boost on its own, for a caller that recalled a wide
    # set without boosting, reranked it itself, and kept a few.
    atom_ids = args.get("atom_ids") or []
    return {"status": "ok", "space": mem.write_space, "boosted": mem.attend(atom_ids)}


def _reinforce(mem: Smrti, args: dict) -> dict:
    atom_ids = args.get("atom_ids") or []
    weight = args.get("weight")
    result = (
        mem.reinforce(atom_ids)
        if weight is None
        else mem.reinforce(atom_ids, weight)
    )
    return {"status": "ok", "space": mem.write_space, **result}


def _reflect(mem: Smrti, args: dict) -> dict:
    return mem.reflect().model_dump()


def _believe(mem: Smrti, args: dict) -> dict:
    atom_id = mem.believe(
        statement=args["statement"],
        probability=args["probability"],
        evidence=args.get("evidence"),
        valence=args.get("valence"),
        intensity=args.get("intensity"),
        source=args.get("source", "user"),
    )
    return {"status": "ok", "atom_id": atom_id, "space": mem.write_space}


def _forget(mem: Smrti, args: dict) -> dict:
    atom_ids = args.get("atom_ids")
    query = args.get("query")
    cascade = args.get("cascade") is True
    if atom_ids:
        forgotten = mem.forget(atom_ids=list(atom_ids), cascade=cascade)
    elif query and query.strip():
        forgotten = mem.forget(query=query, top_k=5, cascade=cascade)
    else:
        return {"error": "smrti_forget needs a non-empty query or atom_ids"}
    result = {"status": "ok", "softened": forgotten}
    if args.get("reason"):
        result["reason"] = args["reason"]
    return result


def _personality(mem: Smrti, args: dict) -> dict:
    action = args["action"]
    if action == "get":
        row = mem.db.fetchone(
            "SELECT * FROM personality WHERE tenant_id = ? AND space = ?",
            (mem.tenant_id, mem.write_space),
        )
        return dict(row) if row else {}
    if action in ("set", "preset"):
        return _set_personality(mem, args.get("preset"), args.get("params"))
    return {"error": f"Unknown action: {action}"}


def _set_personality(mem: Smrti, preset, params) -> dict:
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
    try:
        mem.set_personality(preset)
    except ValueError as exc:  # an unknown preset name
        return {"error": str(exc)}
    return {"status": "ok", "preset": preset}


def _status(mem: Smrti, args: dict) -> dict:
    from smrti import __version__
    result = mem.status()
    result["spaces"] = mem.list_spaces()
    result["version"] = __version__
    return result


def _query_overlap(mem: Smrti, other_space: str, threshold: float) -> dict:
    result = mem.space_overlap(other_space=other_space, threshold=threshold)
    return {
        "space_a": result.space_a,
        "space_b": result.space_b,
        "jaccard": result.jaccard,
        "coverage": _coverage(result),
        "matched_pairs": _pairs_json(result.pairs),
    }


def _query_intersection(mem: Smrti, other_space: str, threshold: float) -> dict:
    result = mem.space_intersection(other_space=other_space, threshold=threshold)
    return {
        "operation": result.operation,
        "spaces": result.spaces,
        "atoms": _atoms_json(result.atoms),
        "jaccard": result.overlap.jaccard if result.overlap else 0.0,
        "coverage": _coverage(result.overlap) if result.overlap else None,
    }


def _query_diff(mem: Smrti, other_space: str, threshold: float) -> dict:
    result = mem.space_difference(other_space=other_space, threshold=threshold)
    return {
        "operation": result.operation,
        "spaces": result.spaces,
        "atoms": _atoms_json(result.atoms),
    }


_SPACE_QUERIES = {
    "overlap": _query_overlap,
    "intersection": _query_intersection,
    "diff": _query_diff,
}


def _space_query(mem: Smrti, args: dict) -> dict:
    op = args["op"]
    other_space = args["other_space"]
    threshold = args.get("threshold", 0.85)
    query = _SPACE_QUERIES.get(op)
    if query is None:
        return {"error": f"Unknown op: {op}"}
    return query(mem, other_space, threshold)


# The legacy space handlers answer without the coverage figures
# smrti_space_query added, as they always have.
def _space_overlap(mem: Smrti, args: dict) -> dict:
    result = mem.space_overlap(
        other_space=args["other_space"],
        threshold=args.get("threshold", 0.85),
    )
    return {
        "space_a": result.space_a,
        "space_b": result.space_b,
        "jaccard": result.jaccard,
        "matched_pairs": _pairs_json(result.pairs),
    }


def _space_intersection(mem: Smrti, args: dict) -> dict:
    result = mem.space_intersection(
        other_space=args["other_space"],
        threshold=args.get("threshold", 0.85),
    )
    return {
        "operation": result.operation,
        "spaces": result.spaces,
        "atoms": _atoms_json(result.atoms),
        "jaccard": result.overlap.jaccard if result.overlap else 0.0,
    }


def _space_diff(mem: Smrti, args: dict) -> dict:
    return _query_diff(mem, args["other_space"], args.get("threshold", 0.85))


def _space_merge(mem: Smrti, args: dict) -> dict:
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


def _list_spaces(mem: Smrti, args: dict) -> dict:
    return {"spaces": mem.list_spaces()}


_HANDLERS = {
    "smrti_remember": _remember,
    "smrti_recall": _recall,
    "smrti_attend": _attend,
    "smrti_reinforce": _reinforce,
    "smrti_reflect": _reflect,
    "smrti_believe": _believe,
    "smrti_forget": _forget,
    "smrti_personality": _personality,
    "smrti_status": _status,
    "smrti_space_query": _space_query,
    "smrti_space_merge": _space_merge,
    # Legacy handlers retained for backward compatibility (REST routes, direct callers)
    "smrti_space_overlap": _space_overlap,
    "smrti_space_intersection": _space_intersection,
    "smrti_space_diff": _space_diff,
    "smrti_list_spaces": _list_spaces,
}


def handle_tool(mem: Smrti, name: str, args: dict) -> dict:
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    return handler(mem, args)


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
        if "error" in result:
            # The SDK reports a raised error as a tool error (isError), which
            # is how a client tells a refused call from a result: returned as
            # a plain dict, a refused personality change read as a success.
            raise ValueError(result["error"])
        if name == "smrti_remember" and cfg.EXTRACT:
            episode_id = result.get("atom_id", "")
            content = arguments.get("content", "")
            if episode_id and content:
                from smrti.extraction.extract import extract_and_link_serialized
                task = asyncio.create_task(
                    extract_and_link_serialized(
                        episode_id, content, mem, "", cfg.EXTRACT_MODEL, api_root(cfg.EXTRACT_URL),
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
            # Extractions still running are graphs for memories the client
            # was already told it stored; they used to be dropped here.
            await drain(_background_tasks)

    asyncio.run(_main())
