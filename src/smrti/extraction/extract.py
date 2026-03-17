"""LLM-based entity and claim extraction from conversational text."""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Optional

import httpx

from .prompts import EXTRACTION_PROMPT

if TYPE_CHECKING:
    from smrti import Smrti

_http: Optional[httpx.AsyncClient] = None


def _get_http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    return _http


async def extract_knowledge(
    text: str,
    http: httpx.AsyncClient,
    upstream: str,
    auth: str,
    model: str,
) -> Optional[dict]:
    """Call the upstream LLM to extract entities and claims from text.

    Returns a dict with 'entities' and 'claims' lists, or None on failure.
    """
    try:
        resp = await http.post(
            f"{upstream}/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": auth},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
                "max_tokens": 512,
            },
            timeout=30.0,
        )
        data = resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception:
        return None


async def extract_and_link(
    episode_id: str,
    content: str,
    mem: "Smrti",
    auth: str,
    model: str,
    upstream: str,
) -> None:
    """Extract entities/claims from content and link them to the episode atom.

    Shared by all serve modes (proxy, MCP, REST). Silently no-ops if the LLM
    call fails or returns no usable structure.
    """
    from .resolve import EntityResolver

    extracted = await extract_knowledge(content, _get_http(), upstream, auth, model)
    if not extracted:
        return

    def _sync_work() -> None:
        resolver = EntityResolver(mem.db, mem.embed)
        entity_ids: dict[str, str] = {}

        for ent in extracted.get("entities", []):
            name = (ent.get("name") or "").strip()
            etype = ent.get("type", "concept")
            if not name:
                continue
            atom_id = resolver.resolve(name, etype, mem.tenant_id, mem.write_space, [mem.write_space])
            entity_ids[name] = atom_id
            for alias in ent.get("aliases", []):
                if alias and alias != name:
                    resolver.aliases.add(atom_id, alias, mem.tenant_id, mem.write_space)
            mem.atomspace.link_atoms(episode_id, atom_id, "mentions", mem.tenant_id, mem.write_space)

        for claim in extracted.get("claims", []):
            subj_id = entity_ids.get(claim.get("subject", ""))
            obj_id = entity_ids.get(claim.get("object", ""))
            if subj_id and obj_id:
                mem.atomspace.link_atoms(
                    subj_id, obj_id, claim.get("predicate", "related_to"),
                    mem.tenant_id, mem.write_space,
                )

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _sync_work)
