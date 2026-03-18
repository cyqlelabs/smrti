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
    entity_context: str = "",
) -> Optional[dict]:
    """Call the upstream LLM to extract entities and claims from text.

    Returns a dict with 'entities' and 'claims' lists, or None on failure.
    """
    if entity_context:
        user_content = (
            f"[Known entities — use these to resolve pronouns and references]\n"
            f"{entity_context}\n\n"
            f"[Text to extract]\n{text}"
        )
    else:
        user_content = text
    try:
        resp = await http.post(
            f"{upstream}/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": auth},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
                "max_tokens": 1024,
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


_COREF_TYPES = {"person", "organization", "project", "tool", "location", "event", "goal"}


def _build_entity_context(mem: "Smrti") -> str:
    """Return a compact list of salient named entities from the memory graph.

    Used to ground coreference resolution in the extraction prompt — e.g. so
    that "I" resolves to "Nico" even when the name isn't in the current message.
    Covers all entity types that can plausibly be referenced by a pronoun or
    short noun phrase: person, organization, project, tool, location, event, goal.
    """
    rows = mem.db.fetchall(
        """SELECT label, entity_type
           FROM atoms
           WHERE tenant_id = ? AND space = ? AND type = 'concept'
             AND source_id IS NULL AND entity_type IS NOT NULL
           ORDER BY (sti + lti) DESC
           LIMIT 30""",
        (mem.tenant_id, mem.write_space),
    )
    lines = []
    for row in rows:
        etype = row["entity_type"] or ""
        if etype in _COREF_TYPES:
            lines.append(f"- {row['label']} ({etype})")
    return "\n".join(lines)


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

    entity_context = await asyncio.get_running_loop().run_in_executor(
        None, _build_entity_context, mem
    )
    extracted = await extract_knowledge(content, _get_http(), upstream, auth, model, entity_context)
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
            entity_ids.setdefault(name.lower(), atom_id)
            for alias in ent.get("aliases", []):
                if alias and alias.lower() != name.lower():
                    resolver.aliases.add(atom_id, alias, mem.tenant_id, mem.write_space)
                    entity_ids.setdefault(alias, atom_id)
                    entity_ids.setdefault(alias.lower(), atom_id)
            mem.atomspace.link_atoms(episode_id, atom_id, "mentions", mem.tenant_id, mem.write_space)

        for claim in extracted.get("claims", []):
            subj_raw = claim.get("subject", "")
            obj_raw = claim.get("object", "")
            subj_id = entity_ids.get(subj_raw) or entity_ids.get(subj_raw.lower())
            obj_id = entity_ids.get(obj_raw) or entity_ids.get(obj_raw.lower())
            if subj_id and obj_id:
                mem.atomspace.link_atoms(
                    subj_id, obj_id, claim.get("predicate", "related_to"),
                    mem.tenant_id, mem.write_space,
                )

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _sync_work)
