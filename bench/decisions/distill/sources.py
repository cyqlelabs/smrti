"""The raw material: conversations and trajectories from the corpora under
``data/``, read into one shape each stage understands.

Nothing here is labelled. What comes out is turns (who said what, when, in
which session) and agent turns (a task, the tool calls and results behind
it, the reply), from:

* LongMemEval-S (``data/longmemeval_s.json``): 500 questions over ~250k
  English chat turns, with the sessions that hold each answer marked.
* HaluMem (``data/HaluMem-Medium.jsonl``): long English dialogues with the
  facts each session established, and which fact updates which.
* Factor transcripts (``data/distill/raw/*-sessions/*.jsonl``): the
  desktop's and the box's own sessions, Spanish and English, with the tool
  calls and results of every turn — the only source that looks like what
  Factor's completion, induction and recovery checks actually see.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

DATA = os.environ.get("SMRTI_DISTILL_DATA", "data")


@dataclass
class Turn:
    text: str
    role: str  # user | assistant
    session: str
    index: int
    date: str = ""
    source: str = ""
    lang: str = "en"


@dataclass
class Session:
    id: str
    turns: list[Turn]
    date: str = ""
    source: str = ""
    # Facts the session established, as sentences (HaluMem carries them).
    facts: list[str] = field(default_factory=list)


@dataclass
class AgentTurn:
    """One of Factor's turns: what was asked, what the tools did, what was
    replied — the shape ``renderTurn`` in Factor's ``induce.go`` produces."""

    task: str
    trajectory: str
    reply: str
    calls: list[tuple[str, str, str]]  # (name, summarised args, result)
    session: str
    source: str
    lang: str = "en"


# ── LongMemEval ────────────────────────────────────────────────────────────


def longmemeval(path: str = f"{DATA}/longmemeval_s.json") -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def longmemeval_sessions(questions: list[dict[str, Any]]) -> dict[str, Session]:
    """Every haystack session once, by id: the same session sits in many
    questions' haystacks."""
    out: dict[str, Session] = {}
    for q in questions:
        for sid, date, turns in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
            if sid in out:
                continue
            out[sid] = Session(
                id=sid, date=date, source="longmemeval",
                turns=[
                    Turn(text=t["content"], role=t["role"], session=sid, index=i, date=date, source="longmemeval")
                    for i, t in enumerate(turns) if t.get("content")
                ],
            )
    return out


# ── HaluMem ────────────────────────────────────────────────────────────────


def halumem(path: str = f"{DATA}/HaluMem-Medium.jsonl") -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def halumem_sessions(records: list[dict[str, Any]]) -> list[Session]:
    out: list[Session] = []
    for r in records:
        for k, s in enumerate(r["sessions"]):
            sid = f"halumem:{r['uuid'][:8]}:{k}"
            out.append(Session(
                id=sid, date=s.get("start_time", ""), source="halumem",
                turns=[
                    Turn(text=t["content"], role=t["role"], session=sid, index=i,
                         date=t.get("timestamp", ""), source="halumem")
                    for i, t in enumerate(s.get("dialogue", [])) if t.get("content")
                ],
                facts=[m["memory_content"] for m in s.get("memory_points", [])],
            ))
    return out


def halumem_updates(records: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """(earlier fact, later fact, persona name) for every fact that updates
    another."""
    out = []
    for r in records:
        name = re.search(r"Name: ([^;]+);", r.get("persona_info", ""))
        who = name.group(1).strip() if name else "the user"
        for s in r["sessions"]:
            for m in s.get("memory_points", []):
                if m.get("is_update") == "True":
                    for old in m.get("original_memories") or []:
                        out.append((old, m["memory_content"], who))
    return out


# ── Factor transcripts ─────────────────────────────────────────────────────

_SPANISH = re.compile(r"\b(que|de|la|el|los|las|para|con|una|por|está|qué|ya|hoy|mañana|dale|gracias)\b", re.I)


def guess_lang(text: str) -> str:
    """Spanish or English, on a handful of function words; enough to
    balance a corpus, not to label a sentence."""
    hits = len(_SPANISH.findall(text))
    return "es" if hits >= 2 and hits >= len(text.split()) / 12 else "en"


def _summarize_args(args: Any) -> str:
    # Factor's summarizeArgs: (k=v, ...) with long values clipped.
    if not isinstance(args, dict) or not args:
        return "()"
    parts = []
    for k, v in args.items():
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        if len(s) > 80:
            s = s[:80] + "…"
        parts.append(f"{k}={s}")
    return "(" + ", ".join(parts) + ")"


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def render_turn(messages: list[dict[str, Any]]) -> str:
    """Factor's ``renderTurn``, byte for byte in shape: user/assistant lines
    clipped at 600, tool calls as ``name(k=v)``, results clipped at 400."""
    out = []
    for m in messages:
        role, content = m.get("role"), (m.get("content") or "")
        if role == "user" and content.strip():
            out.append("user: " + _clip(content, 600))
        elif role == "assistant":
            if content.strip():
                out.append("assistant: " + _clip(content, 600))
            for tc in m.get("tool_calls") or []:
                out.append(f"tool call: {tc.get('name')}{_summarize_args(tc.get('args'))}")
        elif role == "tool" and content.strip():
            out.append("result: " + _clip(content, 400))
    return "\n".join(out) + ("\n" if out else "")


def factor_sessions(pattern: str = f"{DATA}/distill/raw/*-sessions/*.jsonl") -> Iterator[tuple[str, list[dict[str, Any]]]]:
    for path in sorted(glob.glob(pattern)):
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        yield os.path.basename(path), rows


def factor_turns() -> list[Turn]:
    out: list[Turn] = []
    for name, rows in factor_sessions():
        for i, m in enumerate(rows):
            content = (m.get("content") or "").strip()
            if m.get("role") in ("user", "assistant") and content and not content.startswith("[Verification"):
                out.append(Turn(text=content, role=m["role"], session=name, index=i, source="factor",
                                lang=guess_lang(content)))
    return out


def factor_agent_turns() -> list[AgentTurn]:
    """Every turn that used tools: from the user message to the reply that
    closed it."""
    out: list[AgentTurn] = []
    for name, rows in factor_sessions():
        i = 0
        while i < len(rows):
            if rows[i].get("role") != "user":
                i += 1
                continue
            task = (rows[i].get("content") or "").strip()
            j = i + 1
            while j < len(rows) and rows[j].get("role") != "user":
                j += 1
            turn = rows[i + 1 : j]
            calls: list[tuple[str, str, str]] = []
            pending: list[tuple[str, str]] = []
            for m in turn:
                if m.get("role") == "assistant":
                    for tc in m.get("tool_calls") or []:
                        pending.append((tc.get("name") or "?", _summarize_args(tc.get("args"))))
                elif m.get("role") == "tool" and pending:
                    call = pending.pop(0)
                    calls.append((call[0], call[1], (m.get("content") or "")))
            replies = [m for m in turn if m.get("role") == "assistant" and (m.get("content") or "").strip()]
            if calls and replies and task:
                reply = replies[-1]["content"].strip()
                out.append(AgentTurn(
                    task=task, trajectory=render_turn(turn), reply=reply, calls=calls,
                    session=name, source="factor", lang=guess_lang(task + " " + reply),
                ))
            i = j
    return out


def skill_names() -> list[str]:
    names: set[str] = set()
    for path in glob.glob(f"{DATA}/distill/raw/*-skills.txt"):
        with open(path, encoding="utf-8") as fh:
            names.update(fh.read().split())
    return sorted(names)
