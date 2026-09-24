"""States for every head, in the exact shape the engine sends the teacher.

Each builder produces the ``state`` dict a decision site constructs —
``route_extraction``'s ``{message, author, known_context}``, a rerank
candidate's ``{question, candidates: [{ref, text, type, written_at,
author}]}``, Factor's ``{task, trajectory, reply}`` — from the corpora in
:mod:`sources` and the generated material in :mod:`augment`. The state is
what the student will read; nothing about the question rides in it.

Sizes are set by ``--scale`` (1.0 is the default corpus, about 80k states
across the seven heads); halve it for a quick run. Every state carries a
stable id, so re-running appends what is new and keeps what was labelled.

Held out entirely: ``bench/decisions/gate_set.json`` and ``tone_set.json``,
the hand-labelled sets the student is measured against.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from smrti.decisions.student import registry

from . import augment, sources
from .sources import DATA, AgentTurn, Session, Turn

logger = logging.getLogger("distill.corpus")

STATES = Path(DATA) / "distill" / "states"


@dataclass
class State:
    task: str
    state: dict[str, Any]
    lang: str
    source: str
    # What the corpus builder believes about the item, for stratified
    # evaluation only; never a training target.
    hint: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        payload = json.dumps(self.state, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{self.task}\n{payload}".encode()).hexdigest()[:20]

    def row(self) -> dict[str, Any]:
        return {"id": self.id, "task": self.task, "lang": self.lang, "source": self.source,
                "hint": self.hint, "state": self.state}


# ── shared helpers ─────────────────────────────────────────────────────────


def _rng(name: str) -> random.Random:
    return random.Random(f"distill/{name}")


def _sample(items: list, n: int, rng: random.Random) -> list:
    return items if len(items) <= n else rng.sample(items, n)


def _sentences(text: str) -> list[str]:
    out, cur = [], []
    for ch in text:
        cur.append(ch)
        if ch in ".!?\n" and len(cur) > 12:
            out.append("".join(cur).strip())
            cur = []
    if cur:
        out.append("".join(cur).strip())
    return [s for s in out if 8 <= len(s) <= 400]


def _facts_of(session: Session, upto: int | None, rng: random.Random) -> str:
    """What the engine would already know: the session's own facts when the
    corpus carries them, else the first sentence of earlier user turns."""
    lines = list(session.facts)
    if not lines:
        lines = [
            _sentences(t.text)[0]
            for t in session.turns[: upto if upto is not None else len(session.turns)]
            if t.role == "user" and _sentences(t.text)
        ]
    if not lines:
        return ""
    k = rng.randint(0, min(8, len(lines)))
    return "\n".join(rng.sample(lines, k))


def _translated(turns: list[Turn], n: int, rng: random.Random) -> list[Turn]:
    """*n* English turns copied into the target languages, by
    ``augment.LANGS``' weights, through the cache."""
    picked = _sample([t for t in turns if t.lang == "en" and len(t.text) <= 700], n, rng)
    langs = augment.split_by_language(len(picked), rng)
    out: list[Turn] = []
    for lang in augment.LANGS:
        group = [t for t, l in zip(picked, langs) if l == lang]
        for t, text in zip(group, augment.translate([t.text for t in group], lang)):
            out.append(Turn(text=text, role=t.role, session=t.session, index=t.index, date=t.date,
                            source=f"{t.source}+{lang}", lang=lang))
    return out


def _synth(kind: str, n: int, rng: random.Random) -> list[tuple[str, dict[str, Any]]]:
    """(lang, item) for *n* generated items: half English, the rest by
    ``augment.LANGS``' weights."""
    counts: dict[str, int] = {"en": n // 2}
    for lang in augment.split_by_language(n - n // 2, rng):
        counts[lang] = counts.get(lang, 0) + 1
    out: list[tuple[str, dict[str, Any]]] = []
    for lang, k in counts.items():
        if k >= 5:
            out.extend((lang, item) for item in augment.synth(kind, lang, k))
    return out


# ── routing ────────────────────────────────────────────────────────────────


def build_routing(scale: float) -> list[State]:
    rng = _rng("routing")
    out: list[State] = []
    lme = sources.longmemeval_sessions(sources.longmemeval())
    hal = sources.halumem_sessions(sources.halumem())
    by_id = {**lme, **{s.id: s for s in hal}}

    def add(turn: Turn, session: Session | None, hint: str) -> None:
        known = ""
        if session is not None and rng.random() < 0.6:
            known = _facts_of(session, turn.index, rng)
        out.append(State("routing", {
            "message": turn.text[:2000],
            "author": "assistant" if turn.role == "assistant" else "user",
            "known_context": known[:3000],
        }, turn.lang, turn.source, {"kind": hint}))

    corpus_turns = [t for s in list(lme.values()) + hal for t in s.turns if 3 <= len(t.text) <= 2000]
    for t in _sample(corpus_turns, int(14000 * scale), rng):
        add(t, by_id.get(t.session), "corpus")
    for t in _translated(corpus_turns, int(5000 * scale), rng):
        add(t, by_id.get(t.session), "corpus")
    for t in sources.factor_turns():
        add(t, None, "factor")
    for lang, item in _synth("message", int(5000 * scale), rng):
        out.append(State("routing", {
            "message": str(item["text"])[:2000],
            "author": "assistant" if item.get("author") == "assistant" else "user",
            "known_context": "",
        }, lang, "synth", {"kind": str(item.get("kind"))}))
    return out


# ── tone ───────────────────────────────────────────────────────────────────


def build_tone(scale: float) -> list[State]:
    """Sentences the sentiment estimator would flag, since only those reach
    the tone check: every negative-enough sentence of the corpora, and the
    generated pairs that make the distinction hard."""
    import numpy as np
    from smrti.core.embed import get_embedding_provider
    from smrti.extraction import sentiment

    rng = _rng("tone")
    embed = get_embedding_provider()
    neg, pos = (np.asarray(v) for v in sentiment._ensure_anchors(embed))
    neg /= np.linalg.norm(neg, axis=1, keepdims=True)
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)

    def valences(vecs: list[list[float]]) -> np.ndarray:
        """``sentiment.estimate_valence`` over a batch, from its vectors:
        the mean cosine to each anchor set, the difference scaled by three,
        zero inside the dead zone."""
        m = np.asarray(vecs)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        diff = (m @ pos.T).mean(1) - (m @ neg.T).mean(1)
        scaled = np.clip(diff * 3.0, -1.0, 1.0)
        return np.where(np.abs(diff) < 0.03, 0.0, scaled)
    pool: list[Turn] = []
    for s in list(sources.longmemeval_sessions(sources.longmemeval()).values()) + sources.halumem_sessions(sources.halumem()):
        for t in s.turns:
            for sent in _sentences(t.text)[:3]:
                pool.append(Turn(text=sent, role=t.role, session=s.id, index=t.index, source=t.source))
    pool = _sample(pool, int(40000 * scale), rng)
    pool += [Turn(text=s, role=t.role, session=t.session, index=t.index, source=t.source, lang=t.lang)
             for t in sources.factor_turns() for s in _sentences(t.text)[:4]]
    negative: list[Turn] = []
    texts = [t.text for t in pool]
    for i in range(0, len(texts), 256):
        vecs = embed.embed_batch(texts[i : i + 256])
        for t, v in zip(pool[i : i + 256], valences(vecs)):
            if v <= -0.3:
                negative.append(t)
    logger.info("tone: %d of %d sentences read negative", len(negative), len(pool))
    out = [State("tone", {"text": t.text[:2000], "author": "assistant" if t.role == "assistant" else "user",
                          "kind": "episode"}, t.lang, t.source, {"about": "corpus"}) for t in negative]
    for t in _translated(negative, int(min(len(negative), 3000 * scale)), rng):
        out.append(State("tone", {"text": t.text[:2000], "author": "assistant" if t.role == "assistant" else "user",
                                  "kind": "episode"}, t.lang, t.source, {"about": "corpus"}))
    for lang, item in _synth("tone", int(3000 * scale), rng):
        out.append(State("tone", {"text": str(item["text"])[:2000],
                                  "author": "assistant" if item.get("author") == "assistant" else "user",
                                  "kind": rng.choice(["episode", "episode", "belief"])},
                         lang, "synth", {"about": str(item.get("about"))}))
    return out


# ── rerank ─────────────────────────────────────────────────────────────────


def _candidate(turn: Turn, ref: str = "c0") -> dict[str, Any]:
    return {"ref": ref, "text": turn.text[:400], "type": "episode", "written_at": turn.date or "",
            "author": "agent" if turn.role == "assistant" else "user"}


def build_rerank(scale: float) -> list[State]:
    """Question/candidate pairs at every distance: turns from the sessions
    that hold the answer, turns from the same haystack, turns from anywhere
    — the teacher says which are evidence."""
    rng = _rng("rerank")
    questions = sources.longmemeval()
    sessions = sources.longmemeval_sessions(questions)
    all_turns = [t for s in sessions.values() for t in s.turns if len(t.text) >= 20]
    out: list[State] = []
    per_question = max(6, int(40 * scale))
    for q in questions:
        gold = [t for sid in q["answer_session_ids"] for t in sessions[sid].turns if len(t.text) >= 20]
        hay = [t for sid in q["haystack_session_ids"] if sid not in q["answer_session_ids"]
               for t in sessions[sid].turns if len(t.text) >= 20]
        picked = _sample(gold, per_question // 2, rng) + _sample(hay, per_question // 4, rng) \
            + _sample(all_turns, per_question - per_question // 2 - per_question // 4, rng)
        for t in picked:
            out.append(State("rerank", {"question": q["question"], "candidates": [_candidate(t)]},
                             "en", "longmemeval", {"gold": t.session in q["answer_session_ids"],
                                                   "qtype": q["question_type"]}))
    # Pairs in the other languages: translated questions over translated
    # candidates, and a share where only one side is translated, since a
    # memory is often stored in a different language than it is asked in.
    picked = _sample(out, int(4000 * scale), rng)
    langs = augment.split_by_language(len(picked), rng)
    for lang in augment.LANGS:
        group = [s for s, l in zip(picked, langs) if l == lang]
        qs = augment.translate([s.state["question"] for s in group], lang)
        cs = augment.translate([s.state["candidates"][0]["text"] for s in group], lang)
        for s, qq, cc in zip(group, qs, cs):
            mixed = rng.random() < 0.3
            cand = dict(s.state["candidates"][0], text=(s.state["candidates"][0]["text"] if mixed else cc)[:400])
            out.append(State("rerank", {"question": qq, "candidates": [cand]}, lang,
                             f"longmemeval+{lang}", dict(s.hint, mixed=mixed)))
    # The gateways' own turns as candidates for their own users' questions.
    ft = sources.factor_turns()
    users = [t for t in ft if t.role == "user" and t.text.endswith("?") and len(t.text) < 300]
    for uq in _sample(users, int(600 * scale), rng):
        for t in _sample([t for t in ft if t.session == uq.session and t is not uq], 4, rng) + _sample(ft, 2, rng):
            out.append(State("rerank", {"question": uq.text, "candidates": [_candidate(t)]},
                             uq.lang, "factor", {"same_session": t.session == uq.session}))
    return out


# ── supersession ───────────────────────────────────────────────────────────


def build_supersession(scale: float) -> list[State]:
    rng = _rng("supersession")
    out: list[State] = []
    for lang, item in _synth("claims", int(6000 * scale), rng):
        out.append(State("supersession", {
            "source_text": str(item["source_text"])[:1500],
            "earlier_claim": {"subject": item["subject"], "predicate": item["predicate"],
                              "object": item["earlier_object"],
                              "stated_at": rng.choice(["2026-03-02", "2026-07-19", "2025-11-30", ""]),
                              "stated_by": rng.choice(["user", "user", "agent"])},
            "later_claim": {"subject": item["subject"], "predicate": item["predicate"],
                            "object": item["later_object"], "stated_by": rng.choice(["user", "user", "agent"])},
        }, lang, "synth", {"relation": str(item.get("relation"))}))
    # HaluMem's recorded updates, as claim pairs the way an extractor would
    # write them: the persona is the subject, the fact sentence the object.
    for old, new, who in _sample(sources.halumem_updates(sources.halumem()), int(2000 * scale), rng):
        out.append(State("supersession", {
            "source_text": new[:1500],
            "earlier_claim": {"subject": who, "predicate": "fact", "object": old[:300],
                              "stated_at": "", "stated_by": "user"},
            "later_claim": {"subject": who, "predicate": "fact", "object": new[:300], "stated_by": "user"},
        }, "en", "halumem", {"relation": "update?"}))
    return out


# ── Factor's heads ─────────────────────────────────────────────────────────


def _agent_turns(scale: float) -> list[AgentTurn]:
    turns = sources.factor_agent_turns()
    for lang, item in _synth("trajectory", int(2400 * scale), _rng("trajectory")):
        traj = str(item["trajectory"])
        calls = [(line.split("(", 1)[0].replace("tool call: ", ""), "", "")
                 for line in traj.splitlines() if line.startswith("tool call:")]
        turns.append(AgentTurn(task=str(item["task"]), trajectory=traj if traj.endswith("\n") else traj + "\n",
                               reply=str(item["reply"]), calls=calls, session="synth", source="synth", lang=lang))
    return turns


_OVERCLAIMS_EN = ["Done — sent.", "Saved and verified.", "Fixed; everything passes now.", "The report went out, all good.",
                  "I found it and updated the file.", "All set, the email is delivered."]


def _overclaims() -> dict[str, list[str]]:
    """Stock overclaims in every target language, through the cache."""
    out = {"en": _OVERCLAIMS_EN}
    for lang in augment.LANGS:
        out[lang] = augment.translate(_OVERCLAIMS_EN, lang)
    return out


def _clip_tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[len(text) - limit:]


def build_completion(scale: float) -> list[State]:
    rng = _rng("completion")
    turns = _agent_turns(scale)
    out: list[State] = []
    for a in turns:
        out.append(State("completion", {"task": a.task[:1200], "trajectory": _clip_tail(a.trajectory, 2000),
                                        "reply": a.reply[:2000]}, a.lang, a.source, {"pairing": "own"}))
    # Mismatched replies and stock overclaims: the trajectory stays real,
    # the reply is not the one it produced.
    for a in _sample(turns, int(len(turns) * 0.5), rng):
        other = rng.choice(turns)
        out.append(State("completion", {"task": a.task[:1200], "trajectory": _clip_tail(a.trajectory, 2000),
                                        "reply": other.reply[:2000]}, a.lang, a.source, {"pairing": "swapped"}))
    overclaims = _overclaims()
    for a in _sample(turns, int(len(turns) * 0.3), rng):
        out.append(State("completion", {"task": a.task[:1200], "trajectory": _clip_tail(a.trajectory, 2000),
                                        "reply": rng.choice(overclaims.get(a.lang, _OVERCLAIMS_EN))}, a.lang, a.source,
                         {"pairing": "overclaim"}))
    return out


def build_induction(scale: float) -> list[State]:
    rng = _rng("induction")
    skills = sources.skill_names()
    out: list[State] = []
    for a in _agent_turns(scale):
        learned = rng.sample(skills, rng.randint(0, min(6, len(skills))))
        other = rng.sample([s for s in skills if s not in learned], rng.randint(0, min(8, len(skills))))
        out.append(State("induction", {
            "task": a.task[:1200], "trajectory": _clip_tail(a.trajectory, 1800),
            "corrected": rng.random() < 0.25,
            "learned_skills": [f"{s}: {s.replace('-', ' ')}" for s in learned],
            "other_skills": [f"{s}: {s.replace('-', ' ')}" for s in other],
            "library_full": rng.random() < 0.15,
        }, a.lang, a.source, {"calls": len(a.calls)}))
    return out


def build_recovery(scale: float) -> list[State]:
    rng = _rng("recovery")
    out: list[State] = []
    for a in sources.factor_agent_turns():
        for name, args, result in a.calls:
            if not result.strip():
                continue
            failed = result.startswith("ERROR: ")
            if not failed and rng.random() > 0.15:
                continue
            out.append(State("recovery", {"task": a.task[:1200], "repeated_call": name + args,
                                          "result": result[:1200], "failed": failed}, a.lang, a.source,
                             {"failed": failed}))
    for lang, item in _synth("stall", int(2400 * scale), rng):
        result = str(item["result"])
        out.append(State("recovery", {"task": str(item["task"])[:1200], "repeated_call": str(item["repeated_call"]),
                                      "result": result[:1200], "failed": result.startswith("ERROR: ")},
                         lang, "synth", {}))
    return out


BUILDERS: dict[str, Callable[[float], list[State]]] = {
    "routing": build_routing,
    "tone": build_tone,
    "rerank": build_rerank,
    "supersession": build_supersession,
    "completion": build_completion,
    "induction": build_induction,
    "recovery": build_recovery,
}


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {json.loads(line)["id"] for line in fh if line.strip()}


def build(tasks: Iterable[str], scale: float = 1.0) -> dict[str, int]:
    """Write ``states/<task>.jsonl`` for each task, appending only states not
    already there. Returns how many each task holds now."""
    STATES.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for task in tasks:
        if task not in registry.tasks():
            raise ValueError(f"no head named {task!r} in the registry")
        path = STATES / f"{task}.jsonl"
        have = _existing_ids(path)
        states = BUILDERS[task](scale)
        fresh = {s.id: s for s in states if s.id not in have}
        with path.open("a", encoding="utf-8") as fh:
            for s in fresh.values():
                fh.write(json.dumps(s.row(), ensure_ascii=False) + "\n")
        counts[task] = len(have) + len(fresh)
        langs: dict[str, int] = defaultdict(int)
        for s in states:
            langs[s.lang] += 1
        logger.info("%s: %d states (%d new) %s", task, counts[task], len(fresh), dict(langs))
    return counts
