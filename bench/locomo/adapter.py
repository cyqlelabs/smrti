"""LoCoMo: very long-term conversations between two speakers.

Ten conversations, up to thirty-five sessions each, spanning months of
simulated time, with questions annotated by the reasoning they need —
single-hop, multi-hop, temporal, open-domain — and a fifth category that is
adversarial: the question looks answerable and the conversation never says.

Two things make it a different test from LongMemEval. The dialogue has two
speakers rather than a user and an assistant, so both sides are testimony and
neither is discounted. And the annotated evidence is a list of turn ids,
which lets retrieval be scored exactly, by identity, the same way.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

# "1:56 pm on 8 May, 2023"
_SESSION_DATE = "%I:%M %p on %d %B, %Y"

# Category ids as the annotators used them. Published comparisons report the
# first four; the adversarial set is scored on its own because getting it
# right means refusing to answer, which is the opposite of every other row.
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
ADVERSARIAL = 5


def parse_session_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), _SESSION_DATE)
    except ValueError:
        return None


@dataclass(frozen=True)
class Turn:
    session: str
    dia_id: str
    speaker: str
    text: str
    date: datetime | None


@dataclass(frozen=True)
class Question:
    question: str
    answer: str
    category: int
    evidence: frozenset[str]

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, str(self.category))

    @property
    def is_adversarial(self) -> bool:
        return self.category == ADVERSARIAL


@dataclass(frozen=True)
class Conversation:
    sample_id: str
    speakers: tuple[str, ...]
    turns: tuple[Turn, ...]
    questions: tuple[Question, ...] = field(default_factory=tuple)


def _session_order(key: str) -> int:
    match = re.fullmatch(r"session_(\d+)", key)
    return int(match.group(1)) if match else 0


def _turns(conversation: dict) -> tuple[Turn, ...]:
    turns: list[Turn] = []
    sessions = sorted(
        (k for k in conversation if re.fullmatch(r"session_\d+", k)), key=_session_order
    )
    for name in sessions:
        entries = conversation.get(name)
        if not isinstance(entries, list):
            continue
        date = parse_session_date(conversation.get(f"{name}_date_time"))
        for entry in entries:
            text = (entry or {}).get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            turns.append(
                Turn(
                    session=name,
                    dia_id=str(entry.get("dia_id", "")),
                    speaker=str(entry.get("speaker", "")),
                    text=text,
                    date=date,
                )
            )
    return tuple(turns)


def _questions(items: Iterable[dict]) -> tuple[Question, ...]:
    questions = []
    for item in items or []:
        category = item.get("category")
        if not isinstance(category, int):
            continue
        # The adversarial rows carry the plausible-but-unsupported answer under
        # a different key; the right response is to refuse it, so the reference
        # says so rather than repeating the trap.
        if category == ADVERSARIAL:
            answer = "The conversation does not contain this information."
        else:
            answer = str(item.get("answer", ""))
        questions.append(
            Question(
                question=str(item.get("question", "")),
                answer=answer,
                category=category,
                evidence=frozenset(str(e) for e in (item.get("evidence") or [])),
            )
        )
    return tuple(questions)


def parse_conversation(item: dict) -> Conversation:
    conversation = item.get("conversation") or {}
    return Conversation(
        sample_id=str(item.get("sample_id", "")),
        speakers=tuple(
            str(conversation.get(key, ""))
            for key in ("speaker_a", "speaker_b")
            if conversation.get(key)
        ),
        turns=_turns(conversation),
        questions=_questions(item.get("qa")),
    )


def load_conversations(path: str, limit: int | None = None) -> list[Conversation]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload if isinstance(payload, list) else payload.get("data", [])
    conversations = [parse_conversation(item) for item in items]
    return conversations[:limit] if limit else conversations


def select_questions(
    conversation: Conversation, per_conversation: int | None
) -> list[Question]:
    """A deterministic, category-balanced slice of one conversation's questions.

    Taking the front of the list would take one category at a time, and a
    benchmark that only measures single-hop recall is not measuring memory.
    """
    if not per_conversation or per_conversation >= len(conversation.questions):
        return list(conversation.questions)
    by_category: dict[int, list[Question]] = {}
    for question in conversation.questions:
        by_category.setdefault(question.category, []).append(question)
    picked: list[Question] = []
    depth = 0
    while len(picked) < per_conversation:
        taken = False
        for bucket in by_category.values():
            if depth >= len(bucket):
                continue
            picked.append(bucket[depth])
            taken = True
            if len(picked) == per_conversation:
                return picked
        if not taken:
            break
        depth += 1
    return picked


def ingest(conversation: Conversation, mem, extraction: dict | None = None) -> dict[str, Turn]:
    """Store one conversation as episodes, keeping speaker and session date.

    Both speakers are the user here: LoCoMo is two people talking, not a
    person and an assistant, so nothing is filed as agent-authored and the
    source discount never applies.
    """
    from ..harness import run_extraction

    stored: dict[str, Turn] = {}
    episodes: list[tuple[str, str, str]] = []
    for turn in conversation.turns:
        content = f"{turn.speaker}: {turn.text}"
        atom_id = mem.remember(content)
        if not atom_id:
            continue
        episodes.append((atom_id, content, "user"))
        if turn.date is not None:
            stamp = turn.date.strftime("%Y-%m-%d %H:%M:%S")
            mem.db.execute(
                "UPDATE atoms SET created_at = ?, updated_at = ? WHERE id = ?",
                (stamp, stamp, atom_id),
            )
        stored[atom_id] = turn
    run_extraction(mem, episodes, extraction)
    return stored


def evaluate_question(
    question: Question, mem, stored: dict[str, Turn], top_k: int, min_confidence: float
) -> dict:
    results = mem.recall(question.question, top_k=top_k, min_confidence=min_confidence)
    returned = [r.atom.id for r in results]
    gold = {
        atom_id for atom_id, turn in stored.items() if turn.dia_id in question.evidence
    }
    hits = [atom_id for atom_id in returned if atom_id in gold]
    return {
        "question": question.question,
        "reference": question.answer,
        "category": question.category,
        "category_name": question.category_name,
        "adversarial": question.is_adversarial,
        "gold_turns": len(gold),
        "returned": len(returned),
        "evidence_hit": bool(hits),
        "evidence_recall": len(hits) / len(gold) if gold else 0.0,
        "returned_ids": returned,
    }


def aggregate(rows: Iterable[dict]) -> dict:
    """Roll rows into the numbers a release is judged on.

    The headline excludes the adversarial category, matching how published
    LoCoMo comparisons report it — those questions have no evidence to
    retrieve, so scoring them beside the rest would measure a different thing.
    """
    rows = list(rows)
    scored = [r for r in rows if r["gold_turns"] and not r["adversarial"]]
    total = len(scored)
    summary = {
        "questions": len(rows),
        "scored_questions": total,
        "retrieval_hit_rate": (
            sum(1 for r in scored if r["evidence_hit"]) / total if total else 0.0
        ),
        "evidence_recall": (
            sum(r["evidence_recall"] for r in scored) / total if total else 0.0
        ),
    }
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category_name"], []).append(row)
    summary["by_category"] = {
        name: {
            "questions": len(group),
            "retrieval_hit_rate": (
                sum(1 for r in group if r["evidence_hit"]) / len(group)
            ),
        }
        for name, group in sorted(by_category.items())
    }
    return summary
