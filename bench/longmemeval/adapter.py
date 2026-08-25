"""LongMemEval-S as a retrieval regression harness for smrti.

Every retrieval change so far has been validated against one person's
recordings. That evidence is real and it is narrow: it cannot say whether a
ranking change helps or hurts on conversation shapes nobody here happens to
have had. This module ingests the benchmark's conversation histories as
episodes and answers its questions through ``recall``.

Retrieval hit rate is measured on its own, never folded into answer accuracy.
A model that answers well from a bad candidate set hides exactly the
regression this harness exists to catch.

Each question gets its own memory space, so one question's haystack can never
be another's distractor — the benchmark measures retrieval over the history
it hands you, not over every history at once.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

# "2023/05/20 (Sat) 02:36" — the weekday is decoration around a fixed format.
_WEEKDAY = re.compile(r"\s*\([^)]*\)")
_DATE_FORMATS = ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")


def parse_date(value: str | None) -> datetime | None:
    """The benchmark's session timestamp, or None if it is not one."""
    if not value:
        return None
    cleaned = _WEEKDAY.sub("", value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class Turn:
    session_id: str
    index: int
    role: str
    content: str
    has_answer: bool
    date: datetime | None


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    answer: str
    question_type: str
    question_date: str
    turns: tuple[Turn, ...]
    answer_session_ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def gold_turns(self) -> tuple[Turn, ...]:
        """The turns the benchmark marks as carrying the evidence."""
        return tuple(t for t in self.turns if t.has_answer)


def _turns(item: dict) -> tuple[Turn, ...]:
    sessions = item.get("haystack_sessions") or []
    session_ids = item.get("haystack_session_ids") or []
    dates = item.get("haystack_dates") or []
    turns: list[Turn] = []
    for position, session in enumerate(sessions):
        session_id = (
            session_ids[position] if position < len(session_ids) else f"session_{position}"
        )
        date = parse_date(dates[position] if position < len(dates) else None)
        for index, turn in enumerate(session or []):
            content = (turn or {}).get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            turns.append(
                Turn(
                    session_id=session_id,
                    index=index,
                    role=(turn.get("role") or "user"),
                    content=content,
                    has_answer=bool(turn.get("has_answer")),
                    date=date,
                )
            )
    return tuple(turns)


def parse_question(item: dict) -> Question:
    return Question(
        question_id=str(item.get("question_id", "")),
        question=str(item.get("question", "")),
        answer=str(item.get("answer", "")),
        question_type=str(item.get("question_type", "")),
        question_date=str(item.get("question_date", "")),
        turns=_turns(item),
        answer_session_ids=frozenset(item.get("answer_session_ids") or []),
    )


def load_questions(path: str, limit: int | None = None) -> list[Question]:
    """Read the benchmark file, taking a *limit*-sized slice across the abilities.

    The slice is deterministic — a subset that changes between runs measures
    the subset, not the engine — but it is not the front of the file. The file
    is grouped by question type, so the first forty questions are forty
    single-session-user questions, and a gate built on them would say nothing
    about multi-session reasoning, temporal reasoning, or knowledge updates.
    Taking one question from each type in turn covers every ability the
    benchmark separates, in the same order every time.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload if isinstance(payload, list) else payload.get("questions", [])
    questions = [parse_question(item) for item in items]
    if not limit or limit >= len(questions):
        return questions

    by_type: dict[str, list[Question]] = {}
    for question in questions:
        by_type.setdefault(question.question_type, []).append(question)

    picked: list[Question] = []
    depth = 0
    while len(picked) < limit:
        taken = False
        for bucket in by_type.values():
            if depth >= len(bucket):
                continue
            picked.append(bucket[depth])
            taken = True
            if len(picked) == limit:
                return picked
        if not taken:
            break
        depth += 1
    return picked


def ingest(question: Question, mem) -> dict[str, Turn]:
    """Store one question's history as episodes, keeping the session dates.

    Returns the atom id of every stored turn, so a recall result can be told
    apart from the gold evidence by identity rather than by string matching.
    """
    stored: dict[str, Turn] = {}
    for turn in question.turns:
        # The benchmark's assistant turns are the model's own words, and the
        # engine already trusts those less than the user's — the same
        # asymmetry it applies to a live conversation.
        atom_id = mem.remember(
            turn.content,
            metadata={"source": "agent"} if turn.role == "assistant" else None,
        )
        if not atom_id:
            continue
        if turn.date is not None:
            stamp = turn.date.strftime("%Y-%m-%d %H:%M:%S")
            mem.db.execute(
                "UPDATE atoms SET created_at = ?, updated_at = ? WHERE id = ?",
                (stamp, stamp, atom_id),
            )
        stored[atom_id] = turn
    return stored


def evaluate_question(
    question: Question, mem, stored: dict[str, Turn], top_k: int, min_confidence: float
) -> dict:
    """Recall for one question and score what came back."""
    results = mem.recall(
        question.question, top_k=top_k, min_confidence=min_confidence
    )
    returned = [r.atom.id for r in results]
    gold_ids = {
        atom_id for atom_id, turn in stored.items() if turn.has_answer
    }
    session_ids = {
        atom_id
        for atom_id, turn in stored.items()
        if turn.session_id in question.answer_session_ids
    }
    hits = [atom_id for atom_id in returned if atom_id in gold_ids]
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "turns": len(stored),
        "gold_turns": len(gold_ids),
        "returned": len(returned),
        "evidence_hit": bool(hits),
        "evidence_recall": len(hits) / len(gold_ids) if gold_ids else 0.0,
        "session_hit": any(atom_id in session_ids for atom_id in returned),
        "returned_ids": returned,
    }


def aggregate(rows: Iterable[dict]) -> dict:
    """Roll per-question rows into the numbers a release is judged on."""
    rows = list(rows)
    scored = [r for r in rows if r["gold_turns"]]
    total = len(scored)
    return {
        "questions": len(rows),
        "scored_questions": total,
        # The headline: did the evidence the benchmark marked reach the
        # answering model at all.
        "retrieval_hit_rate": (
            sum(1 for r in scored if r["evidence_hit"]) / total if total else 0.0
        ),
        # How much of it reached the model, for the questions needing several
        # pieces — a hit rate alone cannot tell one of four from four of four.
        "evidence_recall": (
            sum(r["evidence_recall"] for r in scored) / total if total else 0.0
        ),
        "session_hit_rate": (
            sum(1 for r in scored if r["session_hit"]) / total if total else 0.0
        ),
    }
