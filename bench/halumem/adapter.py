"""HaluMem: what a memory system says when it does not know.

Twenty personas, sixty-five sessions each, and questions sorted by what they
probe — basic recall, multi-hop inference, conflicting statements, facts that
were updated later, and *memory boundary*: questions whose answer was never
given, where the only correct response is to say so.

That last category is why this benchmark scores three ways instead of two. A
system that invents an answer and one that admits it has no record are both
"wrong" to a binary judge, and they are not remotely the same failure: the
first is dangerous, the second is merely incomplete. Correct, hallucination
and omission are reported separately, and hallucination is the number that
matters.

Only the question-answering task is implemented here. HaluMem also scores
memory extraction and memory updating against reference memory points, which
needs the extraction pipeline running over sixty thousand turns — a separate
job with a separate bill.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

# "Sep 04, 2025, 18:42:18"
_TIMESTAMP = "%b %d, %Y, %H:%M:%S"


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), _TIMESTAMP)
    except ValueError:
        return None


@dataclass(frozen=True)
class Turn:
    session: int
    index: int
    role: str
    content: str
    at: datetime | None


@dataclass(frozen=True)
class Question:
    question: str
    answer: str
    question_type: str
    difficulty: str
    asked_on: str
    has_evidence: bool


@dataclass(frozen=True)
class User:
    uuid: str
    turns: tuple[Turn, ...]
    questions: tuple[Question, ...] = field(default_factory=tuple)


def _parse_user(item: dict) -> User:
    turns: list[Turn] = []
    questions: list[Question] = []
    for position, session in enumerate(item.get("sessions") or []):
        started = session.get("start_time")
        for index, entry in enumerate(session.get("dialogue") or []):
            content = (entry or {}).get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            turns.append(
                Turn(
                    session=position,
                    index=index,
                    role=str(entry.get("role") or "user"),
                    content=content,
                    at=parse_timestamp(entry.get("timestamp") or started),
                )
            )
        for entry in session.get("questions") or []:
            question = (entry or {}).get("question")
            if not isinstance(question, str) or not question.strip():
                continue
            questions.append(
                Question(
                    question=question,
                    answer=str(entry.get("answer", "")),
                    question_type=str(entry.get("question_type", "")),
                    difficulty=str(entry.get("difficulty", "")),
                    asked_on=str(session.get("end_time") or started or ""),
                    has_evidence=bool(entry.get("evidence")),
                )
            )
    return User(uuid=str(item.get("uuid", "")), turns=tuple(turns), questions=tuple(questions))


def load_users(path: str, limit: int | None = None) -> list[User]:
    """Read the JSON-lines dataset, taking the first *limit* personas.

    No stratification here: every persona carries the full spread of question
    types, so the front of the file is already a balanced sample — unlike
    LongMemEval, where the file is grouped by ability.
    """
    users: list[User] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            users.append(_parse_user(json.loads(line)))
            if limit and len(users) >= limit:
                break
    return users


def select_questions(user: User, per_user: int | None) -> list[Question]:
    """A deterministic slice balanced across question types."""
    if not per_user or per_user >= len(user.questions):
        return list(user.questions)
    by_type: dict[str, list[Question]] = {}
    for question in user.questions:
        by_type.setdefault(question.question_type, []).append(question)
    picked: list[Question] = []
    depth = 0
    while len(picked) < per_user:
        taken = False
        for bucket in by_type.values():
            if depth >= len(bucket):
                continue
            picked.append(bucket[depth])
            taken = True
            if len(picked) == per_user:
                return picked
        if not taken:
            break
        depth += 1
    return picked


def ingest(user: User, mem) -> int:
    """Store one persona's dialogue as episodes, keeping the turn timestamps.

    Assistant turns are filed as agent-authored, which is what they are: the
    engine trusts its own past output less than the user's, and a benchmark
    that hid that would measure a configuration nobody runs.
    """
    stored = 0
    for turn in user.turns:
        atom_id = mem.remember(
            turn.content,
            metadata={"source": "agent"} if turn.role == "assistant" else None,
        )
        if not atom_id:
            continue
        if turn.at is not None:
            stamp = turn.at.strftime("%Y-%m-%d %H:%M:%S")
            mem.db.execute(
                "UPDATE atoms SET created_at = ?, updated_at = ? WHERE id = ?",
                (stamp, stamp, atom_id),
            )
        stored += 1
    return stored


def recall_for(question: Question, mem, top_k: int, min_confidence: float) -> list[str]:
    """The memories the answering model will see, newest date first in text."""
    results = mem.recall(question.question, top_k=top_k, min_confidence=min_confidence)
    return [
        f"[{(r.atom.created_at or '')[:10]}] {r.atom.content or r.atom.label}"
        for r in results
    ]


def aggregate(rows: Iterable[dict]) -> dict:
    """Correct, hallucination and omission rates, overall and per question type."""
    rows = [r for r in rows if r.get("verdict")]
    total = len(rows)

    def _rates(group: list[dict]) -> dict:
        n = len(group) or 1
        return {
            "questions": len(group),
            "correct_rate": sum(1 for r in group if r["verdict"] == "correct") / n,
            "hallucination_rate": sum(1 for r in group if r["verdict"] == "hallucination") / n,
            "omission_rate": sum(1 for r in group if r["verdict"] == "omission") / n,
        }

    summary = _rates(rows) if total else {
        "questions": 0, "correct_rate": 0.0,
        "hallucination_rate": 0.0, "omission_rate": 0.0,
    }
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["question_type"] or "unknown", []).append(row)
    summary["by_question_type"] = {
        name: _rates(group) for name, group in sorted(by_type.items())
    }
    # The boundary questions are the benchmark's point: the answer was never
    # given, so anything asserted is invented.
    boundary = [r for r in rows if not r["has_evidence"]]
    summary["boundary_questions"] = len(boundary)
    summary["boundary_hallucination_rate"] = (
        sum(1 for r in boundary if r["verdict"] == "hallucination") / len(boundary)
        if boundary
        else 0.0
    )
    return summary
