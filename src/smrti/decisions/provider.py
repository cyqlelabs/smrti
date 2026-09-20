"""Typed decision questions and answers, and the provider interface.

A decision provider takes a *state* — text or a JSON-compatible structure —
and a set of typed questions about it, and answers each one with a value
rather than prose: a ``Noul`` is the probability that a yes/no proposition
holds, a ``Choice`` picks one of the options the caller supplied and returns
the distribution over all of them, a ``Score`` rates the state against
ordered rubric levels. Every question in a request is evaluated against the
same state, independently of the others.

Nothing here knows which provider answers. :mod:`smrti.decisions.laya` is
the local adapter shipped; a test double that returns fixed answers is
another, and either one can be swapped for a different backend without the
callers in retrieval and extraction noticing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, Union, runtime_checkable

# Documented limits of the Choice and Score primitives. Checked on the way
# out so a malformed question is a ValueError that names the limit rather
# than a 422 that names a field the caller never wrote.
MIN_CHOICE_OPTIONS = 2
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

State = Union[str, Mapping[str, Any], Sequence[Any]]


class DecisionUnavailable(Exception):
    """The provider could not answer: transport failure, deadline, an
    unreadable or invalid reply, or a task it does not support.

    Every caller treats it the same way — the deterministic path the engine
    took before decisions existed. A decision that cannot be made is not a
    decision that was made badly.
    """


@dataclass(frozen=True)
class Noul:
    """A yes/no proposition; the answer is the probability that it holds.

    ``true``/``false`` optionally describe what each side means, which is
    where the rubric lives for a question that has one.
    """

    instructions: Any
    true: Any = None
    false: Any = None

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.true is not None or self.false is not None:
            out["criteria"] = {"true": self.true or "", "false": self.false or ""}
        return out


@dataclass(frozen=True)
class Choice:
    """Pick one option. ``criteria`` maps each option's name to its rubric."""

    instructions: Any
    criteria: Mapping[str, Any]

    def __post_init__(self) -> None:
        n = len(self.criteria)
        if not MIN_CHOICE_OPTIONS <= n <= MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"a choice takes {MIN_CHOICE_OPTIONS} to {MAX_CHOICE_OPTIONS} options, got {n}"
            )

    def payload(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class Score:
    """Rate the state against ordered levels; the answer may fall between two."""

    instructions: Any
    criteria: Sequence[Any]

    def __post_init__(self) -> None:
        n = len(self.criteria)
        if not MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"a score takes {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {n}"
            )

    def payload(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = Union[Noul, Choice, Score]


@dataclass(frozen=True)
class NoulAnswer:
    """Probability that the proposition holds. Carries no separate confidence:
    the probability is the uncertainty."""

    probability: float

    @property
    def value(self) -> float:
        return self.probability


@dataclass(frozen=True)
class ChoiceAnswer:
    """The winning option, the full distribution, and a confidence in [0, 1]
    derived from how concentrated that distribution is."""

    choice: str
    probabilities: dict[str, float]
    confidence: float

    @property
    def value(self) -> str:
        return self.choice


@dataclass(frozen=True)
class ScoreAnswer:
    """A probability-weighted level index with the legend it was scored against."""

    score: float
    probabilities: dict[str, float]
    confidence: float
    legend: dict[str, str] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.score

    @property
    def normalized(self) -> float:
        """The score as 0..1 whatever the number of levels, so two rubrics of
        different lengths can be compared."""
        top = max(len(self.legend) - 1, 1)
        return max(0.0, min(1.0, self.score / top))


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


@dataclass
class Decisions:
    """One answered request."""

    answers: dict[str, Answer]
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]

    def noul(self, key: str, default: float = 0.0) -> float:
        answer = self.answers.get(key)
        return answer.probability if isinstance(answer, NoulAnswer) else default

    def choice(self, key: str) -> ChoiceAnswer | None:
        answer = self.answers.get(key)
        return answer if isinstance(answer, ChoiceAnswer) else None

    def compact(self) -> dict[str, Any]:
        """The answers as plain JSON for the audit log."""
        out: dict[str, Any] = {}
        for key, answer in self.answers.items():
            if isinstance(answer, NoulAnswer):
                out[key] = round(answer.probability, 4)
            elif isinstance(answer, ChoiceAnswer):
                out[key] = {"choice": answer.choice, "confidence": round(answer.confidence, 4)}
            else:
                out[key] = {"score": round(answer.score, 4), "confidence": round(answer.confidence, 4)}
        return out


def questions_payload(questions: Mapping[str, Question]) -> dict[str, dict[str, Any]]:
    return {key: q.payload() for key, q in questions.items()}


def _unit(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionUnavailable(f"{what} is not a number: {value!r}") from exc
    if not 0.0 <= number <= 1.0 or number != number:
        raise DecisionUnavailable(f"{what} is outside [0, 1]: {number!r}")
    return number


def parse_answer(key: str, raw: Any, question: Question) -> Answer:
    """Read one raw answer object into its typed form, validated against
    the question it answers.

    Validation is the boundary: a choice the caller never offered, a
    probability outside the unit interval, or an answer of the wrong kind is
    refused here rather than branched on downstream. A valid label can still
    be factually wrong, but it is at least a label the code was ready for.
    """
    if not isinstance(raw, Mapping):
        raise DecisionUnavailable(f"answer {key!r} is not an object")
    kind = raw.get("type")
    try:
        if isinstance(question, Noul):
            if kind not in (None, "noul"):
                raise DecisionUnavailable(f"answer {key!r} is {kind!r}, expected noul")
            return NoulAnswer(probability=_unit(raw["noul"], f"answer {key!r}"))
        if isinstance(question, Choice):
            if kind not in (None, "choice"):
                raise DecisionUnavailable(f"answer {key!r} is {kind!r}, expected choice")
            choice = str(raw["choice"])
            if choice not in question.criteria:
                raise DecisionUnavailable(f"answer {key!r} chose {choice!r}, not an offered option")
            probabilities = {
                str(k): _unit(v, f"answer {key!r} probability {k!r}")
                for k, v in (raw.get("probabilities") or {}).items()
                if str(k) in question.criteria
            }
            return ChoiceAnswer(
                choice=choice,
                probabilities=probabilities,
                confidence=_unit(raw.get("confidence", 0.0), f"answer {key!r} confidence"),
            )
        if kind not in (None, "score"):
            raise DecisionUnavailable(f"answer {key!r} is {kind!r}, expected score")
        score = float(raw["score"])
        if score != score or score < 0.0 or score > len(question.criteria) - 1:
            raise DecisionUnavailable(f"answer {key!r} scored {score!r}, outside the rubric")
        legend = {str(k): str(v) for k, v in (raw.get("legend") or {}).items()}
        if not legend:
            legend = {str(i): str(level) for i, level in enumerate(question.criteria)}
        return ScoreAnswer(
            score=score,
            probabilities={
                str(k): _unit(v, f"answer {key!r} probability {k!r}")
                for k, v in (raw.get("probabilities") or {}).items()
            },
            confidence=_unit(raw.get("confidence", 0.0), f"answer {key!r} confidence"),
            legend=legend,
        )
    except DecisionUnavailable:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise DecisionUnavailable(f"answer {key!r} is not readable: {exc}") from exc


def parse_response(data: Any, questions: Mapping[str, Question], latency_ms: float = 0.0) -> Decisions:
    """Read a whole reply. Every question asked must be answered."""
    if not isinstance(data, Mapping):
        raise DecisionUnavailable("reply is not an object")
    raw_answers = data.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise DecisionUnavailable("reply carries no answers object")
    answers: dict[str, Answer] = {}
    for key, question in questions.items():
        if key not in raw_answers:
            raise DecisionUnavailable(f"reply has no answer for {key!r}")
        answers[key] = parse_answer(key, raw_answers[key], question)
    usage = data.get("usage") or {}
    if not isinstance(usage, Mapping):
        usage = {}

    def _count(name: str) -> int:
        try:
            return int(usage.get(name, 0) or 0)
        except (TypeError, ValueError):
            return 0

    return Decisions(
        answers=answers,
        model=str(data.get("model") or ""),
        input_tokens=_count("input_tokens"),
        output_tokens=_count("output_tokens"),
        latency_ms=latency_ms,
    )


@runtime_checkable
class DecisionProvider(Protocol):
    """What the engine needs from a backend.

    ``ask`` is synchronous because most of smrti's decision points run in an
    executor thread already (retrieval, the resolver, the supersession
    writer); ``ask_async`` serves the extraction pipeline, which is async.
    Both raise :class:`DecisionUnavailable` for anything that is not an
    answer.
    """

    name: str
    model: str

    def ask(self, state: State, questions: Mapping[str, Question], *, timeout: float) -> Decisions: ...

    async def ask_async(
        self, state: State, questions: Mapping[str, Question], *, timeout: float
    ) -> Decisions: ...


class StaticProvider:
    """A provider that answers from a function of the questions — the test
    double, and the shape a local model would take.

    ``answer(key, question, state)`` returns the raw answer object for one
    question, in the wire shape; it is parsed and validated like a network
    reply so a double cannot pass an answer the real provider could not.
    """

    name = "static"

    def __init__(self, answer, model: str = "static") -> None:
        self._answer = answer
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def ask(self, state: State, questions: Mapping[str, Question], *, timeout: float) -> Decisions:
        self.calls.append({"state": state, "questions": dict(questions), "timeout": timeout})
        raw = {key: self._answer(key, q, state) for key, q in questions.items()}
        return parse_response({"answers": raw, "model": self.model}, questions)

    async def ask_async(
        self, state: State, questions: Mapping[str, Question], *, timeout: float
    ) -> Decisions:
        return self.ask(state, questions, timeout=timeout)
