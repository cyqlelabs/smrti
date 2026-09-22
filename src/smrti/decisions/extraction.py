"""Decisions on the write path: routing, supersession, entity identity, tone.

Four judgements, each a bounded question the deterministic pipeline could
not ask. None of them stores or deletes anything: the routing gate decides
whether the LLM is *called*, never whether the episode is kept (it always
is — a false negative must not erase the only record of what was said);
the supersession check decides whether an older claim is *marked* replaced,
and under the line both claims stay; the entity check decides whether an
uncertain match is *accepted*, and under the line a provisional duplicate
is made rather than an identity link nothing justified; the tone check
decides whether an *estimated* valence is damped, and under the line the
estimate stands.

Every function returns ``None`` when the task is off or the provider could
not answer, and a verdict with ``applied=False`` in shadow mode. Callers
branch on ``applied`` and fall back to what they did before.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .engine import DecisionEngine
from .policies import TASK_ENTITY, TASK_ROUTING, TASK_SUPERSESSION, TASK_TONE
from .provider import Choice, Noul

logger = logging.getLogger("smrti.decisions.extraction")

# How much text the provider sees of a message, of the context the graph
# already holds, and of the episode a supersession was read from.
MESSAGE_CHARS = 2000
CONTEXT_CHARS = 3000
EPISODE_CHARS = 1500

ROUTE_SKIP = "skip"
ROUTE_LLM = "llm"
ROUTE_DEFAULT = "default"


# ── extraction routing ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Route:
    decision: str
    durable: float
    corrects: float
    novel: float
    applied: bool

    @property
    def skip(self) -> bool:
        return self.applied and self.decision == ROUTE_SKIP

    @property
    def force_llm(self) -> bool:
        return self.applied and self.decision == ROUTE_LLM


ROUTING_QUESTIONS = {
    "durable": Noul(
        "Does the message state something durable about the speaker or someone else — a fact, "
        "a preference, a goal, a constraint or rule — that would still matter in a later conversation?",
        true="A fact about a person, place, organisation or project; a preference or dislike; a goal; "
             "a rule or constraint such as 'never do X'; a plan with a date.",
        false="Thanks, acknowledgements, greetings, a one-off request or question that carries no new "
              "information about anyone, or a restatement of what the known context already holds.",
    ),
    "corrects": Noul(
        "Does the message explicitly correct, update or replace something stated earlier?",
        true="It says a previous fact is no longer true, or gives a new value for something the known "
             "context records differently (a new employer, a new city, a changed preference).",
        false="It adds or repeats information without contradicting what is known.",
    ),
    "novel": Noul(
        "Does the message add information that is absent from the known context?",
        true="It names a person, place, organisation, fact or event that the known context does not hold.",
        false="Everything it says is already in the known context, or it says nothing about anything.",
    ),
}


def decide_route(durable: float, corrects: float, novel: float, *, skip: float, force: float) -> str:
    """The route the three judgements point to, under the policy's lines.

    Skip needs all three under the line: a message that adds nothing, corrects
    nothing and holds nothing durable is chatter. Force needs a durable fact
    or a correction over the line: those are what the entity-count heuristic
    misses, since "never deploy on Fridays" names no entity.
    """
    if durable <= skip and corrects <= skip and novel <= skip:
        return ROUTE_SKIP
    if durable >= force or corrects >= force:
        return ROUTE_LLM
    return ROUTE_DEFAULT


async def route_extraction(
    engine: DecisionEngine,
    text: str,
    *,
    source: str,
    entity_context: str,
    tenant_id: str,
    space: str,
) -> Route | None:
    if not engine.enabled(TASK_ROUTING):
        return None
    state = {
        "message": text[:MESSAGE_CHARS],
        "author": "assistant" if source == "agent" else "user",
        "known_context": entity_context[:CONTEXT_CHARS] if entity_context else "",
    }
    outcome = await engine.decide_async(
        TASK_ROUTING, state, ROUTING_QUESTIONS, tenant_id=tenant_id, space=space
    )
    if outcome is None:
        return None
    d = outcome.decisions
    durable, corrects, novel = d.noul("durable"), d.noul("corrects"), d.noul("novel")
    decision = decide_route(
        durable, corrects, novel, skip=engine.policy.routing_skip, force=engine.policy.routing_force
    )
    engine.conclude(
        TASK_ROUTING, outcome, tenant_id=tenant_id, space=space, result=decision, applied=True,
        summary={"text": text[:200], "author": state["author"]},
    )
    return Route(decision=decision, durable=durable, corrects=corrects, novel=novel, applied=outcome.applied)


# ── supersession verification ─────────────────────────────────────────────


SAME_CLAIM = "same_claim"
EXPLICIT_UPDATE = "explicit_update"
COMPATIBLE = "compatible"
CONTRADICTION = "contradiction"
INSUFFICIENT_CONTEXT = "insufficient_context"

# The labels that permit the older claim to be marked replaced. A
# contradiction between two statements of the same kind about the same
# subject is a replacement too — the later statement wins, as it did before
# the check existed — but "compatible", "same claim" and "not enough context"
# keep both.
SUPERSESSION_ALLOWS = frozenset({EXPLICIT_UPDATE, CONTRADICTION})

SUPERSESSION_CRITERIA = {
    SAME_CLAIM: "The later statement says the same thing as the earlier one; nothing changed.",
    EXPLICIT_UPDATE: "The later statement replaces the earlier one: a move, a new employer, a changed "
                     "preference, a rule withdrawn. The earlier state is no longer current.",
    COMPATIBLE: "Both can be true at once: a visit is not a move, a project-specific choice is not a "
                "changed general preference, a second favourite is not a replaced one.",
    CONTRADICTION: "The two cannot both be true and the later one is stated as a fact about the current "
                   "state, not as a guess.",
    INSUFFICIENT_CONTEXT: "The text does not say enough to tell whether the earlier statement still holds.",
}


@dataclass(frozen=True)
class SupersessionVerdict:
    label: str
    confidence: float
    applied: bool

    @property
    def allow(self) -> bool:
        return self.label in SUPERSESSION_ALLOWS


def verify_supersession(
    engine: DecisionEngine,
    *,
    episode_text: str,
    subject: str,
    predicate: str,
    old_object: str,
    new_object: str,
    old_stated_at: str,
    old_author: str,
    new_author: str,
    tenant_id: str,
    space: str,
) -> SupersessionVerdict | None:
    if not engine.enabled(TASK_SUPERSESSION):
        return None
    state = {
        "source_text": episode_text[:EPISODE_CHARS],
        "earlier_claim": {"subject": subject, "predicate": predicate, "object": old_object,
                          "stated_at": old_stated_at, "stated_by": old_author},
        "later_claim": {"subject": subject, "predicate": predicate, "object": new_object,
                        "stated_by": new_author},
    }
    questions = {
        "relation": Choice(
            "How does the later claim relate to the earlier claim about the same subject, "
            "given the source text it was read from?",
            SUPERSESSION_CRITERIA,
        )
    }
    outcome = engine.decide(TASK_SUPERSESSION, state, questions, tenant_id=tenant_id, space=space)
    if outcome is None:
        return None
    answer = outcome.decisions.choice("relation")
    if answer is None:
        return None
    confident = answer.confidence >= engine.policy.supersession_min_confidence
    label = answer.choice if confident else INSUFFICIENT_CONTEXT
    verdict = SupersessionVerdict(label=label, confidence=answer.confidence, applied=outcome.applied)
    engine.conclude(
        TASK_SUPERSESSION, outcome, tenant_id=tenant_id, space=space,
        result=label, applied=True,
        summary={"subject": subject, "predicate": predicate, "old": old_object, "new": new_object,
                 "raw_label": answer.choice, "allowed": verdict.allow},
    )
    return verdict


# ── entity verification ───────────────────────────────────────────────────


ENTITY_NONE = "none"
ENTITY_AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class EntityVerdict:
    atom_id: str | None
    label: str
    confidence: float
    applied: bool

    @property
    def matched(self) -> bool:
        return self.atom_id is not None


def verify_entity(
    engine: DecisionEngine,
    *,
    name: str,
    entity_type: str,
    context: str,
    candidates: list[dict[str, Any]],
    tenant_id: str,
    space: str,
) -> EntityVerdict | None:
    """Which of *candidates* the mention names, if any.

    Each candidate is ``{"id", "label", "entity_type", "facts": [...]}``.
    The provider chooses among them by reference (``c0``…), plus ``none``
    and ``ambiguous``; the reference is mapped back to an id here and only
    an id that was offered can come out.
    """
    if not candidates or not engine.enabled(TASK_ENTITY):
        return None
    refs = {f"c{i}": c for i, c in enumerate(candidates)}
    criteria: dict[str, str] = {}
    for ref, c in refs.items():
        facts = "; ".join(str(f) for f in (c.get("facts") or [])[:4])
        criteria[ref] = f"{c.get('label')} ({c.get('entity_type') or 'concept'})" + (f": {facts}" if facts else "")
    criteria[ENTITY_NONE] = "The mention names something that is not any of the candidates."
    criteria[ENTITY_AMBIGUOUS] = "The mention could name more than one candidate and the text does not settle it."
    state = {
        "mention": name,
        "mention_type": entity_type,
        "sentence": context[:EPISODE_CHARS],
        "candidates": [
            {"ref": ref, "label": c.get("label"), "type": c.get("entity_type"), "facts": (c.get("facts") or [])[:4]}
            for ref, c in refs.items()
        ],
    }
    questions = {"identity": Choice("Which known entity, if any, does the mention refer to?", criteria)}
    outcome = engine.decide(TASK_ENTITY, state, questions, tenant_id=tenant_id, space=space)
    if outcome is None:
        return None
    answer = outcome.decisions.choice("identity")
    if answer is None:
        return None
    confident = answer.confidence >= engine.policy.entity_min_confidence
    label = answer.choice if confident else ENTITY_AMBIGUOUS
    chosen = refs.get(label)
    verdict = EntityVerdict(
        atom_id=str(chosen["id"]) if chosen else None,
        label=label if chosen is None else "match",
        confidence=answer.confidence,
        applied=outcome.applied,
    )
    engine.conclude(
        TASK_ENTITY, outcome, tenant_id=tenant_id, space=space,
        result=verdict.label, applied=True,
        summary={"mention": name, "type": entity_type, "raw_label": answer.choice,
                 "chosen": chosen.get("label") if chosen else None, "offered": len(candidates)},
    )
    return verdict


# ── tone verification ─────────────────────────────────────────────────────


TONE_THING = "thing"
TONE_SPEAKER = "speaker"

# Only an estimate at or under this is asked about. Nothing in the engine
# turns on a tone milder than −0.3 — the claim writer lowers tone only under
# −0.3, the salience shift starts at −0.5, the pruning floor at −0.7 — and
# nothing at all turns on a positive one, so a verdict on either would
# change nothing, and a write is what pays for the ask.
TONE_ASK_FROM = 0.3

# The metadata key a damped estimate is filed under on the atom: what the
# estimator read, what the model called it, and how sure it was. Decision
# confidence lives here and in the audit record, never in the truth value.
TONE_DECISION = "tone_decision"

# Two options, not three. Measured on the multilingual Laya checkpoint
# against the 29 stored sentences in Spanish, English and German of the
# `bench/decisions` tone set: a three-way choice with an "incidental" option
# never answered above 0.31 confidence — the model spread its mass over the
# long rubrics and no line was ever crossed — and called an apology a
# verdict. This pair puts every failure, bug, rule and preference on
# ``thing`` (0.65–0.99 probability, none ever on ``speaker``) and apologies
# on ``speaker`` at 0.52–0.61 confidence; "Thanks, that worked!" sits just
# under the 0.4 line at 0.39. What it does not catch is the curt imperative
# ("cerra el navegador" is about a thing, if not a verdict on one), which
# sits near 50/50 under every wording tried; that is the estimate the line
# keeps, on purpose. The state carries no hint of the estimate's sign: with
# one, "Odio cuando el WAF bloquea el scraper" — a preference — went to
# ``speaker`` at 0.02, and without one nothing did.
TONE_CRITERIA = {
    TONE_THING: "It reports a failure, error, harm, danger or rule about the thing described.",
    TONE_SPEAKER: "It expresses the speaker's frustration, thanks, apology or impatience in the moment.",
}
TONE_QUESTION = Choice(
    "Is the negative tone of the text about the thing it describes, or the speaker's own mood?",
    TONE_CRITERIA,
)


@dataclass(frozen=True)
class ToneVerdict:
    label: str
    confidence: float
    applied: bool
    estimate: float
    valence: float

    @property
    def damped(self) -> bool:
        """Whether the caller stores something milder than the estimate."""
        return self.applied and self.valence != self.estimate


def judge_tone(
    engine: DecisionEngine,
    *,
    text: str,
    estimate: float,
    source: str,
    kind: str,
    tenant_id: str,
    space: str,
) -> ToneVerdict | None:
    """Whose is the tone the sentiment estimator read off *text*?

    The estimator scores the mood of the words, and stored conversation is
    full of mood that is nobody's verdict on anything: in one Factor graph a
    curt "cerra el navegador" scored −0.55, an apology −0.64, and a report
    that a request timed out −1.0, which is under the line that hands a
    memory a pruning floor for life. An apology is the speaker's moment, not
    a property of what the memory records, and the estimator cannot tell the
    two apart because both are written in the same charged words.

    This asks. A confident ``speaker`` verdict shrinks the estimate by the
    policy's ``tone_damping``; ``thing``, an unsure answer, or no answer at
    all keeps it, since damping a real warning is the destructive direction.
    The result is still an estimate — nothing here sets ``VALENCE_STATED`` —
    and only ever milder than the one that came in: a stated valence is never
    offered to this function, so a decision can take a pruning floor away
    from an apology but cannot hand one to anything.
    """
    if estimate > -TONE_ASK_FROM or not engine.enabled(TASK_TONE):
        return None
    state = {
        "text": text[:MESSAGE_CHARS],
        "author": "assistant" if source == "agent" else "user",
        "kind": kind,
    }
    outcome = engine.decide(TASK_TONE, state, {"tone": TONE_QUESTION}, tenant_id=tenant_id, space=space)
    if outcome is None:
        return None
    answer = outcome.decisions.choice("tone")
    if answer is None:
        return None
    confident = answer.confidence >= engine.policy.tone_min_confidence
    label = answer.choice if confident else TONE_THING
    valence = estimate
    if label == TONE_SPEAKER and outcome.applied:
        valence = round(estimate * engine.policy.tone_damping, 3)
    verdict = ToneVerdict(
        label=label, confidence=answer.confidence, applied=outcome.applied,
        estimate=estimate, valence=valence,
    )
    engine.conclude(
        TASK_TONE, outcome, tenant_id=tenant_id, space=space, result=label, applied=True,
        summary={"text": text[:200], "author": state["author"], "kind": kind,
                 "estimate": round(estimate, 3), "valence": valence, "raw_label": answer.choice},
    )
    return verdict
