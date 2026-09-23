"""Evidence judgements over the retrieval shortlist.

Local retrieval decides who is a candidate: the vector and lexical
searches, the graph expansion and the personality-weighted salience score
stay the first stage, and nothing enters the shortlist that they did not
rank. The provider is then asked, about one candidate at a time, four
yes/no questions about the candidate *as evidence for this question* —
things similarity cannot see, because "is about the same thing" and
"answers it" are different properties:

- does it state something that directly answers the question;
- does it hold an intermediate fact a multi-hop answer needs;
- does it describe a superseded or historical state rather than the
  current one;
- does it contradict a premise of the question.

The four are folded locally into one evidence score, the score is blended
with the local salience rank (``rerank_weight`` says how much of the final
order it decides), and the result goes through the same diversity cap and
cut to top_k as before. Only what survives that cut is boosted, which is
the attention boundary the reranker needs: a candidate the provider judged
and the cut discarded was never read.

One candidate per call, and the clock decides how many. The shortlist used
to travel whole, every question carrying all twenty candidates: measured
against the real checkpoint that is 1024 tokens a call, the model's whole
window, 1.7 s each and 80 calls a recall, so every rerank ran out its
deadline and the cooldown after it silenced every other task for a minute.
Worse than slow, it did not rank — past the window the candidate asked
about was not in the state at all, and the answers for all twenty sat
between 0.70 and 0.94. Alone with the question a candidate is ~200 tokens
and 130 ms, and the same model tells 0.28 from 0.82. Asking the four
questions in one call is out for the reason ``laya.MAX_QUESTIONS_PER_CALL``
gives: measured, it is slower than four calls and moves the answers by up
to 0.6. So the shortlist is walked in salience order, one candidate and
four calls at a time, until the next candidate would not fit inside the
task's deadline; the candidates judged are applied and the rest keep the
salience share :func:`rerank` always gave a candidate outside the judged
set. ``rerank_shortlist`` is the ceiling, the deadline is the budget, and
stopping on it is a plan rather than a failure: no cooldown opens. Every
candidate is presented as ``c0`` and carries no version stamp, so the
engine caches it by what it says rather than by where it ranked or when
it was last boosted, and a recall that meets it again pays nothing for it.

Filtering is separate from reranking and off by default. A candidate under
``rerank_min_evidence`` is dropped only when that line is set, and a stated
critical warning is never dropped: the one memory the engine promises to
deliver is not the reranker's to lose.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from smrti.core.models import AtomType, RecallResult
from smrti.core.provenance import SOURCE_AGENT
from smrti.retrieval.classify import is_critical_warning

from .engine import DecisionEngine, DecisionOutcome
from .policies import TASK_RERANK
from .provider import Decisions, Noul

logger = logging.getLogger("smrti.decisions.retrieval")

# How much of a memory the provider sees. Enough for a fact; a stored
# transcript is cut, since the question is whether it holds the answer and
# not whether it holds everything.
CANDIDATE_TEXT_CHARS = 400

# Weights of the four judgements in the folded evidence score. A direct
# answer is worth the whole score; a contradiction of the premise nearly
# so, because the corrective memory is the useful one; a linking fact is
# partial evidence. A historical state discounts rather than cancels: for
# a question about history it *is* the answer, and the provider was not
# asked which kind of question this is.
_W_CONTRADICTS = 0.9
_W_LINK = 0.6
_HISTORICAL_DISCOUNT = 0.4

# The reference every candidate is presented under: the model sees one
# candidate per call, and a name that does not change with rank is what
# lets the engine's cache recognise the candidate the next time.
_REF = "c0"

# How much longer than the slowest candidate so far the next one is allowed
# to need before the walk stops. A candidate that would outrun the deadline
# is a provider timeout, and a timeout opens the cooldown for every task;
# the margin keeps the ordinary variance between calls from reading as one.
_COST_MARGIN = 1.25

# Why the walk ended, in the audit summary.
STOP_COMPLETE = "complete"
STOP_BUDGET = "budget"
STOP_UNAVAILABLE = "unavailable"

Judge = Callable[[str, list[RecallResult]], list[RecallResult]]


def _candidate_state(ref: str, r: RecallResult) -> dict[str, Any]:
    atom = r.atom
    text = (atom.content or atom.label or "")[:CANDIDATE_TEXT_CHARS]
    state: dict[str, Any] = {
        "ref": ref,
        "text": text,
        "type": atom.type.value,
        "written_at": atom.created_at or "",
        "author": SOURCE_AGENT if atom.metadata.get("source") == SOURCE_AGENT else "user",
    }
    if atom.entity_type is not None:
        state["entity_type"] = atom.entity_type.value
    if atom.type in (AtomType.BELIEF, AtomType.GOAL):
        # A belief a later claim replaced was cut under 0.3; that is the one
        # supersession signal the atom itself carries.
        state["status"] = "superseded" if atom.truth.probability < 0.3 else "current"
    temporal = atom.metadata.get("temporal")
    if isinstance(temporal, list) and temporal:
        state["dates"] = [
            f"{t.get('text')} = {t.get('resolved')}" for t in temporal
            if isinstance(t, dict) and t.get("text") and t.get("resolved")
        ][:6]
    return state


def evidence_questions(refs: list[str]) -> dict[str, Noul]:
    questions: dict[str, Noul] = {}
    for ref in refs:
        questions[f"{ref}_direct"] = Noul(
            f"Does candidate {ref} state something that directly answers the question?",
            true="The candidate contains a fact, event or statement that answers the question on its own.",
            false="The candidate is about a related topic but does not answer the question.",
        )
        questions[f"{ref}_link"] = Noul(
            f"Does candidate {ref} provide an intermediate fact that is needed to reach the answer?",
            true="It links the question to the answer — an identity, a date, a place, a relationship — "
                 "without stating the answer itself.",
            false="It neither answers the question nor connects to the answer.",
        )
        questions[f"{ref}_historical"] = Noul(
            f"Does candidate {ref} describe a state that has since been replaced, or a past situation, "
            "rather than the current one?",
            true="It reports something that was true then and has been updated since, or is dated before a later change.",
            false="It reports the current state, or the question is about what happened rather than what is.",
        )
        questions[f"{ref}_contradicts"] = Noul(
            f"Does candidate {ref} contradict something the question takes for granted?",
            true="The question assumes a fact that the candidate shows to be false or changed.",
            false="The candidate is consistent with the question's assumptions, or says nothing about them.",
        )
    return questions


def fold_evidence(direct: float, link: float, historical: float, contradicts: float) -> float:
    useful = max(direct, _W_CONTRADICTS * contradicts, _W_LINK * link)
    return max(0.0, min(1.0, useful * (1.0 - _HISTORICAL_DISCOUNT * historical)))


def rerank(results: list[RecallResult], evidence: dict[str, float], weight: float) -> list[RecallResult]:
    """The candidates reordered by a blend of local salience and evidence.

    Salience is normalised to the best candidate's, so the blend is between
    two unit-scale quantities. A candidate outside the judged shortlist
    keeps its salience share alone — it ranked below every judged one
    locally, and the blend leaves it there unless the judged ones fall.
    """
    if not results:
        return []
    weight = max(0.0, min(1.0, weight))
    top = max(r.salience for r in results) or 1.0

    def score(r: RecallResult) -> float:
        local = (1.0 - weight) * (r.salience / top)
        judged = evidence.get(r.atom.id)
        return local + (weight * judged if judged is not None else 0.0)

    return sorted(results, key=lambda r: (-score(r), -r.salience, r.atom.id))


def _walk(
    engine: DecisionEngine,
    query: str,
    shortlist: list[RecallResult],
    *,
    tenant_id: str,
    space: str,
) -> tuple[list[DecisionOutcome], str]:
    """Judge the shortlist in order, one candidate per call, for as long as
    the task's deadline holds: the outcomes for the candidates judged, in
    order, and why the walk ended."""
    questions = evidence_questions([_REF])
    deadline = time.monotonic() + engine.policy.timeout
    slowest = 0.0
    outcomes: list[DecisionOutcome] = []
    for r in shortlist:
        remaining = deadline - time.monotonic()
        if outcomes and remaining < slowest * _COST_MARGIN:
            return outcomes, STOP_BUDGET
        # No version stamp: every recall boosts what it returns and rewrites
        # ``updated_at`` as it does, so a key carrying it never hit twice.
        # The state holds every field the answer is judged on, so it is its
        # own version.
        state = {"question": query, "candidates": [_candidate_state(_REF, r)]}
        started = time.monotonic()
        outcome = engine.decide(
            TASK_RERANK, state, questions, tenant_id=tenant_id, space=space, timeout=max(remaining, 0.001)
        )
        if outcome is None:
            return outcomes, STOP_UNAVAILABLE
        if not outcome.cached:
            slowest = max(slowest, time.monotonic() - started)
        outcomes.append(outcome)
    return outcomes, STOP_COMPLETE


def judge_evidence(
    engine: DecisionEngine,
    query: str,
    results: list[RecallResult],
    *,
    tenant_id: str,
    space: str,
) -> list[RecallResult]:
    """Judge the shortlist and return the candidates in their new order.

    Returns *results* unchanged when the task is off, the provider could not
    answer, or the task runs in shadow — in shadow the judgement is filed and
    the ``evidence`` field on each result is still annotated, so a reader
    can compare it against the local order without changing anything.
    """
    if not results or not engine.enabled(TASK_RERANK):
        return results
    policy = engine.policy
    shortlist = results[: max(1, policy.rerank_shortlist)]
    outcomes, stopped = _walk(engine, query, shortlist, tenant_id=tenant_id, space=space)
    if not outcomes:
        return results

    judged = shortlist[: len(outcomes)]
    evidence: dict[str, float] = {}
    answers: dict[str, Any] = {}
    for i, (r, outcome) in enumerate(zip(judged, outcomes)):
        d = outcome.decisions
        score = fold_evidence(
            d.noul(f"{_REF}_direct"),
            d.noul(f"{_REF}_link"),
            d.noul(f"{_REF}_historical"),
            d.noul(f"{_REF}_contradicts"),
        )
        evidence[r.atom.id] = score
        r.evidence = score
        for key, answer in d.answers.items():
            answers[f"c{i}" + key[len(_REF):]] = answer
    combined = DecisionOutcome(
        decisions=Decisions(
            answers=answers,
            model=next((o.decisions.model for o in outcomes if o.decisions.model), ""),
            input_tokens=sum(o.decisions.input_tokens for o in outcomes),
            output_tokens=sum(o.decisions.output_tokens for o in outcomes),
            latency_ms=sum(o.decisions.latency_ms for o in outcomes),
        ),
        mode=outcomes[0].mode,
        cached=all(o.cached for o in outcomes),
    )

    reordered = rerank(results, evidence, policy.rerank_weight)
    dropped: list[str] = []
    if policy.rerank_min_evidence > 0.0:
        kept = []
        for r in reordered:
            judged_score = evidence.get(r.atom.id)
            if (
                judged_score is not None
                and judged_score < policy.rerank_min_evidence
                and not is_critical_warning(r.atom)
            ):
                dropped.append(r.atom.id)
                continue
            kept.append(r)
        reordered = kept

    before = [r.atom.id for r in results[:5]]
    after = [r.atom.id for r in reordered[:5]]
    engine.conclude(
        TASK_RERANK, combined, tenant_id=tenant_id, space=space,
        result="reordered" if before != after or dropped else "unchanged",
        applied=True,
        summary={
            "query": query[:200],
            "judged": len(judged),
            "shortlist": len(shortlist),
            "stopped": stopped,
            "dropped": len(dropped),
            "top_before": before,
            "top_after": after,
            "sufficiency": round(max(evidence.values(), default=0.0), 3),
        },
    )
    return reordered if combined.applied else results


def make_judge(engine: DecisionEngine, tenant_id: str, space: str) -> Judge | None:
    """The judge retrieval runs the ranked candidates through, or None when
    the task is off — so a disabled task costs retrieval nothing."""
    if not engine.enabled(TASK_RERANK):
        return None

    def judge(query: str, results: list[RecallResult]) -> list[RecallResult]:
        try:
            return judge_evidence(engine, query, results, tenant_id=tenant_id, space=space)
        except Exception:  # a judge that raises must never lose the local ranking
            logger.warning("evidence judge failed; keeping the local ranking", exc_info=True)
            return results

    return judge
