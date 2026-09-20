"""Evidence judgements over the retrieval shortlist.

Local retrieval decides who is a candidate: the vector and lexical
searches, the graph expansion and the personality-weighted salience score
stay the first stage, and nothing enters the shortlist that they did not
rank. The provider is then asked, for each of the top candidates, four
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

Filtering is separate from reranking and off by default. A candidate under
``rerank_min_evidence`` is dropped only when that line is set, and a stated
critical warning is never dropped: the one memory the engine promises to
deliver is not the reranker's to lose.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from smrti.core.models import AtomType, RecallResult
from smrti.core.provenance import SOURCE_AGENT
from smrti.retrieval.classify import is_critical_warning

from .engine import DecisionEngine
from .policies import TASK_RERANK
from .provider import Noul

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
    refs = [f"c{i}" for i in range(len(shortlist))]
    state = {
        "tenant": tenant_id,
        "space": space,
        "question": query,
        "candidates": [_candidate_state(ref, r) for ref, r in zip(refs, shortlist)],
        # The candidate versions are part of the state so the cache key
        # changes when one of them does.
        "versions": [r.atom.updated_at or "" for r in shortlist],
    }
    outcome = engine.decide(
        TASK_RERANK, state, evidence_questions(refs), tenant_id=tenant_id, space=space
    )
    if outcome is None:
        return results

    decisions = outcome.decisions
    evidence: dict[str, float] = {}
    for ref, r in zip(refs, shortlist):
        score = fold_evidence(
            decisions.noul(f"{ref}_direct"),
            decisions.noul(f"{ref}_link"),
            decisions.noul(f"{ref}_historical"),
            decisions.noul(f"{ref}_contradicts"),
        )
        evidence[r.atom.id] = score
        r.evidence = score

    reordered = rerank(results, evidence, policy.rerank_weight)
    dropped: list[str] = []
    if policy.rerank_min_evidence > 0.0:
        kept = []
        for r in reordered:
            judged = evidence.get(r.atom.id)
            if (
                judged is not None
                and judged < policy.rerank_min_evidence
                and not is_critical_warning(r.atom)
            ):
                dropped.append(r.atom.id)
                continue
            kept.append(r)
        reordered = kept

    before = [r.atom.id for r in results[:5]]
    after = [r.atom.id for r in reordered[:5]]
    engine.conclude(
        TASK_RERANK, outcome, tenant_id=tenant_id, space=space,
        result="reordered" if before != after or dropped else "unchanged",
        applied=True,
        summary={
            "query": query[:200],
            "judged": len(shortlist),
            "dropped": len(dropped),
            "top_before": before,
            "top_after": after,
            "sufficiency": round(max(evidence.values(), default=0.0), 3),
        },
    )
    return reordered if outcome.applied else results


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
