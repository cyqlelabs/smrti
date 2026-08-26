"""Diversity cap applied to salience-ranked recall results.

Salience ranks each atom on its own merits, which says nothing about how a
*set* of them reads. Measured failure: asking what the engine knew about the
user's family returned five episodes stored minutes apart, all of them
restatements of the question — one conversational moment, served five times,
carrying no fact at all. Every one of those atoms was correctly ranked; the
set was still worthless.

So the cap runs after ranking and before the cut to top_k: an episode that
repeats one already selected from the same moment yields its slot, and slots
are held back for beliefs so episodes cannot crowd out the standing facts
entirely. Nothing is dropped — a skipped atom comes back to fill any slot
left over, so a graph holding one single topic still answers with a full
top_k.

Both halves of "same moment" are load-bearing. Time alone was tried first and
cost the LongMemEval-S harness thirty points of retrieval hit rate: a whole
stored session shares one timestamp, so capping per time window capped the
session — including the two turns that actually held the answer. Repetition
is what makes a response worthless, and two turns of one conversation saying
different things are not repetition.
"""
from __future__ import annotations

from datetime import datetime

from smrti.core.models import AtomType, RecallResult
from smrti.retrieval.text import containment, word_set

# Atoms written within the same window are one conversational moment. Ten
# minutes is long enough to cover a turn and its immediate follow-ups, short
# enough that two genuinely separate sessions rarely land in one bucket.
_TIME_CLUSTER_SECONDS = 600

# How much of the shorter text two episodes must share to count as the same
# thing said again. The same threshold the echo test uses against the query.
_DUPLICATE_OVERLAP = 0.7

# How many times one moment may say the same thing, per six slots of answer.
# The floor of two leaves room for a statement and its correction in a small
# answer, where one duplicate wastes a fifth of the response; a fifty-slot
# answer can afford more repetition and cannot afford to evict evidence, so
# the allowance grows with what a duplicate actually costs the reader.
_MAX_REPEATS = 2
_SLOTS_PER_REPEAT = 6


def _repeat_allowance(top_k: int) -> int:
    return max(_MAX_REPEATS, top_k // _SLOTS_PER_REPEAT)

# Slots held for beliefs when the candidate pool has them. Beliefs are the
# standing facts — what the engine actually knows — and they are the first
# thing a wall of episodes buries.
_BELIEF_RESERVE = 2


def _cluster_key(result: RecallResult) -> str:
    """The time bucket an atom was written in.

    An unparseable or missing timestamp yields the atom's own id, which makes
    it a bucket of one: an unknown write time is not evidence that two atoms
    came from the same moment, and guessing they did would cap a memory for a
    reason that was never observed.
    """
    stamp = result.atom.created_at
    if not stamp:
        return result.atom.id
    try:
        written = datetime.fromisoformat(stamp)
    except ValueError:
        return result.atom.id
    return str(int(written.timestamp()) // _TIME_CLUSTER_SECONDS)


def diversify(
    results: list[RecallResult], top_k: int, min_confidence: float = 0.0
) -> list[RecallResult]:
    """Return at most *top_k* results, varied across moments and types.

    *results* must already be sorted by descending salience; the returned list
    keeps that order. The cap only ever changes which candidates make the cut,
    never how the survivors rank against each other.
    """
    if top_k <= 0:
        return []
    if len(results) <= top_k:
        return list(results)

    beliefs = [
        r
        for r in results
        if r.atom.type == AtomType.BELIEF
        and r.atom.truth.confidence >= min_confidence
    ]
    # Reserving half the response for beliefs would be its own distortion, so
    # the floor is bounded by the answer's size as well as by what exists.
    reserve = min(_BELIEF_RESERVE, len(beliefs), top_k // 2)

    selected: list[RecallResult] = list(beliefs[:reserve])
    selected_ids = {r.atom.id for r in selected}
    deferred: list[RecallResult] = []
    # Word sets of the episodes already selected, per time bucket. Only
    # episodes from the same moment are ever compared, so this stays small
    # however wide the candidate pool is.
    kept_by_cluster: dict[str, list[set[str]]] = {}

    # The reserve needs no second guard here. Those slots are already filled
    # above, so an episode can only be considered while fewer than top_k
    # atoms are held, and the walk stops at top_k either way.
    for result in results:
        if len(selected) >= top_k:
            break
        if result.atom.id in selected_ids:
            continue
        if result.atom.type == AtomType.EPISODE:
            cluster = _cluster_key(result)
            words = word_set(result.atom.content or result.atom.label)
            kept = kept_by_cluster.setdefault(cluster, [])
            repeats = sum(
                1 for other in kept if containment(words, other) >= _DUPLICATE_OVERLAP
            )
            if repeats >= _repeat_allowance(top_k):
                deferred.append(result)
                continue
            kept.append(words)
        selected.append(result)
        selected_ids.add(result.atom.id)

    # Whatever the caps skipped fills the slots nothing else claimed: a graph
    # holding a single topic must still answer with a full top_k.
    for result in deferred:
        if len(selected) >= top_k:
            break
        selected.append(result)

    selected.sort(key=lambda r: r.salience, reverse=True)
    return selected[:top_k]
