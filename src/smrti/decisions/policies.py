"""What each decision task is allowed to do, and at which thresholds.

Four tasks, each with its own mode and its own rollback switch:

- ``routing`` gates the LLM claim-extraction call behind three yes/no
  judgements about the message (durable? a correction? new?).
- ``rerank`` judges the retrieval shortlist as evidence for the query and
  reorders it before the cut to top_k.
- ``supersession`` verifies that a claim really replaces the earlier one
  before the graph marks the earlier one a loser.
- ``entity`` verifies an uncertain fuzzy or embedding match in entity
  resolution against the other candidates, plus "none" and "ambiguous".

Each runs ``off`` (the deterministic fallback), ``shadow`` (ask, record the
decision, apply nothing) or ``active`` (the default). ``SMRTI_DECISIONS``
sets all four; ``SMRTI_DECISIONS_<TASK>`` overrides one. Thresholds are the
lines the code draws through the provider's probabilities: the provider says
how likely, the policy says what follows, and a decision below its line is
the same as no decision at all.

Decision confidence and memory confidence are different quantities. The
first summarises a distribution over options; the second is accumulated
evidence. The first is stored in the audit record and in decision metadata
on the atom it concerned, never in the atom's truth value.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ACTIVE = "active"
MODES = (MODE_OFF, MODE_SHADOW, MODE_ACTIVE)

TASK_ROUTING = "routing"
TASK_RERANK = "rerank"
TASK_SUPERSESSION = "supersession"
TASK_ENTITY = "entity"
TASKS = (TASK_ROUTING, TASK_RERANK, TASK_SUPERSESSION, TASK_ENTITY)

# The routing gate's lines. A message every one of whose three judgements
# sits under ``skip`` gets no claim extraction; one whose durable-fact or
# correction judgement clears ``force`` gets the LLM even where the entity
# count alone would not have called it. Between the two, the existing
# heuristics decide. Skip is the destructive direction — a false negative is
# a claim never extracted — so its line is low.
DEFAULT_ROUTING_SKIP = 0.2
DEFAULT_ROUTING_FORCE = 0.75

# Reranking: how many of the salience-ranked candidates are judged (one
# request), how much of the final order the judgement decides against the
# local salience rank, and the evidence score under which a candidate is
# dropped — zero, which is "rerank first, filter later": a cutoff comes in
# only once false exclusions have been measured.
DEFAULT_RERANK_SHORTLIST = 20
DEFAULT_RERANK_WEIGHT = 0.5
DEFAULT_RERANK_MIN_EVIDENCE = 0.0

# The choice confidence a verification needs before the code acts on it.
# Under the line both claims are kept (supersession) or a provisional
# duplicate is made (entity): ambiguity preserves rather than merges.
DEFAULT_SUPERSESSION_MIN_CONFIDENCE = 0.6
DEFAULT_ENTITY_MIN_CONFIDENCE = 0.6
DEFAULT_ENTITY_CANDIDATES = 5

# How long a decision may hold up the work it is advising. A decision is an
# aside inside a request a client is waiting on — retrieval itself is tens of
# milliseconds — so the deadline belongs on that scale and not on a model
# download's. At 30s it was longer than the whole budget every caller gives
# the engine: a recall that waited it out returned well after Factor's 10s
# ambient recall and its 30s HTTP ceiling had both given up, so the deadline
# was only ever reached by a request nobody was still listening to.
DEFAULT_TIMEOUT = 5.0

# How long the engine stops asking after the provider could not answer. A
# provider that just failed almost certainly fails again, and paying the
# deadline per request is how one unavailable model becomes a wall in front
# of every recall. One request per window pays it; the rest fall straight
# through to the deterministic path.
DEFAULT_COOLDOWN = 60.0

DEFAULT_CACHE_SIZE = 256


def _mode(value: str | None, fallback: str, name: str) -> str:
    if value is None or value.strip() == "":
        return fallback
    mode = value.strip().lower()
    if mode not in MODES:
        raise ValueError(f"{name} must be one of {', '.join(MODES)}; got {value!r}")
    return mode


def _float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; got {raw!r}") from exc


def _int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc


@dataclass(frozen=True)
class DecisionPolicy:
    modes: dict[str, str] = field(default_factory=lambda: {task: MODE_ACTIVE for task in TASKS})
    model: str = ""
    device: str = ""
    timeout: float = DEFAULT_TIMEOUT
    cooldown: float = DEFAULT_COOLDOWN
    cache_size: int = DEFAULT_CACHE_SIZE
    routing_skip: float = DEFAULT_ROUTING_SKIP
    routing_force: float = DEFAULT_ROUTING_FORCE
    rerank_shortlist: int = DEFAULT_RERANK_SHORTLIST
    rerank_weight: float = DEFAULT_RERANK_WEIGHT
    rerank_min_evidence: float = DEFAULT_RERANK_MIN_EVIDENCE
    supersession_min_confidence: float = DEFAULT_SUPERSESSION_MIN_CONFIDENCE
    entity_min_confidence: float = DEFAULT_ENTITY_MIN_CONFIDENCE
    entity_candidates: int = DEFAULT_ENTITY_CANDIDATES

    def mode(self, task: str) -> str:
        return self.modes.get(task, MODE_OFF)

    def enabled(self, task: str) -> bool:
        """Whether the task asks the provider at all (shadow or active)."""
        return self.mode(task) != MODE_OFF

    def active(self, task: str) -> bool:
        """Whether the task's decisions are applied."""
        return self.mode(task) == MODE_ACTIVE

    @property
    def any_enabled(self) -> bool:
        return any(self.enabled(task) for task in TASKS)

    def with_modes(self, **modes: str) -> "DecisionPolicy":
        """A copy with some task modes replaced — for tests and the bench."""
        merged = dict(self.modes)
        for task, mode in modes.items():
            if task not in TASKS:
                raise ValueError(f"unknown decision task {task!r}")
            merged[task] = _mode(mode, MODE_OFF, task)
        return DecisionPolicy(**{**self.__dict__, "modes": merged})

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "DecisionPolicy":
        env = os.environ if environ is None else environ
        default_mode = _mode(env.get("SMRTI_DECISIONS"), MODE_ACTIVE, "SMRTI_DECISIONS")
        modes = {
            task: _mode(env.get(f"SMRTI_DECISIONS_{task.upper()}"), default_mode, f"SMRTI_DECISIONS_{task.upper()}")
            for task in TASKS
        }
        return cls(
            modes=modes,
            model=env.get("SMRTI_DECISIONS_MODEL", "").strip(),
            device=env.get("SMRTI_DECISIONS_DEVICE", "").strip(),
            timeout=_float(env, "SMRTI_DECISIONS_TIMEOUT", DEFAULT_TIMEOUT),
            cooldown=_float(env, "SMRTI_DECISIONS_COOLDOWN", DEFAULT_COOLDOWN),
            cache_size=_int(env, "SMRTI_DECISIONS_CACHE", DEFAULT_CACHE_SIZE),
            routing_skip=_float(env, "SMRTI_DECISIONS_ROUTING_SKIP", DEFAULT_ROUTING_SKIP),
            routing_force=_float(env, "SMRTI_DECISIONS_ROUTING_FORCE", DEFAULT_ROUTING_FORCE),
            rerank_shortlist=_int(env, "SMRTI_DECISIONS_RERANK_SHORTLIST", DEFAULT_RERANK_SHORTLIST),
            rerank_weight=_float(env, "SMRTI_DECISIONS_RERANK_WEIGHT", DEFAULT_RERANK_WEIGHT),
            rerank_min_evidence=_float(env, "SMRTI_DECISIONS_RERANK_MIN_EVIDENCE", DEFAULT_RERANK_MIN_EVIDENCE),
            supersession_min_confidence=_float(
                env, "SMRTI_DECISIONS_SUPERSESSION_MIN_CONFIDENCE", DEFAULT_SUPERSESSION_MIN_CONFIDENCE
            ),
            entity_min_confidence=_float(env, "SMRTI_DECISIONS_ENTITY_MIN_CONFIDENCE", DEFAULT_ENTITY_MIN_CONFIDENCE),
            entity_candidates=_int(env, "SMRTI_DECISIONS_ENTITY_CANDIDATES", DEFAULT_ENTITY_CANDIDATES),
        )
