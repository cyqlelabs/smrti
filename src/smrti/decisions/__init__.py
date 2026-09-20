"""Optional semantic decisions at the points where rules are too coarse.

Smrti's memory engine — SQLite, embeddings, the PLN arithmetic, the
lifecycle rules — is deterministic and stays so. This package adds bounded
judgements at four points where a rule cannot see what a sentence means and
a generative LLM call would be the expensive way to find out: whether a
message holds anything worth extracting (``routing``), which retrieved
memories are evidence for a question (``rerank``), whether a new claim
really replaces an older one (``supersession``), and whether an uncertain
name match is the same entity (``entity``). Each is off unless configured,
runs in shadow before it runs active, and falls back to the deterministic
path on any failure.

The provider is TypeSafe's Jev by default (``smrti.decisions.jev``); the
interface in ``provider.py`` is what a replacement implements.

What a decision may never do: restore a forgotten atom, confer permanence,
change tenant or space scope, or mint a critical warning. A critical
warning needs a valence the caller stated, and a model's reading of the
text is an estimate, which is why no code path here writes
``VALENCE_STATED``.
"""
from __future__ import annotations

import logging
import threading

from .audit import counters as decision_counters, get_all as decision_records
from .engine import DecisionEngine, DecisionOutcome
from .policies import (
    MODE_ACTIVE,
    MODE_OFF,
    MODE_SHADOW,
    TASK_ENTITY,
    TASK_RERANK,
    TASK_ROUTING,
    TASK_SUPERSESSION,
    TASKS,
    DecisionPolicy,
)
from .provider import (
    Choice,
    DecisionProvider,
    DecisionUnavailable,
    Decisions,
    Noul,
    Score,
    StaticProvider,
)

__all__ = [
    "Choice",
    "DecisionEngine",
    "DecisionOutcome",
    "DecisionPolicy",
    "DecisionProvider",
    "DecisionUnavailable",
    "Decisions",
    "MODE_ACTIVE",
    "MODE_OFF",
    "MODE_SHADOW",
    "Noul",
    "Score",
    "StaticProvider",
    "TASKS",
    "TASK_ENTITY",
    "TASK_RERANK",
    "TASK_ROUTING",
    "TASK_SUPERSESSION",
    "build_engine",
    "decision_counters",
    "decision_records",
    "get_decisions",
    "reset_decisions",
]

logger = logging.getLogger("smrti.decisions")

_engine: DecisionEngine | None = None
_engine_lock = threading.Lock()


def build_engine(policy: DecisionPolicy) -> DecisionEngine:
    """An engine for *policy*, with the provider it names when one is
    configured and none otherwise — in which case every task is off
    whatever the policy says, with one warning that says why."""
    provider = None
    if policy.any_enabled:
        if policy.provider == "jev":
            if policy.api_key:
                from .jev import DEFAULT_BASE_URL, DEFAULT_MODEL, JevProvider

                provider = JevProvider(
                    policy.api_key,
                    base_url=policy.url or DEFAULT_BASE_URL,
                    model=policy.model or DEFAULT_MODEL,
                    timeout=policy.timeout,
                )
            else:
                logger.warning(
                    "SMRTI_DECISIONS is enabled but no API key is set "
                    "(SMRTI_DECISIONS_API_KEY or TYPESAFE_API_KEY) — every decision task is off"
                )
        else:
            logger.warning(
                "unknown decision provider %r — every decision task is off", policy.provider
            )
    return DecisionEngine(policy, provider)


def get_decisions() -> DecisionEngine:
    """The process-wide engine, built from the environment on first use."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = build_engine(DecisionPolicy.from_env())
        return _engine


def reset_decisions(engine: DecisionEngine | None = None) -> None:
    """Replace (or drop, to rebuild from the environment) the shared engine."""
    global _engine
    with _engine_lock:
        _engine = engine
