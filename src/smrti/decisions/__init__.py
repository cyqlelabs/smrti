"""Local semantic decisions at the points where rules are too coarse.

Smrti's memory engine — SQLite, embeddings, the PLN arithmetic, the
lifecycle rules — is deterministic and stays so. This package adds bounded
judgements at four points where a rule cannot see what a sentence means and
a generative LLM call would be the expensive way to find out: whether a
message holds anything worth extracting (``routing``), which retrieved
memories are evidence for a question (``rerank``), whether a new claim
really replaces an older one (``supersession``), and whether an uncertain
name match is the same entity (``entity``). Each is active by default, can be
put in shadow or disabled independently, and falls back to the deterministic
path on any failure.

The provider is the local multilingual Laya model
(``smrti.decisions.laya``); the interface in ``provider.py`` keeps callers
independent of the model runtime.

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
    """An engine for *policy*: the server ``SMRTI_DECISIONS_URL`` names when
    there is one, a local Laya provider otherwise, and no provider when no
    task is configured."""
    provider = None
    if policy.any_enabled and policy.url:
        from .remote import RemoteProvider

        provider = RemoteProvider(policy.url)
    elif policy.any_enabled:
        from .laya import DEFAULT_MODEL, LayaProvider

        provider = LayaProvider(
            model=policy.model or DEFAULT_MODEL,
            device=policy.device or None,
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
    """Replace the shared engine and release the provider it owned."""
    global _engine
    with _engine_lock:
        previous = _engine
        _engine = engine
    if previous is None or previous is engine:
        return
    close = getattr(previous.provider, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.warning("could not close the previous decision provider", exc_info=True)
