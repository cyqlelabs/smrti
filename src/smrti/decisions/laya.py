"""Local adapter for the multilingual Laya typed-decision model.

The model is Laya; the runtime is EdgeJev, which serves the same checkpoint
as an int8 ONNX graph through onnxruntime. Nothing about the decisions
changes — same three primitives, same calibration, same thresholds — but the
weights are 343 MB instead of 1290 MB, resident size is about 560 MB instead
of 2.9 GB, and torch is gone from the dependency tree entirely. On the boxes
Smrti is meant to share with an agent, that is the difference between a
decision engine and a machine that swaps.

Laya exposes the same three primitives Smrti uses (``noul``, ``choice`` and
``score``), so the adapter only has to translate the question dataclasses to
Laya's dictionaries and validate its answer through the shared provider
boundary. The loading, deadline and accounting machinery is
:class:`smrti.decisions.local.LocalProvider`'s.

A local model directory can be supplied through ``SMRTI_DECISIONS_MODEL`` to
avoid any first-run download; see :mod:`smrti.decisions.model` for where the
weights are looked for otherwise.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from . import model as model_source
from .local import LOAD_RETRY_SECONDS, LocalProvider  # noqa: F401  (re-exported for callers)
from .provider import DecisionUnavailable

logger = logging.getLogger("smrti.decisions.laya")


def _threads() -> int | None:
    """How many cores inference may hold.

    ``SMRTI_DECISIONS_THREADS`` decides; unset leaves the runtime's own
    default. The cap exists because the machines this matters on have two
    slow cores, and a decision that takes the whole box for 200 ms is worse
    than one that takes half of it for 400.
    """
    raw = os.environ.get("SMRTI_DECISIONS_THREADS", "").strip()
    if not raw:
        return None
    try:
        threads = int(raw)
    except ValueError:
        logger.warning("SMRTI_DECISIONS_THREADS=%r is not a number; ignoring it", raw)
        return None
    return threads if threads > 0 else None

# Empty means "resolve it": an explicit directory, the one Factor already
# has, or Smrti's own — see :func:`smrti.decisions.model.resolve`.
DEFAULT_MODEL = ""

# How many questions ride in one native call.
#
# Batching is not an optimization here, and it does not leave the answer
# alone either. Measured on the real checkpoint against twenty candidates'
# eighty evidence questions: one at a time took 185.6s and peaked at 1015 MB,
# while all eighty in a single call took 197.9s and peaked at 21798 MB. The
# one big call is slower *and* twenty-one times the memory, because the
# runtime reserves the whole batch's attention buffers before it starts and
# its session holds them at that peak afterwards. That is how a 100 MB engine
# became a 25 GB one within a minute of a single turn.
#
# It is also a different answer, reproducibly: run twice, each shape repeats
# itself exactly (drift 0.0000), while one question's probability moves by up
# to 0.33 depending only on which other questions were sent with it. These
# are independent propositions about independent candidates, so an answer
# that moves with its neighbours is the contaminated one. One question per
# call is the only shape that answers each on its own evidence.
MAX_QUESTIONS_PER_CALL = 1


class LayaProvider(LocalProvider):
    """Laya through EdgeJev, loaded lazily and asked one question at a time."""

    name = "laya"
    questions_per_call = MAX_QUESTIONS_PER_CALL

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        agent: Any | None = None,
    ) -> None:
        super().__init__(model)
        self.device = device or None
        self._agent = agent

    def _load_agent(self) -> Any:
        try:
            import edgejev
        except ImportError as exc:
            raise DecisionUnavailable(
                "the decision runtime is missing from the Smrti installation; "
                "reinstall Smrti's dependencies"
            ) from exc
        try:
            directory = self.model or str(model_source.resolve())
        except Exception as exc:
            raise DecisionUnavailable(f"no decision model to load: {exc}") from exc
        try:
            agent = edgejev.Agent(directory, threads=_threads(), provider=self.device or None)
        except Exception as exc:
            raise DecisionUnavailable(f"could not load the decision model at {directory!r}: {exc}") from exc
        self.model = directory
        return agent
