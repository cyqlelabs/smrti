"""The teacher's answers: Laya, asked the registry's own questions about
every state, one question per call, over the server ``SMRTI_DECISIONS_URL``
names (the desktop's EdgeJev answers about forty a second).

What is stored is the probability the teacher gave — a noul's probability,
a choice's whole distribution — never a hard label: the student is trained
to reproduce the teacher's calibration, not only its argmax. Labels are
appended to ``labels/<task>.jsonl`` by state id, so a re-run labels only
what is new.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

from smrti.decisions.provider import Choice, DecisionUnavailable, Noul, Score
from smrti.decisions.remote import RemoteProvider
from smrti.decisions.student import registry

from .corpus import STATES
from .sources import DATA

logger = logging.getLogger("distill.label")

LABELS = Path(DATA) / "distill" / "labels"
TEACHER_URL = os.environ.get("SMRTI_DECISIONS_URL", "http://127.0.0.1:8731")
CONCURRENCY = int(os.environ.get("SMRTI_DISTILL_CONCURRENCY", "8"))
TIMEOUT = 60.0


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _targets(provider: RemoteProvider, spec: registry.TaskSpec, state: Any) -> dict[str, float]:
    """The teacher's probabilities for one state, keyed by head key."""
    if spec.kind == registry.NOULS:
        out: dict[str, float] = {}
        for key, q in spec.questions.items():
            assert isinstance(q, Noul)
            out[key] = provider.ask(state, {key: q}, timeout=TIMEOUT).noul(key)
        return out
    name, q = next(iter(spec.questions.items()))
    answer = provider.ask(state, {name: q}, timeout=TIMEOUT)[name]
    if isinstance(q, Choice):
        dist = {k: float(answer.probabilities.get(k, 0.0)) for k in spec.keys}
    else:
        assert isinstance(q, Score)
        dist = {k: float(answer.probabilities.get(k, 0.0)) for k in spec.keys}
    total = sum(dist.values()) or 1.0
    return {k: v / total for k, v in dist.items()}


def label(tasks: Iterable[str], url: str = TEACHER_URL, limit: int | None = None) -> dict[str, int]:
    LABELS.mkdir(parents=True, exist_ok=True)
    provider = RemoteProvider(url)
    counts: dict[str, int] = {}
    for task in tasks:
        spec = registry.tasks()[task]
        path = LABELS / f"{task}.jsonl"
        done = {r["id"] for r in read_rows(path)}
        todo = [r for r in read_rows(STATES / f"{task}.jsonl") if r["id"] not in done]
        if limit is not None:
            todo = todo[:limit]
        logger.info("%s: %d states to label (%d already)", task, len(todo), len(done))
        started = time.monotonic()
        failed = 0
        with path.open("a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(CONCURRENCY) as ex:
            futures = {ex.submit(_targets, provider, spec, r["state"]): r for r in todo}
            for n, fut in enumerate(cf.as_completed(futures), 1):
                row = futures[fut]
                try:
                    targets = fut.result()
                except DecisionUnavailable as exc:
                    failed += 1
                    if failed <= 5:
                        logger.warning("%s %s: %s", task, row["id"], exc)
                    continue
                fh.write(json.dumps({"id": row["id"], "task": task, "lang": row.get("lang"),
                                     "source": row.get("source"), "targets": targets}, ensure_ascii=False) + "\n")
                if n % 500 == 0:
                    rate = n / (time.monotonic() - started)
                    logger.info("%s: %d/%d labelled, %.1f/s, %d failed", task, n, len(todo), rate, failed)
            fh.flush()
        counts[task] = len(done) + len(todo) - failed
        logger.info("%s: %d labelled in %.0fs (%d failed)", task, counts[task], time.monotonic() - started, failed)
    provider.close()
    return counts
