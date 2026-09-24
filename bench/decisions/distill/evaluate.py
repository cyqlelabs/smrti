"""What the exported student is worth: agreement with the teacher on the
held-out tenth of every task, per language, through the same ONNX runtime
the servers use; then the two hand-labelled sets (``bench/decisions/run.py``
and ``tone.py``) through the in-process provider, which is the check that
matters — those sets were never in the corpus.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from smrti.decisions.student import registry
from smrti.decisions.student.runtime import Student

from .corpus import STATES
from .label import LABELS, read_rows
from .train import OUT, is_held_out

logger = logging.getLogger("distill.evaluate")


def held_out_agreement(student_dir: Path, tasks: list[str], limit: int = 2000) -> dict[str, Any]:
    student = Student(student_dir, threads=os.cpu_count())
    report: dict[str, Any] = {}
    for task in tasks:
        spec = registry.tasks()[task]
        states = {r["id"]: r for r in read_rows(STATES / f"{task}.jsonl")}
        rows = [r for r in read_rows(LABELS / f"{task}.jsonl") if is_held_out(r["id"]) and r["id"] in states][:limit]
        agree = total = 0
        abs_err = 0.0
        by_lang: dict[str, list[float]] = defaultdict(lambda: [0, 0])
        started = time.monotonic()
        for r in rows:
            answer = student.predict(states[r["id"]]["state"], {
                name: q.payload() for name, q in spec.questions.items()
            } if spec.kind != registry.NOULS else {k: q.payload() for k, q in spec.questions.items()})["answers"]
            if spec.kind == registry.NOULS:
                for key in spec.keys:
                    p, t = answer[key]["noul"], r["targets"][key]
                    ok = (p > 0.5) == (t > 0.5)
                    agree += ok
                    total += 1
                    abs_err += abs(p - t)
                    by_lang[r["lang"]][0] += ok
                    by_lang[r["lang"]][1] += 1
            else:
                name = next(iter(spec.questions))
                choice = answer[name]["choice"]
                teacher = max(r["targets"], key=r["targets"].get)
                ok = choice == teacher
                agree += ok
                total += 1
                abs_err += sum(abs(answer[name]["probabilities"][k] - r["targets"][k]) for k in spec.keys) / spec.width
                by_lang[r["lang"]][0] += ok
                by_lang[r["lang"]][1] += 1
        report[task] = {
            "n": len(rows), "agreement": round(agree / total, 4) if total else None,
            "mean_abs_error": round(abs_err / total, 4) if total else None,
            "ms_per_state": round(1000 * (time.monotonic() - started) / max(len(rows), 1), 1),
            "by_lang": {l: round(a / t, 3) for l, (a, t) in sorted(by_lang.items()) if t},
        }
        logger.info("%s: %s", task, report[task])
    return report


def labelled_sets(student_dir: Path) -> dict[str, Any]:
    """The gate and tone benches under the student, as the engine would run
    it on a machine that chose it."""
    env = {**os.environ, "SMRTI_DECISIONS_ENGINE": "student", "SMRTI_DECISIONS_MODEL": str(student_dir),
           "SMRTI_DECISIONS_URL": ""}
    out: dict[str, Any] = {}
    for name, module in (("gate", "bench.decisions.run"), ("tone", "bench.decisions.tone")):
        result = OUT / f"{name}_student.json"
        proc = subprocess.run([sys.executable, "-m", module, "--json", str(result), "--concurrency", "1"]
                              if name == "gate" else [sys.executable, "-m", module, "--json", str(result)],
                              env=env, capture_output=True, text=True)
        out[name] = {"exit": proc.returncode, "stdout": proc.stdout[-3000:], "stderr": proc.stderr[-1500:]}
        logger.info("%s bench exit %d\n%s", name, proc.returncode, proc.stdout[-2000:])
    return out


def evaluate(student_dir: Path = OUT / "student", tasks: list[str] | None = None) -> dict[str, Any]:
    tasks = tasks or list(registry.tasks())
    report = {"held_out": held_out_agreement(student_dir, tasks), "labelled_sets": labelled_sets(student_dir)}
    (OUT / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
