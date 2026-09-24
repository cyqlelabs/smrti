"""Answering with a trained student: one encoder pass per head, no question
text in the input.

An artifact directory holds ``student.json`` (the heads: task, kind, keys,
where each one's logits sit in the graph's one output, and the temperature
its probabilities are read at), the ONNX graph and the tokenizer. The
runtime's whole dependency list is onnxruntime, tokenizers and numpy.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..provider import Choice, DecisionUnavailable, Noul, Question, Score
from . import registry

CONFIG_NAME = "student.json"
FORMAT = "smrti-student/1"

# What the graph reads at most. The encoder has 512 positions; the rest of
# the window is the question head Laya would carry, which the student does
# not, so a caller sizing its state against ``max_len - head_max_len``
# (Factor does) lands on the encoder's whole window.
MAX_LEN = 512
HEAD_MAX_LEN = 192


def is_ready(directory: Path | str) -> bool:
    """Whether *directory* holds a usable student: the metadata, and the
    graph and tokenizer it names."""
    directory = Path(directory)
    try:
        meta = json.loads((directory / CONFIG_NAME).read_text())
    except (OSError, ValueError):
        return False
    if meta.get("format") != FORMAT:
        return False
    try:
        return all(
            (directory / meta[key]).stat().st_size > 0 for key in ("encoder", "tokenizer")
        )
    except (OSError, KeyError):
        return False


def questions_from_payload(payload: Mapping[str, Any]) -> dict[str, Question]:
    """The typed questions of a wire request, validated the way the
    provider boundary validates answers: a malformed question is refused
    here, by name."""
    out: dict[str, Question] = {}
    for name, raw in payload.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"question {name!r} is not an object")
        kind = raw.get("type")
        instructions = raw.get("instructions", "")
        criteria = raw.get("criteria")
        try:
            if kind == "noul":
                crit = criteria if isinstance(criteria, Mapping) else {}
                out[name] = Noul(instructions, true=crit.get("true") or None, false=crit.get("false") or None)
            elif kind == "choice":
                if not isinstance(criteria, Mapping):
                    raise ValueError("a choice needs a criteria object")
                out[name] = Choice(instructions, dict(criteria))
            elif kind == "score":
                if not isinstance(criteria, list):
                    raise ValueError("a score needs a criteria list")
                out[name] = Score(instructions, list(criteria))
            else:
                raise ValueError(f"unknown question type {kind!r}")
        except ValueError as exc:
            raise ValueError(f"question {name!r}: {exc}") from exc
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class Student:
    """A loaded student: tokenizer, graph, heads."""

    def __init__(self, directory: Path | str, *, threads: int | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.directory = Path(directory)
        meta = json.loads((self.directory / CONFIG_NAME).read_text())
        if meta.get("format") != FORMAT:
            raise ValueError(f"{self.directory / CONFIG_NAME} is not a {FORMAT} artifact")
        self.model_name = str(meta.get("model_name") or "smrti-student")
        self.max_len = int(meta.get("max_len") or MAX_LEN)
        self.heads: dict[str, dict[str, Any]] = {h["task"]: h for h in meta["heads"]}
        self.tokenizer = Tokenizer.from_file(str(self.directory / meta["tokenizer"]))
        self.tokenizer.enable_truncation(self.max_len)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or (os.cpu_count() or 2)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Scratch returns to the allocator between runs instead of staying
        # reserved at its peak; the box this exists for has no spare.
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            str(self.directory / meta["encoder"]), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self.session.get_inputs()}

    def _logits(self, text: str) -> tuple[np.ndarray, int]:
        encoded = self.tokenizer.encode(text)
        ids = np.array([encoded.ids], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.session.run(None, feed)[0]
        return np.asarray(out[0], dtype=np.float64), len(encoded.ids)

    def _answer(self, m: registry.Match, state: Any) -> tuple[dict[str, Any], int]:
        head = self.heads.get(m.spec.name)
        if head is None:
            raise DecisionUnavailable(f"this student has no head for {m.spec.name!r}")
        logits, tokens = self._logits(registry.encoder_text(m.spec.name, state))
        offset, width = int(head["offset"]), int(head["width"])
        slice_ = logits[offset : offset + width] / float(head.get("temperature") or 1.0)
        keys = [str(k) for k in head["keys"]]
        answers: dict[str, Any] = {}
        if m.spec.kind == registry.NOULS:
            by_key = {k: 1.0 / (1.0 + math.exp(-float(v))) for k, v in zip(keys, slice_)}
            for name, key in m.names.items():
                p = by_key[key]
                answers[name] = {"type": "noul", "noul": round(p, 4), "confidence": round(max(p, 1 - p), 4)}
            return answers, tokens
        probs = _softmax(slice_)
        name = next(iter(m.names))
        dist = {k: round(float(p), 4) for k, p in zip(keys, probs)}
        top = int(np.argmax(probs))
        if m.spec.kind == registry.CHOICE:
            answers[name] = {"type": "choice", "choice": keys[top], "probabilities": dist,
                             "confidence": round(float(probs[top]), 4)}
        else:
            expected = float((np.arange(len(probs)) * probs).sum())
            answers[name] = {"type": "score", "score": round(expected, 4), "probabilities": dist,
                             "confidence": round(float(probs[top]), 4)}
        return answers, tokens

    def predict(self, state: Any, questions_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Answer a wire request's questions. Raises ``DecisionUnavailable``
        for a question no head answers, so the caller falls back rather
        than parses a partial reply."""
        questions = questions_from_payload(questions_payload)
        try:
            matches = registry.match(questions)
        except LookupError as exc:
            raise DecisionUnavailable(f"the student cannot answer this: {exc}") from exc
        answers: dict[str, Any] = {}
        tokens = 0
        for m in matches:
            part, n = self._answer(m, state)
            answers.update(part)
            tokens += n
        return {"model": self.model_name, "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0}}
