"""The student as an in-process provider: the same lazy load and deadlines
as Laya's adapter, answering every question of a request in one call
because one pass answers a whole head."""
from __future__ import annotations

from typing import Any

from .. import model as model_source
from ..laya import _threads
from ..local import LocalProvider
from ..provider import DecisionUnavailable


class StudentProvider(LocalProvider):
    name = "student"
    questions_per_call = 256

    def __init__(self, model: str = "", *, agent: Any | None = None) -> None:
        super().__init__(model)
        self._agent = agent

    def _load_agent(self) -> Any:
        from .runtime import Student

        try:
            directory = self.model or str(model_source.resolve_student())
        except Exception as exc:
            raise DecisionUnavailable(f"no student model to load: {exc}") from exc
        try:
            student = Student(directory, threads=_threads())
        except Exception as exc:
            raise DecisionUnavailable(f"could not load the student model at {directory!r}: {exc}") from exc
        self.model = directory
        return student
