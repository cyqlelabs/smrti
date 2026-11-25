"""Regression tests for the retrieval defects the audit reproduced.

The echo tests run on the real multilingual encoder, since the line they pin
was calibrated on it. Everything else uses an embedder answering from a
table, so each similarity is exactly the one the scenario needs.
"""
from __future__ import annotations

import hashlib
import math
import random

import pytest

from smrti import Smrti
from smrti.core.models import Atom, AtomType, AttentionValue, TruthValue, Valence


DIM = 384


def _at(similarity: float, axis: int) -> list[float]:
    """A unit vector whose cosine with the query vector (axis 0) is *similarity*."""
    vec = [0.0] * DIM
    vec[0] = similarity
    vec[axis] = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    return vec


QUERY = _at(1.0, 1)


class _TableEmbedder:
    """Embeds a text from a table, or as a deterministic random unit vector."""

    def __init__(self) -> None:
        self.table: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text in self.table:
            return list(self.table[text])
        rng = random.Random(int(hashlib.sha256(text.encode()).hexdigest(), 16))
        vec = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def mem(tmp_path):
    return Smrti(db_path=str(tmp_path / "audit.db"), personality="balanced",
                 tenant_id="t", write_space="s")


@pytest.fixture
def table_mem(tmp_path):
    m = Smrti(db_path=str(tmp_path / "table.db"), personality="balanced",
              tenant_id="t", write_space="s")
    m.embed = m.atomspace._embed = _TableEmbedder()
    return m


_axes = iter(range(2, DIM))


def _add(m, text: str, similarity: float, type: str = "episode", **cols) -> str:
    """Store *text* at exactly *similarity* to ``QUERY``, then set *cols*."""
    vec = _at(similarity, next(_axes) if similarity < 1.0 else 1)
    metadata = cols.pop("metadata", {})
    atom = Atom(type=AtomType(type), label=text[:100], content=text,
                truth=TruthValue(probability=cols.pop("probability", 0.5),
                                 confidence=cols.get("confidence", 0.5)),
                attention=AttentionValue(), valence=Valence(
                    valence=cols.pop("valence", 0.0), intensity=cols.pop("intensity", 0.0)),
                tenant_id=m.tenant_id, space=cols.pop("space", m.write_space), metadata=metadata)
    atom_id = m.atomspace.add_atom(atom, embedding=vec)
    if cols:
        sets = ", ".join(f"{k} = ?" for k in cols)
        m.db.execute(f"UPDATE atoms SET {sets} WHERE id = ?", (*cols.values(), atom_id))
    m.embed.table[text] = vec
    return atom_id


# ── T11: a short answer is not an echo of its question ────────────────────────

@pytest.mark.parametrize("question, answer", [
    ("What is my sister's name?", "My sister's name is Laura."),
    ("¿Cómo se llama mi hermana?", "Mi hermana se llama Laura."),
    ("मेरी बहन का नाम क्या है", "मेरी बहन का नाम लौरा है"),
])
def test_a_short_answer_sharing_the_questions_words_is_returned_first(mem, question, answer):
    """The answer covers 0.83 of its question's words, as a padded echo does,
    and was zeroed as one; it sits far below the near-identity of a real
    restatement."""
    answer_id = mem.remember(answer)
    mem.remember("The team offsite is planned for October.")
    results = mem.recall(question, min_confidence=0.0, boost=False)
    assert results and results[0].atom.id == answer_id


def test_a_restatement_too_short_to_compare_is_still_an_echo(mem):
    """Two words in common prove nothing, so short pairs were never echoes and
    "deploy friday" outranked the answer; near-identity now decides them, and
    a short answer that is not near-identical still surfaces."""
    echo = mem.remember("deploy friday")
    answer = mem.remember("we never deploy on friday because the on-call is thin")
    hits = {r.atom.id for r in mem.recall("deploy on friday?", min_confidence=0.0, boost=False)}
    assert echo not in hits and answer in hits

    short_answer = mem.remember("favorite color: blue")
    hits = {r.atom.id for r in mem.recall("favorite color", min_confidence=0.0, boost=False)}
    assert short_answer in hits
