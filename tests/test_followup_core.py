"""Regressions from the follow-up to the audit release, storage and
extraction: an atom embedded on its text once, and older vectors re-embedded
by the epoch; pronouns kept off the person a text names; entity creation and
claim records that two processes cannot interleave; and the epoch's
contradiction and prune writes fenced by its lease."""
from __future__ import annotations

import json
import struct
import threading
import time
import types

import numpy as np
import pytest

from smrti import Smrti
from smrti.core import atomspace as atomspace_module
from smrti.core.db import Database, clear_registry, stable_rowid, vec_delete, vec_insert
from smrti.evolution import epoch as epoch_module
from smrti.evolution.epoch import LeaseLost
from smrti.extraction import extract as extract_module
from smrti.extraction.extract import _record_assertion, _resolve_ner_entities
from smrti.extraction.pronouns import merge_pronoun_entities_in_batch
from smrti.extraction.resolve import EntityResolver

T, S = "t", "s"


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "followup_core.db")


@pytest.fixture
def mem(db_path):
    engine = Smrti(db_path=db_path, tenant_id=T, write_space=S)
    yield engine
    clear_registry()


def _vector(db, atom_id):
    row = db.fetchone("SELECT embedding FROM vec_atoms WHERE rowid = ?", (stable_rowid(atom_id),))
    return np.asarray(struct.unpack(f"{len(row['embedding']) // 4}f", row["embedding"]))


def _cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _row(mem, atom_id):
    return mem.db.fetchone("SELECT * FROM atoms WHERE id = ?", (atom_id,))


# ── one copy of the text in each vector ─────────────────────────────────────

def test_an_episode_is_embedded_on_its_text_once(mem):
    """An episode's label is the first hundred characters of its content, and
    both were embedded, so a verbatim copy of a short stored sentence reached
    a median 0.93 against its own vector. A new file is born in the current
    format, with nothing to re-embed."""
    text = "My sister's name is Laura."
    atom_id = mem.remember(text)

    assert _cosine(_vector(mem.db, atom_id), mem.embed.embed(text)) > 0.999
    assert mem.db.meta_get("vector_format") == "2"
    assert mem.db.fetchall("SELECT key FROM smrti_meta WHERE key LIKE 'vector_reembed %'") == []


def test_a_file_with_older_vectors_is_re_embedded_a_batch_per_epoch(db_path, monkeypatch):
    """Vectors written by an older build embed the opening twice, and mixing
    them with new ones in one index skews every search. Opening the file
    marks its spaces, and each epoch re-embeds the next batch until the space
    is done, rather than stalling the open on a whole-file encoding."""
    from smrti.core.db import reembed_key

    mem = Smrti(db_path=db_path, tenant_id=T, write_space=S)
    texts = [
        "The spare keys are in the blue box.",
        "We moved the meeting to Thursday afternoon.",
        "Laura prefers the window seat on long flights.",
    ]
    ids = [mem.remember(text) for text in texts]
    old = {}
    for atom_id, text in zip(ids, texts):
        old[atom_id] = mem.embed.embed(f"{text[:100]} {text}")
        mem.db.execute_batch([
            *vec_delete([atom_id]),
            vec_insert(atom_id, struct.pack(f"{len(old[atom_id])}f", *old[atom_id]), T, S, text[:100]),
        ])
    mem.db.execute("DELETE FROM smrti_meta WHERE key = 'vector_format'")
    clear_registry()

    mem = Smrti(db_path=db_path, tenant_id=T, write_space=S)
    assert mem.db.meta_get(reembed_key(T, S)) == ""
    assert all(_cosine(_vector(mem.db, atom_id), old[atom_id]) > 0.999 for atom_id in ids)

    monkeypatch.setattr(epoch_module, "_REEMBED_BATCH", 2)
    mem.reflect()
    fresh = [atom_id for atom_id, text in zip(ids, texts)
             if _cosine(_vector(mem.db, atom_id), mem.embed.embed(text)) > 0.999]
    assert len(fresh) == 2
    assert mem.db.meta_get(reembed_key(T, S)) == max(fresh)

    mem.reflect()
    assert all(_cosine(_vector(mem.db, atom_id), mem.embed.embed(text)) > 0.999
               for atom_id, text in zip(ids, texts))
    assert mem.db.meta_get(reembed_key(T, S)) is None
    clear_registry()
