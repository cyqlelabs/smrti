"""A behavioral constraint takes a stated valence, not a negative reading."""
import os
import tempfile

import pytest

from smrti import Smrti
from smrti.core.models import Atom, AtomType, RecallResult, Valence
from smrti.core.provenance import VALENCE_STATED
from smrti.retrieval.classify import classify_memory
from smrti.servers.mcp import handle_tool


def _result(atom_type=AtomType.EPISODE, valence=-0.9, stated=True, **truth):
    atom = Atom(
        type=atom_type,
        label="x",
        content="x",
        valence=Valence(valence=valence, intensity=abs(valence)),
        metadata={VALENCE_STATED: True} if stated else {},
        tenant_id="t",
        space="s",
    )
    for key, value in truth.items():
        setattr(atom.truth, key, value)
    return RecallResult(atom=atom, salience=0.0, similarity=0.0)


def test_a_stated_negative_valence_is_a_critical_warning():
    assert classify_memory(_result()) == "critical_warning"


def test_an_estimated_negative_valence_is_only_context():
    """'I didn't understand you' scores negative and is not a mistake."""
    assert classify_memory(_result(stated=False)) == "context"


def test_a_concept_is_never_a_critical_warning():
    """Valence propagates into concepts from every neighbour that mentions them."""
    assert classify_memory(_result(atom_type=AtomType.CONCEPT)) == "context"


def test_mild_negative_valence_stays_context():
    assert classify_memory(_result(valence=-0.2)) == "context"


def test_a_disproven_belief_is_still_an_antipattern():
    got = classify_memory(
        _result(atom_type=AtomType.BELIEF, valence=0.0, stated=False,
                probability=0.1, confidence=0.9)
    )
    assert got == "known_antipattern"


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


def _metadata(mem, atom_id):
    return mem.db.fetchone(
        "SELECT metadata FROM atoms WHERE id = ?", (atom_id,)
    )["metadata"]


@pytest.mark.parametrize("atom_type", ["episode", "belief"])
def test_a_stated_valence_is_recorded(mem, atom_type):
    result = handle_tool(
        mem,
        "smrti_remember",
        {"content": "I deleted the wrong file.", "type": atom_type, "valence": -0.9},
    )
    assert VALENCE_STATED in _metadata(mem, result["atom_id"])


@pytest.mark.parametrize("atom_type", ["episode", "belief"])
def test_an_estimated_valence_is_not_recorded(mem, atom_type):
    """The common write keeps the metadata earlier releases stored."""
    result = handle_tool(
        mem, "smrti_remember", {"content": "Roxana studies pharmacy.", "type": atom_type}
    )
    assert VALENCE_STATED not in _metadata(mem, result["atom_id"])


def test_ambient_conversation_cannot_mint_a_constraint(mem):
    """The regression: every stored turn ran through sentiment estimation."""
    result = handle_tool(
        mem, "smrti_remember", {"content": "¿Qué? No te entendí. Esto es horrible."}
    )
    hits = [h for h in mem.recall("no entendí", top_k=5, min_confidence=0.0)
            if h.atom.id == result["atom_id"]]
    assert hits, "the episode should still be recallable"
    assert classify_memory(hits[0]) == "context"


def test_a_stated_warning_keeps_its_floor_whoever_wrote_it(mem):
    """Classification and retention once drew different lines, so a stated
    warning at -0.6 was handed to the agent as a rule and then pruned."""
    atom_id = mem.remember(
        "Running the migration twice corrupted the ledger",
        valence=-0.6,
        metadata={"source": "agent"},
    )
    for _ in range(100):
        mem.reflect()
    row = mem.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (atom_id,))
    assert row is not None and row["lti"] >= 0.5


def test_an_agents_estimated_distress_earns_no_floor(mem):
    """A model's apology read as severe once outlived every real memory and
    ranked first at a confidence of nothing."""
    atom = Atom(
        type=AtomType.EPISODE,
        label="apology",
        content="I am so sorry, I failed you again",
        valence=Valence(valence=-0.9, intensity=0.9),
        metadata={"source": "agent"},
        tenant_id=mem.tenant_id,
        space=mem.write_space,
    )
    mem.atomspace.add_atom(atom)
    for _ in range(100):
        mem.reflect()
    assert mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (atom.id,)) is None
