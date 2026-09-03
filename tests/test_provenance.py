"""Tests for source provenance: agent-authored memory must fade unless adopted.

What the user says outranks what the model replies. A model turn that merely
offers suggestions produces entities nobody asked for, and without provenance
those are indistinguishable from facts the user stated — same confidence, same
attention, same permanence. These tests pin the asymmetry end to end.
"""
import json
import os
import tempfile

import pytest

from smrti import Smrti
from smrti.core.db import Database
from smrti.core.embed import EmbeddingProvider
from smrti.extraction.resolve import EntityResolver


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def mem(db_path):
    m = Smrti(db_path=db_path, personality="balanced", tenant_id="t1", write_space="s1")
    yield m
    m.db.close()


@pytest.fixture
def resolvers(db_path):
    """A user-sourced and an agent-sourced resolver over one database."""
    db = Database(db_path)
    db.initialize()
    embed = EmbeddingProvider()
    yield (
        EntityResolver(db, embed, source="user", episode_id="ep-user"),
        EntityResolver(db, embed, source="agent", agent_trust=0.5, episode_id="ep-agent"),
        db,
    )
    db.close()


def _atom(db, atom_id):
    return db.fetchone("SELECT * FROM atoms WHERE id = ?", (atom_id,))


# ── provenance is recorded on derived atoms ───────────────────────────────────

def test_agent_derived_atoms_are_stamped(resolvers):
    user_res, agent_res, db = resolvers
    user_id = user_res.resolve("Kubernetes", "technology", "t1", "s1", ["s1"])
    agent_id = agent_res.resolve("Feria del Libro", "event", "t1", "s1", ["s1"])

    assert json.loads(_atom(db, agent_id)["metadata"])["source"] == "agent"
    # Absence of a source means user-authored, which is also what pre-upgrade
    # atoms look like — they must not be retroactively distrusted.
    assert json.loads(_atom(db, user_id)["metadata"]) == {}


def test_agent_derived_atoms_start_weaker(resolvers):
    user_res, agent_res, db = resolvers
    user_id = user_res.resolve("Kubernetes", "technology", "t1", "s1", ["s1"])
    agent_id = agent_res.resolve("Feria del Libro", "event", "t1", "s1", ["s1"])

    user_atom, agent_atom = _atom(db, user_id), _atom(db, agent_id)
    for field in ("confidence", "sti", "lti"):
        assert agent_atom[field] < user_atom[field], f"{field} must be damped"
    assert agent_atom["confidence"] == pytest.approx(user_atom["confidence"] * 0.5)


def test_trust_scales_the_damping(db_path):
    db = Database(db_path)
    db.initialize()
    embed = EmbeddingProvider()
    trusting = EntityResolver(db, embed, source="agent", agent_trust=0.9)
    skeptical = EntityResolver(db, embed, source="agent", agent_trust=0.1)

    a = trusting.resolve("Alpha", "concept", "t1", "s1", ["s1"])
    b = skeptical.resolve("Beta", "concept", "t2", "s1", ["s1"])
    assert _atom(db, a)["confidence"] > _atom(db, b)["confidence"]
    db.close()


def test_user_source_is_never_damped_even_with_low_trust(db_path):
    """agent_trust must not leak into user-authored extraction."""
    db = Database(db_path)
    db.initialize()
    from smrti.core.models import INITIAL_CONFIDENCE

    res = EntityResolver(db, embed_engine=EmbeddingProvider(), source="user", agent_trust=0.1)
    atom_id = res.resolve("Kubernetes", "technology", "t1", "s1", ["s1"])
    assert _atom(db, atom_id)["confidence"] == pytest.approx(INITIAL_CONFIDENCE["concept"])
    db.close()


# ── re-mention writes weighted corroboration ──────────────────────────────────

def test_re_mention_logs_evidence_weighted_by_source(resolvers):
    user_res, agent_res, db = resolvers
    atom_id = user_res.resolve("Kubernetes", "technology", "t1", "s1", ["s1"])
    user_res.resolve("Kubernetes", "technology", "t1", "s1", ["s1"])   # user repeats it
    agent_res.resolve("Kubernetes", "technology", "t1", "s1", ["s1"])  # model repeats it

    rows = db.fetchall(
        "SELECT weight, source_episode_id FROM evidence WHERE atom_id = ? ORDER BY weight",
        (atom_id,),
    )
    assert [r["weight"] for r in rows] == [0.5, 1.0]
    assert {r["source_episode_id"] for r in rows} == {"ep-user", "ep-agent"}


def test_creation_does_not_log_evidence(resolvers):
    """Only re-mentions corroborate; the first sighting is the atom itself."""
    user_res, _, db = resolvers
    atom_id = user_res.resolve("Kubernetes", "technology", "t1", "s1", ["s1"])
    assert db.fetchall("SELECT id FROM evidence WHERE atom_id = ?", (atom_id,)) == []


def test_user_corroboration_outpaces_agent_corroboration(mem):
    """Equal numbers of mentions must not build equal confidence."""
    res_user = EntityResolver(mem.db, mem.embed, source="user", episode_id="e1")
    res_agent = EntityResolver(mem.db, mem.embed, source="agent", episode_id="e2")
    spoken = res_user.resolve("Kubernetes", "technology", "t1", "s1", ["s1"])
    echoed = res_user.resolve("Terraform", "technology", "t1", "s1", ["s1"])
    # Same starting point; only the re-mention source differs.
    mem.db.execute(
        "UPDATE atoms SET confidence = 0.3 WHERE id IN (?, ?)", (spoken, echoed)
    )
    for _ in range(5):
        res_user.resolve("Kubernetes", "technology", "t1", "s1", ["s1"])
        res_agent.resolve("Terraform", "technology", "t1", "s1", ["s1"])
    mem.reflect()

    assert _atom(mem.db, spoken)["confidence"] > _atom(mem.db, echoed)["confidence"]


# ── decay and pruning asymmetry ───────────────────────────────────────────────

def test_agent_atoms_decay_faster_than_user_atoms(mem):
    user_id = mem.remember("user stated fact", type="concept")
    agent_id = mem.remember(
        "model suggested aside", type="concept", metadata={"source": "agent"}
    )
    mem.db.execute(
        "UPDATE atoms SET confidence = 0.8, lti = 0.6 WHERE id IN (?, ?)",
        (user_id, agent_id),
    )
    for _ in range(10):
        mem.reflect()

    assert _atom(mem.db, agent_id)["confidence"] < _atom(mem.db, user_id)["confidence"]
    assert _atom(mem.db, agent_id)["lti"] < _atom(mem.db, user_id)["lti"]


def test_agent_episodes_are_prunable_but_user_episodes_are_not(mem):
    """Episodes are exempt from pruning only when the user authored them."""
    user_ep = mem.remember("something the user told us")
    agent_ep = mem.remember("a reply nobody acted on", metadata={"source": "agent"})
    mem.db.execute(
        "UPDATE atoms SET confidence = 0.01, lti = 0.0 WHERE id IN (?, ?)",
        (user_ep, agent_ep),
    )
    mem.reflect()

    assert _atom(mem.db, user_ep) is not None
    assert _atom(mem.db, agent_ep) is None


def test_agent_beliefs_are_prunable_but_user_beliefs_are_not(mem):
    """A preference the model invented must not outlive its usefulness."""
    user_belief = mem.remember("user prefers dark mode", type="belief")
    agent_belief = mem.remember(
        "prefers free events", type="belief", metadata={"source": "agent"}
    )
    mem.db.execute(
        "UPDATE atoms SET confidence = 0.01, lti = 0.0 WHERE id IN (?, ?)",
        (user_belief, agent_belief),
    )
    mem.reflect()

    assert _atom(mem.db, user_belief) is not None
    assert _atom(mem.db, agent_belief) is None


def test_atoms_without_source_are_treated_as_user_authored(mem):
    """Pre-upgrade atoms carry no provenance and must keep full trust."""
    legacy = mem.remember("recorded before provenance existed")
    mem.db.execute("UPDATE atoms SET metadata = '{}' WHERE id = ?", (legacy,))
    mem.db.execute("UPDATE atoms SET confidence = 0.01, lti = 0.0 WHERE id = ?", (legacy,))
    mem.reflect()
    assert _atom(mem.db, legacy) is not None


def test_null_metadata_does_not_break_decay(mem):
    """A NULL metadata column must not silently drop atoms from the decay pass."""
    atom_id = mem.remember("legacy row", type="concept")
    mem.db.execute(
        "UPDATE atoms SET metadata = NULL, confidence = 0.8 WHERE id = ?", (atom_id,)
    )
    mem.reflect()
    assert _atom(mem.db, atom_id)["confidence"] < 0.8


# ── the scenario this behaviour exists for ────────────────────────────────────

def test_high_fan_out_agent_reply_does_not_outrank_user_facts(mem):
    """A model turn that lists 40 venues must not bury what the user told us.

    Reproduces the failure this provenance work addresses: one assistant reply
    ("here are some things to do this weekend") produced roughly half the
    concepts in a real graph, all pinned at maximum attention, permanently.
    """
    mem.set_personality("maverick")
    user_res = EntityResolver(mem.db, mem.embed, source="user")
    agent_res = EntityResolver(
        mem.db, mem.embed, source="agent", agent_trust=0.5,
    )

    user_ep = mem.remember("Mi nombre es Nicolas y soy programador")
    user_ids = [
        user_res.resolve(name, etype, "t1", "s1", ["s1"])
        for name, etype in (
            ("Nicolas", "person"), ("programador", "role"), ("Python", "technology"),
        )
    ]
    for atom_id in user_ids:
        mem.atomspace.link_atoms(user_ep, atom_id, "mentions", "t1", "s1")

    agent_ep = mem.remember(
        "Aqui tienes actividades para el fin de semana", metadata={"source": "agent"}
    )
    # Distinct names on purpose: near-identical labels would fuzzy-merge into a
    # handful of atoms and repeatedly re-boost them, which is a different bug.
    agent_ids = [
        agent_res.resolve(name, "event", "t1", "s1", ["s1"])
        for name in (
            "Feria del Libro", "Teatro 3 de Febrero", "Milonga La Pagana",
            "Sala Mayo", "Vieja Usina", "Mercado Sud", "Puerto Nuevo",
            "Sociedad Friulana", "Gregoria Matorras", "Mauricio Dayub",
            "Bv. Racedo 250", "Oro Verde", "Villaguay", "Nogoya",
            "Kermes para las infancias", "Festival Agite", "Virtual Groove",
            "Culpables de este humor", "Parientes del Bar", "Suma de Voluntades",
            "Centro Cultural Juan L. Ortiz", "Esmeralda", "Cautelares",
            "alimento no perecedero", "Una gaviota en el rio", "El Equilibrista",
        )
    ]
    for atom_id in agent_ids:
        mem.atomspace.link_atoms(agent_ep, atom_id, "mentions", "t1", "s1")

    def alive(ids):
        ph = ",".join("?" * len(ids))
        return mem.db.fetchone(
            f"SELECT COUNT(*) n FROM atoms WHERE id IN ({ph})", tuple(ids)
        )["n"]

    # The fan-out must not make the noise more salient than the signal.
    peak_agent = mem.db.fetchone(
        f"SELECT MAX(sti) m FROM atoms WHERE id IN ({','.join('?' * len(agent_ids))})",
        tuple(agent_ids),
    )["m"]
    peak_user = mem.db.fetchone(
        f"SELECT MAX(sti) m FROM atoms WHERE id IN ({','.join('?' * len(user_ids))})",
        tuple(user_ids),
    )["m"]
    assert peak_agent <= peak_user

    for _ in range(300):
        mem.reflect()

    assert alive(agent_ids) == 0, "unadopted model output must not persist"
    assert alive(user_ids) == len(user_ids), "user-stated facts must survive"
    assert _atom(mem.db, agent_ep) is None
    assert _atom(mem.db, user_ep) is not None


# ── metadata robustness on pre-existing databases ─────────────────────────────

@pytest.mark.parametrize(
    "metadata",
    ["not json at all", "", "[1, 2, 3]", '{"source": null}', '{"other": "field"}'],
)
def test_epoch_survives_unparseable_metadata(mem, metadata):
    """json_extract raises on invalid JSON instead of returning NULL.

    A whole-table decay pass touches every atom, so one bad row would abort
    consolidation for the entire space — on databases that worked fine before
    the upgrade.
    """
    atom_id = mem.remember("an atom from an older build", type="concept")
    mem.db.execute(
        "UPDATE atoms SET metadata = ?, confidence = 0.8 WHERE id = ?", (metadata, atom_id)
    )
    mem.reflect()  # must not raise
    assert _atom(mem.db, atom_id)["confidence"] < 0.8


def test_unparseable_metadata_is_treated_as_user_authored(mem):
    """Unreadable provenance must fail safe: keep the memory, don't prune it."""
    atom_id = mem.remember("episode with corrupt metadata")
    mem.db.execute(
        "UPDATE atoms SET metadata = 'garbage', confidence = 0.01, lti = 0.0 WHERE id = ?",
        (atom_id,),
    )
    mem.reflect()
    assert _atom(mem.db, atom_id) is not None


def test_contradiction_resolution_survives_unparseable_metadata(mem):
    """The resolved-marker lookup reads metadata on every contradiction edge."""
    a = mem.remember("the deploy is safe", type="belief")
    b = mem.remember("the deploy is not safe", type="belief")
    mem.db.execute("UPDATE atoms SET confidence = 0.9 WHERE id = ?", (a,))
    mem.db.execute("UPDATE atoms SET confidence = 0.2 WHERE id = ?", (b,))
    edge = mem.atomspace.link_atoms(a, b, "contradicts", "t1", "s1")
    mem.db.execute("UPDATE atoms SET metadata = 'not json' WHERE id = ?", (edge,))

    result = mem.reflect()  # must not raise
    assert result.contradictions_resolved == 1


# ── durability floor: user facts persist, model output does not ───────────────

def test_user_concepts_survive_indefinite_idleness(mem):
    """A fact the user stated must not rot away just because it goes unmentioned.

    Both terms of the prune predicate fall monotonically without new mentions,
    so an unfloored LTI puts every user fact on a one-way trip to deletion —
    and core identity facts are exactly what goes unmentioned longest.
    """
    mem.set_personality("maverick")
    res = EntityResolver(mem.db, mem.embed, source="user")
    atom_id = res.resolve("Nicolas Iglesias", "person", "t1", "s1", ["s1"])

    for _ in range(1000):
        mem.reflect()

    row = _atom(mem.db, atom_id)
    assert row is not None, "user-stated fact deleted after idle epochs"
    assert row["lti"] >= 0.1, "user LTI floor breached"


def test_agent_concepts_still_decay_to_nothing(mem):
    """The floor must not resurrect the noise it was introduced alongside."""
    mem.set_personality("maverick")
    res = EntityResolver(mem.db, mem.embed, source="agent", agent_trust=0.5)
    atom_id = res.resolve("Feria del Libro", "event", "t1", "s1", ["s1"])

    for _ in range(300):
        mem.reflect()

    assert _atom(mem.db, atom_id) is None, "unadopted model output persisted"


def test_lti_floor_does_not_lift_atoms_that_never_earned_it(mem):
    """The floor holds an atom that reached it; it never promotes one to it."""
    atom_id = mem.remember("never promoted", type="concept")
    mem.db.execute("UPDATE atoms SET lti = 0.02, sti = 0.0 WHERE id = ?", (atom_id,))
    mem.reflect()

    row = _atom(mem.db, atom_id)
    if row is not None:  # may be pruned outright, which is also correct
        assert row["lti"] < 0.1


# ── adoption: the user incorporating model output ─────────────────────────────

def test_user_mention_adopts_an_agent_authored_atom(mem):
    """Model output earns permanence exactly when the user picks it up."""
    agent_res = EntityResolver(mem.db, mem.embed, source="agent", agent_trust=0.5)
    atom_id = agent_res.resolve("Feria del Libro", "event", "t1", "s1", ["s1"])
    assert json.loads(_atom(mem.db, atom_id)["metadata"])["source"] == "agent"

    EntityResolver(mem.db, mem.embed, source="user").resolve(
        "Feria del Libro", "event", "t1", "s1", ["s1"]
    )
    assert json.loads(_atom(mem.db, atom_id)["metadata"])["source"] == "user"


def test_adopted_atoms_become_durable(mem):
    """Adoption must actually change the outcome, not just the label."""
    mem.set_personality("maverick")
    agent_res = EntityResolver(mem.db, mem.embed, source="agent", agent_trust=0.5)
    adopted = agent_res.resolve("Feria del Libro", "event", "t1", "s1", ["s1"])
    ignored = agent_res.resolve("Milonga La Pagana", "event", "t1", "s1", ["s1"])

    EntityResolver(mem.db, mem.embed, source="user").resolve(
        "Feria del Libro", "event", "t1", "s1", ["s1"]
    )
    for _ in range(300):
        mem.reflect()

    assert _atom(mem.db, adopted) is not None, "adopted suggestion was pruned"
    assert _atom(mem.db, ignored) is None, "unadopted suggestion survived"


def test_agent_mention_does_not_adopt(mem):
    """The model repeating itself is not the user incorporating anything."""
    agent_res = EntityResolver(mem.db, mem.embed, source="agent", agent_trust=0.5)
    atom_id = agent_res.resolve("Feria del Libro", "event", "t1", "s1", ["s1"])
    agent_res.resolve("Feria del Libro", "event", "t1", "s1", ["s1"])
    assert json.loads(_atom(mem.db, atom_id)["metadata"])["source"] == "agent"


def test_adoption_preserves_other_metadata(mem):
    """Adoption rewrites the source key only."""
    agent_res = EntityResolver(mem.db, mem.embed, source="agent", agent_trust=0.5)
    atom_id = agent_res.resolve("Feria del Libro", "event", "t1", "s1", ["s1"])
    mem.db.execute(
        """UPDATE atoms SET metadata = '{"source": "agent", "keep": "me"}' WHERE id = ?""",
        (atom_id,),
    )
    EntityResolver(mem.db, mem.embed, source="user").resolve(
        "Feria del Libro", "event", "t1", "s1", ["s1"]
    )
    meta = json.loads(_atom(mem.db, atom_id)["metadata"])
    assert meta == {"source": "user", "keep": "me"}


def test_adoption_survives_unparseable_metadata(mem):
    """Corrupt metadata must not abort the mention path."""
    agent_res = EntityResolver(mem.db, mem.embed, source="agent", agent_trust=0.5)
    atom_id = agent_res.resolve("Feria del Libro", "event", "t1", "s1", ["s1"])
    mem.db.execute("UPDATE atoms SET metadata = 'garbage' WHERE id = ?", (atom_id,))

    # Unreadable provenance already reads as user-authored, so there is nothing
    # to adopt; the call must simply not raise.
    EntityResolver(mem.db, mem.embed, source="user").resolve(
        "Feria del Libro", "event", "t1", "s1", ["s1"]
    )
    assert _atom(mem.db, atom_id) is not None
