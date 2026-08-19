"""Hardening tests for audited evolution/spaces fixes."""
from __future__ import annotations

import uuid

import pytest

from smrti import Smrti
from smrti.core.models import Atom, AtomType, EntityType, TruthValue
from smrti.evolution.connections import discover_connections
from smrti.evolution.healing import heal_orphaned_episodes
from smrti.evolution.truth import update_truth
from smrti.evolution.valence import propagate_valence
from smrti.spaces.emergence import materialize_bridge
from smrti.spaces.set_ops import space_overlap


@pytest.fixture
def mem(tmp_path):
    db_path = str(tmp_path / "hardening.db")
    return Smrti(db_path=db_path, personality="balanced", tenant_id="test", write_space="default")


def _add_atom(mem, label: str, *, type_: str = "concept", tenant: str | None = None,
              space: str | None = None, valence: float = 0.0, intensity: float = 0.0,
              confidence: float = 0.5, sti: float = 0.5, lti: float = 0.5) -> str:
    atom_id = str(uuid.uuid4())
    mem.db.execute(
        """INSERT INTO atoms (id, type, label, tenant_id, space, probability, confidence, sti, lti, valence, intensity)
           VALUES (?, ?, ?, ?, ?, 0.8, ?, ?, ?, ?, ?)""",
        (atom_id, type_, label, tenant or mem.tenant_id, space or mem.write_space,
         confidence, sti, lti, valence, intensity),
    )
    return atom_id


def _add_relation(mem, source_id: str, target_id: str, relation: str, *,
                  tenant: str | None = None, space: str | None = None) -> str:
    rel_id = str(uuid.uuid4())
    mem.db.execute(
        """INSERT INTO atoms (id, type, label, source_id, target_id, relation, tenant_id, space, probability, confidence, sti, lti, valence, intensity)
           VALUES (?, 'relation', ?, ?, ?, ?, ?, ?, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)""",
        (rel_id, relation, source_id, target_id, relation,
         tenant or mem.tenant_id, space or mem.write_space),
    )
    return rel_id


# ── truth.update_truth zero-weight guard ──────────────────────────────────────

def test_update_truth_zero_weight_preserves_prior():
    current = TruthValue(probability=0.9, confidence=0.0)
    updated = update_truth(current, 0.1, 0.0, 0.3)
    assert updated.probability == pytest.approx(0.9)
    assert updated.confidence == pytest.approx(0.0)


def test_update_truth_negative_weight_preserves_prior():
    current = TruthValue(probability=0.7, confidence=0.5)
    updated = update_truth(current, 0.0, -1.0, 0.3)
    assert updated.probability == pytest.approx(0.7)
    assert updated.confidence == pytest.approx(0.5)


# ── contradiction resolution fires once ───────────────────────────────────────

def test_contradiction_weakens_loser_once_then_stays_resolved(mem):
    id_a = mem.believe("the deployment will succeed", probability=0.9)
    id_b = mem.believe("the deployment will fail", probability=0.2)
    mem.atomspace.link_atoms(id_a, id_b, "contradicts", "test", "default")
    mem.db.execute(
        "UPDATE personality SET confidence_decay_rate = 0.0 WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )
    mem.db.execute("UPDATE atoms SET confidence = 0.9 WHERE id = ?", (id_a,))
    mem.db.execute("UPDATE atoms SET confidence = 0.5 WHERE id = ?", (id_b,))

    r1 = mem.reflect()
    assert r1.contradictions_resolved == 1
    conf1 = mem.db.fetchone("SELECT confidence FROM atoms WHERE id = ?", (id_b,))["confidence"]
    assert conf1 == pytest.approx(0.4)

    r2 = mem.reflect()
    assert r2.contradictions_resolved == 0
    conf2 = mem.db.fetchone("SELECT confidence FROM atoms WHERE id = ?", (id_b,))["confidence"]
    assert conf2 == pytest.approx(conf1)


# ── valence propagation nudges toward source ──────────────────────────────────

def test_valence_blend_moves_neighbor_toward_source_not_zero(mem):
    src = _add_atom(mem, "critical failure", valence=-0.5, intensity=0.9)
    nb = _add_atom(mem, "related module", valence=-0.9, intensity=0.8)
    _add_relation(mem, src, nb, "associated")

    propagate_valence(src, -0.5, 0.9, 0.5, mem.db, "test", "default", mood_inertia=0.8)

    row = mem.db.fetchone("SELECT valence FROM atoms WHERE id = ?", (nb,))
    # step = 0.5 * (1 - 0.8) = 0.1: -0.9 + 0.1 * (-0.5 - (-0.9)) = -0.86
    assert row["valence"] == pytest.approx(-0.86)
    # Fixpoint is the source valence, never zero
    assert -0.9 < row["valence"] < -0.5


def test_positive_valence_propagates_in_epoch(mem):
    src = _add_atom(mem, "joyful launch", valence=0.9, intensity=0.9)
    nb = _add_atom(mem, "team project", valence=0.0, intensity=0.0)
    _add_relation(mem, src, nb, "associated")

    mem.reflect()

    row = mem.db.fetchone("SELECT valence FROM atoms WHERE id = ?", (nb,))
    assert row["valence"] > 0.0


# ── LTI promotion cap and counting ────────────────────────────────────────────

def test_lti_promotion_counts_qualifying_rows_without_ratcheting(mem):
    """Promotion never raises LTI above what a saturated STI is worth (0.5).

    STI caps at 3.0, so scaling it unclamped pins LTI to its own ceiling on the
    first promotion and decay can never walk it back — the atom becomes
    permanently unprunable no matter how irrelevant it later turns out to be.
    """
    a = mem.remember("critical memory one")
    b = mem.remember("critical memory two")
    mem.db.execute("UPDATE atoms SET sti = 3.0, lti = 0.4 WHERE id IN (?, ?)", (a, b))
    mem.db.execute(
        "UPDATE personality SET sti_decay_rate = 0.0, lti_decay_rate = 0.0 "
        "WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )

    result = mem.reflect()
    # Both atoms satisfied the promotion criteria even though lti was already > 0
    assert result.lti_promoted == 2
    for atom_id in (a, b):
        row = mem.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (atom_id,))
        assert row["lti"] == pytest.approx(0.5)


def test_lti_promotion_never_lowers_existing_lti(mem):
    """An atom already above the promotion target keeps its higher LTI."""
    a = mem.remember("well established fact")
    mem.db.execute("UPDATE atoms SET sti = 3.0, lti = 0.9 WHERE id = ?", (a,))
    mem.db.execute(
        "UPDATE personality SET sti_decay_rate = 0.0, lti_decay_rate = 0.0 "
        "WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )

    mem.reflect()
    row = mem.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (a,))
    assert row["lti"] == pytest.approx(0.9)


def test_lti_decays_so_stale_atoms_become_prunable(mem):
    """LTI must fall over time, or the prune predicate is unsatisfiable.

    Promotion only ever raises LTI. Without a downward force, any atom that
    trends briefly salient sits above the prune floor forever and the graph
    grows without bound.
    """
    a = mem.remember("passing mention", type="concept")
    mem.db.execute("UPDATE atoms SET sti = 0.0, lti = 0.6 WHERE id = ?", (a,))
    mem.db.execute(
        "UPDATE personality SET lti_decay_rate = 0.1 WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )

    seen = []
    for _ in range(5):
        mem.reflect()
        row = mem.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (a,))
        if row is None:
            break
        seen.append(row["lti"])

    assert seen == sorted(seen, reverse=True), "LTI must be monotonically decreasing"
    assert seen[-1] < 0.6


def test_lti_decay_does_not_erode_the_critical_valence_floor(mem):
    """Severe negative-valence atoms keep the LTI floor that shields them.

    atomspace.add_atom pins these at lti >= 0.5 so past failures stay out of
    reach of the pruner; if decay eats that floor the error-avoidance
    guarantee simply expires on a timer.
    """
    critical = mem.remember("dropped the production database", valence=-0.95)
    mem.db.execute(
        "UPDATE personality SET lti_decay_rate = 0.3 WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )

    for _ in range(20):
        mem.reflect()

    row = mem.db.fetchone("SELECT lti, confidence FROM atoms WHERE id = ?", (critical,))
    assert row is not None, "critical memory must survive"
    assert row["lti"] >= 0.5
    # The floor protects LTI only — confidence still decays normally.
    assert row["confidence"] < 0.5


# ── user-testimony confidence floor ───────────────────────────────────────────

def _surface_floor(mem) -> float:
    return mem.db.fetchone(
        "SELECT min_confidence_to_surface FROM personality WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )["min_confidence_to_surface"]


def test_user_memory_confidence_decays_to_the_surfacing_floor_not_zero(mem):
    """What the user stated stays recallable forever.

    Recall filters out atoms below min_confidence_to_surface, and a memory
    that cannot surface can never be restated and re-evidenced — decay to
    zero is a one-way trip to invisibility for exactly the facts that go
    unmentioned longest (family, identity). User episodes and beliefs
    therefore hold at the surfacing line.
    """
    episode = mem.remember("Nicolás lives with Roxana and Esmeralda")
    belief = mem.believe("Lourdes is Nicolás's daughter", probability=0.95)
    mem.db.execute(
        "UPDATE personality SET confidence_decay_rate = 0.5 WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )

    for _ in range(20):
        mem.reflect()

    floor = _surface_floor(mem)
    for atom_id in (episode, belief):
        row = mem.db.fetchone("SELECT confidence FROM atoms WHERE id = ?", (atom_id,))
        assert row["confidence"] == pytest.approx(floor)


def test_agent_memory_confidence_still_decays_to_nothing(mem):
    """Model-volunteered content the user never adopted keeps fading out."""
    a = mem.remember("the model guessed the user likes jazz", metadata={"source": "agent"})
    mem.db.execute(
        "UPDATE personality SET confidence_decay_rate = 0.5 WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )

    for _ in range(20):
        mem.reflect()

    row = mem.db.fetchone("SELECT confidence FROM atoms WHERE id = ?", (a,))
    assert row is None or row["confidence"] < _surface_floor(mem)


def test_forgotten_user_memory_is_not_lifted_back_to_the_floor(mem):
    """The floor only holds an atom still at or above it.

    forget() sinks a memory to 0.3× its confidence on purpose; a floor that
    reached down would undo every deliberate forget one epoch later.
    """
    a = mem.remember("outdated fact the user corrected")
    floor = _surface_floor(mem)
    mem.db.execute("UPDATE atoms SET confidence = ? WHERE id = ?", (floor * 0.3, a))

    mem.reflect()

    row = mem.db.fetchone("SELECT confidence FROM atoms WHERE id = ?", (a,))
    assert row["confidence"] < floor * 0.3


def test_concept_confidence_is_not_floored(mem):
    """Concepts are derived index nodes, not testimony — they decay freely,
    which is what keeps them prunable once nothing references them."""
    c = _add_atom(mem, "familia", type_="concept", confidence=0.5, sti=0.0, lti=0.0)
    mem.db.execute(
        "UPDATE personality SET confidence_decay_rate = 0.5 WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )

    for _ in range(20):
        mem.reflect()

    row = mem.db.fetchone("SELECT confidence FROM atoms WHERE id = ?", (c,))
    assert row is None or row["confidence"] < _surface_floor(mem)


# ── prune cascade tenant scoping ──────────────────────────────────────────────

def test_prune_relation_cascade_is_tenant_scoped_but_cross_space(mem):
    dead = _add_atom(mem, "prunable junk", confidence=0.01, sti=0.0, lti=0.0)
    anchor_same = _add_atom(mem, "bridge anchor", space="side_space")
    rel_same = _add_relation(mem, dead, anchor_same, "associated", space="side_space")
    # Relations in another tenant must never be touched by the cascade
    other_a = _add_atom(mem, "other tenant a", tenant="other")
    other_b = _add_atom(mem, "other tenant b", tenant="other")
    rel_other = _add_relation(mem, other_a, other_b, "associated", tenant="other")

    result = mem.reflect()
    assert result.atoms_pruned >= 1
    assert mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (dead,)) is None
    # Same-tenant cross-space relation (e.g. bridge edge) is cleaned
    assert mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (rel_same,)) is None
    # Another tenant's graph is untouched
    assert mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (rel_other,)) is not None


# ── set_ops absent-signal redistribution ──────────────────────────────────────

def test_identical_untyped_atoms_match_with_embed_engine(mem):
    for space in ("left", "right"):
        mem.atomspace.add_atom(Atom(
            type=AtomType.CONCEPT, label="quantum tunneling",
            content="quantum tunneling", tenant_id="test", space=space,
        ))

    overlap = space_overlap(
        "test", "left", "right", mem.db, threshold=0.85, embed_engine=mem.embed,
    )
    assert len(overlap.pairs) == 1
    assert overlap.pairs[0].similarity == pytest.approx(1.0, abs=1e-3)


# ── bridge discovery is direction-agnostic ────────────────────────────────────

def test_bridge_discovery_bidirectional_no_duplicates(mem):
    for space in ("s1", "s2"):
        mem.atomspace.add_atom(Atom(
            type=AtomType.CONCEPT, label="systems programming in Rust",
            content="systems programming in Rust", tenant_id="test", space=space,
        ))

    overlap_ab = space_overlap("test", "s1", "s2", mem.db, threshold=0.85, embed_engine=mem.embed)
    n1 = materialize_bridge(overlap_ab, "test", mem.db, mem.embed, mem.atomspace, min_jaccard=0.0)
    assert n1 == 1

    overlap_ba = space_overlap("test", "s2", "s1", mem.db, threshold=0.85, embed_engine=mem.embed)
    n2 = materialize_bridge(overlap_ba, "test", mem.db, mem.embed, mem.atomspace, min_jaccard=0.0)
    assert n2 == 1

    bridge_atoms = mem.db.fetchall(
        "SELECT id FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'",
        ("test", overlap_ab.bridge_space_name),
    )
    assert len(bridge_atoms) == 1
    bridge_edges = mem.db.fetchall(
        "SELECT id FROM atoms WHERE tenant_id = ? AND space = ? AND type = 'relation' AND relation = 'bridge'",
        ("test", overlap_ab.bridge_space_name),
    )
    assert len(bridge_edges) == 2


# ── healing multi-person attribution ──────────────────────────────────────────

def test_healing_multi_person_attributes_by_embedding_similarity(mem):
    alice = mem.atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Alice",
        content="Alice loves alpine hiking and mountain trails",
        entity_type=EntityType.PERSON, tenant_id="test", space="default",
    ))
    bob = mem.atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Bob",
        content="Bob bakes croissants and pastries at the bakery",
        entity_type=EntityType.PERSON, tenant_id="test", space="default",
    ))
    # Bob is the most salient — old behavior would attribute everything to him
    mem.db.execute("UPDATE atoms SET sti = 2.0, lti = 0.9 WHERE id = ?", (bob,))

    episode = mem.atomspace.add_atom(Atom(
        type=AtomType.EPISODE, label="Alice",
        content="Alice loves alpine hiking and mountain trails",
        tenant_id="test", space="default",
    ))
    gear = mem.atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="hiking gear",
        content="hiking gear", tenant_id="test", space="default",
    ))
    mem.atomspace.link_atoms(episode, gear, "mentions", "test", "default")

    healed = heal_orphaned_episodes("test", "default", mem.db)
    assert healed == 1

    to_alice = mem.db.fetchone(
        """SELECT id FROM atoms WHERE type = 'relation' AND relation = 'mentions'
           AND source_id = ? AND target_id = ?""",
        (episode, alice),
    )
    assert to_alice is not None
    to_bob = mem.db.fetchone(
        """SELECT id FROM atoms WHERE type = 'relation' AND relation = 'mentions'
           AND source_id = ? AND target_id = ?""",
        (episode, bob),
    )
    assert to_bob is None


def test_healing_multi_person_skips_episode_without_embedding(mem):
    for name in ("Alice", "Bob"):
        mem.atomspace.add_atom(Atom(
            type=AtomType.CONCEPT, label=name, content=f"{name} profile",
            entity_type=EntityType.PERSON, tenant_id="test", space="default",
        ))
    episode = _add_atom(mem, "unattributable note", type_="episode")
    concept = _add_atom(mem, "loose concept")
    _add_relation(mem, episode, concept, "mentions")

    healed = heal_orphaned_episodes("test", "default", mem.db)
    assert healed == 0
    linked = mem.db.fetchone(
        """SELECT r.id FROM atoms r JOIN atoms p ON p.id = r.target_id
           WHERE r.type = 'relation' AND r.relation = 'mentions'
             AND r.source_id = ? AND p.entity_type = 'person'""",
        (episode,),
    )
    assert linked is None


# ── connections stay within the space ─────────────────────────────────────────

def test_connections_do_not_cross_spaces(mem):
    a = mem.atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="graph neural networks",
        content="graph neural networks research", tenant_id="test", space="default",
    ))
    b = mem.atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="graph neural networks",
        content="graph neural networks research", tenant_id="test", space="elsewhere",
    ))
    mem.db.execute("UPDATE atoms SET lti = 0.9 WHERE id IN (?, ?)", (a, b))

    count = discover_connections("test", "default", mem.db, mem.embed)
    assert count == 0
    rel = mem.db.fetchone(
        """SELECT id FROM atoms WHERE type = 'relation'
           AND ((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))""",
        (a, b, b, a),
    )
    assert rel is None
