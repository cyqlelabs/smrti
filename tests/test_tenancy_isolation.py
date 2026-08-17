"""Multi-tenancy and multi-space isolation guarantees.

The README states the contract these tests pin down:

    Tenants are hard walls: atoms, embeddings, and attention weights never
    cross them. Spaces are permeable layers within a tenant: you write to one,
    read from many, and each has its own personality and consolidation cycle.

Two invariants follow, and every test here is an instance of one of them:

  * **Tenant walls.** No read, write, epoch, set operation, or entity
    resolution in one tenant may observe or modify another tenant's rows.
  * **Overlay reads are read-only.** A space may *read* its whole overlay, but
    only the write space may be *mutated* — attention, truth, provenance, and
    deletions all stop at the write-space boundary.
"""
from __future__ import annotations

import os
import tempfile
import uuid

import pytest

from smrti import Smrti
from smrti.core.atomspace import AtomSpace
from smrti.core.db import Database
from smrti.core.embed import EmbeddingProvider
from smrti.core.models import (
    Atom,
    AtomType,
    EntityType,
    Evidence,
    TruthValue,
)
from smrti.evolution.epoch import run_epoch
from smrti.extraction.resolve import EntityResolver
from smrti.retrieval.fan_out import retrieve
from smrti.spaces.set_ops import (
    space_difference,
    space_intersection,
    space_overlap,
    space_symmetric_difference,
    space_union,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "memory.db")


@pytest.fixture
def db(db_path):
    database = Database(db_path)
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def embed():
    return EmbeddingProvider()


@pytest.fixture
def atomspace(db, embed):
    return AtomSpace(db, embed)


def _concept(label, tenant, space, **kw):
    kw.setdefault("truth", TruthValue(probability=0.8, confidence=0.8))
    return Atom(type=AtomType.CONCEPT, label=label, tenant_id=tenant, space=space, **kw)


def _belief(label, tenant, space, probability=0.5, confidence=0.5, **kw):
    return Atom(
        type=AtomType.BELIEF,
        label=label,
        tenant_id=tenant,
        space=space,
        truth=TruthValue(probability=probability, confidence=confidence),
        **kw,
    )


def _personality(db, tenant, space, **overrides):
    """Insert a personality row directly, bypassing the Smrti facade."""
    columns = ", ".join(["tenant_id", "space", *overrides])
    marks = ", ".join("?" * (2 + len(overrides)))
    db.execute(
        f"INSERT INTO personality ({columns}) VALUES ({marks})",
        (tenant, space, *overrides.values()),
    )


def _row(db, atom_id, *fields):
    row = db.fetchone(
        f"SELECT {', '.join(fields)} FROM atoms WHERE id = ?", (atom_id,)
    )
    return dict(row) if row else None


# ══════════════════════════════════════════════════════════════════════════════
# Tenant walls
# ══════════════════════════════════════════════════════════════════════════════

def test_recall_never_returns_another_tenants_atoms(atomspace, db, embed):
    """Identical content under the same space name in two tenants stays apart."""
    text = "the deploy key rotates every ninety days"
    atomspace.add_atom(_concept(text, "acme", "default"))
    atomspace.add_atom(_concept(text, "globex", "default"))

    acme = retrieve(text, "acme", ["default"], db, embed, write_space="default")
    globex = retrieve(text, "globex", ["default"], db, embed, write_space="default")

    assert len(acme) == 1 and len(globex) == 1
    assert acme[0].atom.tenant_id == "acme"
    assert globex[0].atom.tenant_id == "globex"
    assert acme[0].atom.id != globex[0].atom.id


def test_knn_index_is_partitioned_by_tenant(atomspace, db, embed):
    """The vec0 partition key must keep a crowded tenant out of another's KNN.

    Without partitioning the 40 near-identical vectors below would fill the
    candidate budget and the lone atom in the quiet tenant would never surface.
    """
    query = "quarterly revenue forecast spreadsheet"
    for i in range(40):
        atomspace.add_atom(_concept(f"{query} {i}", "loud", "default"))
    atomspace.add_atom(_concept(query, "quiet", "default"))

    results = retrieve(query, "quiet", ["default"], db, embed, write_space="default")
    assert [r.atom.tenant_id for r in results] == ["quiet"]


def test_epoch_leaves_other_tenants_untouched(atomspace, db, embed):
    """Decay, promotion, and pruning are all partitioned by tenant."""
    mine = _concept("shared label", "mine", "s", attention={"sti": 1.0, "lti": 0.4})
    theirs = _concept("shared label", "theirs", "s", attention={"sti": 1.0, "lti": 0.4})
    atomspace.add_atom(mine)
    atomspace.add_atom(theirs)
    _personality(db, "mine", "s")

    before = _row(db, theirs.id, "sti", "lti", "confidence", "updated_at")
    result = run_epoch("mine", "s", db, embed)

    assert result.atoms_decayed == 1  # only this tenant's atom was counted
    assert _row(db, theirs.id, "sti", "lti", "confidence", "updated_at") == before
    assert _row(db, mine.id, "sti")["sti"] < 1.0


def test_epoch_prune_cannot_delete_another_tenants_atoms(atomspace, db, embed):
    """A prunable atom in another tenant survives an epoch run here."""
    doomed = _concept("ephemeral", "mine", "s", truth=TruthValue(probability=0.5, confidence=0.0))
    survivor = _concept("ephemeral", "theirs", "s", truth=TruthValue(probability=0.5, confidence=0.0))
    atomspace.add_atom(doomed)
    atomspace.add_atom(survivor)
    _personality(db, "mine", "s")

    run_epoch("mine", "s", db, embed)

    assert _row(db, doomed.id, "id") is None
    assert _row(db, survivor.id, "id") is not None


def test_entity_resolution_never_crosses_tenants(db, embed):
    """An identically named entity in another tenant is not a match."""
    theirs = EntityResolver(db, embed).resolve(
        "Kubernetes", "technology", "theirs", "s", ["s"]
    )
    mine = EntityResolver(db, embed).resolve(
        "Kubernetes", "technology", "mine", "s", ["s"]
    )

    assert mine != theirs
    assert _row(db, mine, "tenant_id")["tenant_id"] == "mine"
    assert _row(db, theirs, "tenant_id")["tenant_id"] == "theirs"


def test_alias_table_is_tenant_scoped(db, embed):
    """An alias registered by one tenant must not resolve for another."""
    resolver = EntityResolver(db, embed)
    owner = resolver.resolve("Margarethe Vollenweider", "person", "mine", "s", ["s"])
    resolver.aliases.add(owner, "Maggie", "mine", "s")

    assert resolver.aliases.lookup("Maggie", "mine", ["s"]) == owner
    assert resolver.aliases.lookup("Maggie", "theirs", ["s"]) is None


def test_add_atom_refuses_to_move_an_atom_across_tenants(atomspace, db):
    """Reusing an ID under a different tenant must fail loudly, not overwrite.

    ``INSERT OR REPLACE`` keys on the primary key alone, so without this guard
    the row — and its vector — would silently relocate into the other tenant.
    """
    original = _concept("payroll figures", "acme", "hr")
    atomspace.add_atom(original)

    hijack = Atom(
        id=original.id, type=AtomType.CONCEPT, label="pwned",
        tenant_id="globex", space="hr",
    )
    with pytest.raises(ValueError, match="refusing to move"):
        atomspace.add_atom(hijack)

    stored = _row(db, original.id, "tenant_id", "label")
    assert stored == {"tenant_id": "acme", "label": "payroll figures"}
    vec = db.fetchall("SELECT tenant_id FROM vec_atoms WHERE atom_id = ?", (original.id,))
    assert [r["tenant_id"] for r in vec] == ["acme"]


def test_add_atom_refuses_to_move_an_atom_across_spaces(atomspace):
    original = _concept("release checklist", "acme", "a")
    atomspace.add_atom(original)

    moved = Atom(id=original.id, type=AtomType.CONCEPT, label="release checklist",
                 tenant_id="acme", space="b")
    with pytest.raises(ValueError, match="refusing to move"):
        atomspace.add_atom(moved)


def test_add_atom_still_updates_in_place_within_the_same_partition(atomspace, db):
    """The guard must not break the ordinary re-add/update path."""
    atom = _concept("draft", "acme", "a")
    atomspace.add_atom(atom)
    atom.label = "final"
    atomspace.add_atom(atom)

    assert _row(db, atom.id, "label")["label"] == "final"
    assert db.fetchone(
        "SELECT COUNT(*) n FROM vec_atoms WHERE atom_id = ?", (atom.id,)
    )["n"] == 1


def test_boost_sti_is_a_no_op_outside_the_given_partition(atomspace, db):
    """The optional tenant/space arguments confine the write to one partition.

    A caller that resolved an ID out of an overlay space holds a perfectly
    valid ID for an atom it must not touch; passing the partition it is
    allowed to write turns that into a no-op instead of a silent reach-in.
    """
    mine = _concept("shared label", "t", "mine")
    theirs = _concept("shared label", "t", "other")
    atomspace.add_atom(mine)
    atomspace.add_atom(theirs)

    atomspace.boost_sti(mine.id, 0.5, tenant_id="t", space="mine")
    atomspace.boost_sti(theirs.id, 0.5, tenant_id="t", space="mine")

    assert _row(db, mine.id, "sti")["sti"] == pytest.approx(0.5)
    assert _row(db, theirs.id, "sti")["sti"] == 0.0


def test_boost_sti_without_a_partition_stays_unscoped(atomspace, db):
    """Omitting the arguments keeps the pre-existing behaviour for old callers."""
    atom = _concept("note", "t", "s")
    atomspace.add_atom(atom)

    atomspace.boost_sti(atom.id, 1.0)

    assert _row(db, atom.id, "sti")["sti"] == pytest.approx(1.0)


def test_epoch_skips_a_contradiction_edge_with_a_missing_endpoint(atomspace, db, embed):
    """A dangling contradiction edge is skipped, not dereferenced.

    The pruner deletes relations alongside their endpoints, but a row written
    directly — or left by an older release — can still carry a NULL endpoint.
    """
    src = _belief("mine", "t", "mine", confidence=0.9)
    atomspace.add_atom(src)
    db.execute(
        """INSERT INTO atoms (id, type, label, source_id, target_id, relation, tenant_id, space)
           VALUES (?, 'relation', 'contradicts', ?, NULL, 'contradicts', 't', 'mine')""",
        (str(uuid.uuid4()), src.id),
    )
    _personality(db, "t", "mine", confidence_decay_rate=0.0)

    result = run_epoch("t", "mine", db, embed)

    assert result.contradictions_resolved == 0
    assert _row(db, src.id, "confidence")["confidence"] == pytest.approx(0.9)


def test_a_bridge_space_does_not_initiate_further_bridging(atomspace, db, embed):
    """Bridging out of a bridge space would build meta-bridges without end."""
    from smrti.evolution.epoch import _discover_bridges

    text = "Postgres is the database we run in production"
    atomspace.add_atom(_concept(text, "t", "a"))
    atomspace.add_atom(_concept(text, "t", "a_x_b"))

    assert _discover_bridges("t", "a_x_b", db, embed) == 0


def test_bridge_discovery_skips_spaces_that_are_already_bridges(atomspace, db, embed):
    """A parent space bridges its siblings, never their bridges."""
    from smrti.evolution.epoch import _discover_bridges

    text = "Postgres is the database we run in production"
    atomspace.add_atom(_concept(text, "t", "a"))
    atomspace.add_atom(_concept(text, "t", "a_x_b"))

    assert _discover_bridges("t", "a", db, embed) == 0
    spaces = db.fetchall("SELECT DISTINCT space FROM atoms WHERE tenant_id = 't' ORDER BY space")
    assert [r["space"] for r in spaces] == ["a", "a_x_b"]


def test_clear_space_leaves_other_tenants_intact(db_path, embed):
    mine = Smrti(db_path=db_path, tenant_id="mine", write_space="s")
    theirs = Smrti(db_path=db_path, tenant_id="theirs", write_space="s")
    mine.remember("a note that belongs to me")
    theirs.remember("a note that belongs to me")

    assert mine.clear_space() == 1
    assert mine.status()["total_atoms"] == 0
    assert theirs.status()["total_atoms"] == 1


def test_list_spaces_is_tenant_scoped(db_path):
    Smrti(db_path=db_path, tenant_id="mine", write_space="alpha").remember("alpha note")
    Smrti(db_path=db_path, tenant_id="mine", write_space="beta").remember("beta note")
    Smrti(db_path=db_path, tenant_id="theirs", write_space="gamma").remember("gamma note")

    mine = Smrti(db_path=db_path, tenant_id="mine", write_space="alpha")
    assert mine.list_spaces() == ["alpha", "beta"]


def test_personality_is_per_tenant_and_space(db_path):
    a = Smrti(db_path=db_path, tenant_id="t", write_space="a", personality="analytical")
    b = Smrti(db_path=db_path, tenant_id="t", write_space="b", personality="curious")
    other_tenant = Smrti(db_path=db_path, tenant_id="u", write_space="a", personality="maverick")

    assert a.status()["personality"]["preset_name"] == "analytical"
    assert b.status()["personality"]["preset_name"] == "curious"
    assert other_tenant.status()["personality"]["preset_name"] == "maverick"

    a.set_personality("empathetic")
    assert a.status()["personality"]["preset_name"] == "empathetic"
    assert b.status()["personality"]["preset_name"] == "curious"
    assert other_tenant.status()["personality"]["preset_name"] == "maverick"


def test_set_operations_are_tenant_scoped(atomspace, db, embed):
    """Same space names, same labels, different tenants — no overlap."""
    for tenant in ("mine", "theirs"):
        atomspace.add_atom(_concept("Redis cluster failover", tenant, "a"))
        atomspace.add_atom(_concept("Redis cluster failover", tenant, "b"))

    # Within a tenant the two spaces overlap...
    assert space_overlap("mine", "a", "b", db, 0.8, embed).jaccard > 0
    # ...but each tenant only ever sees its own two atoms.
    overlap = space_overlap("mine", "a", "b", db, 0.8, embed)
    assert all(
        p.atom_a.tenant_id == "mine" and p.atom_b.tenant_id == "mine"
        for p in overlap.pairs
    )


def test_bridge_materialization_is_tenant_scoped(db_path):
    mine = Smrti(db_path=db_path, tenant_id="mine", write_space="a")
    Smrti(db_path=db_path, tenant_id="mine", write_space="b").remember(
        "Redis is the cache we run in production"
    )
    Smrti(db_path=db_path, tenant_id="theirs", write_space="b").remember(
        "Redis is the cache we run in production"
    )
    mine.remember("Redis is the cache we run in production")

    assert mine.materialize_bridge("b", threshold=0.8, min_jaccard=0.05) >= 1

    bridged = mine.db.fetchall(
        "SELECT DISTINCT tenant_id FROM atoms WHERE space = 'a_x_b'"
    )
    assert [r["tenant_id"] for r in bridged] == ["mine"]


# ══════════════════════════════════════════════════════════════════════════════
# Overlay reads are read-only
# ══════════════════════════════════════════════════════════════════════════════

def test_overlay_recall_reads_across_spaces(db_path):
    Smrti(db_path=db_path, tenant_id="t", write_space="shared").remember(
        "the staging cluster lives in eu-west-1"
    )
    private = Smrti(
        db_path=db_path, tenant_id="t", write_space="private",
        read_spaces=["private", "shared"],
    )
    private.remember("my own scratch note about clusters")

    spaces = {r.atom.space for r in private.recall("staging cluster region")}
    assert "shared" in spaces


def test_read_spaces_are_copied_not_aliased(db_path):
    """Mutating the list you passed in must not change the instance's overlay."""
    spaces = ["private", "shared"]
    mem = Smrti(db_path=db_path, tenant_id="t", write_space="private", read_spaces=spaces)
    spaces.append("someone-elses")
    assert mem.read_spaces == ["private", "shared"]


def test_repeated_read_spaces_are_probed_once(atomspace, db, embed):
    """A duplicated space name is redundant work, not extra candidates.

    read_spaces reaches the proxy straight from a request header, so a caller
    can otherwise multiply the KNN probe count by repeating one name.
    """
    atomspace.add_atom(_concept("the load balancer drains connections", "t", "s"))
    probes = []
    real_fetchall = db.fetchall

    def counting_fetchall(sql, params=()):
        if "embedding MATCH" in sql:
            probes.append(sql)
        return real_fetchall(sql, params)

    db.fetchall = counting_fetchall
    try:
        results = retrieve(
            "load balancer", "t", ["s", "s", "s", "s"], db, embed, write_space="s"
        )
    finally:
        db.fetchall = real_fetchall

    assert len(probes) == 1
    assert len(results) == 1  # and no duplicated candidates


def test_recall_does_not_boost_sti_in_a_foreign_read_space(atomspace, db, embed):
    """Reading an overlay space must not drive that space's attention weights."""
    foreign = _concept("incident postmortem for the June outage", "t", "shared")
    atomspace.add_atom(foreign)
    mine = _concept("incident postmortem for the June outage", "t", "private")
    atomspace.add_atom(mine)
    _personality(db, "t", "private", sti_boost_on_access=0.5)

    results = retrieve(
        "June outage postmortem", "t", ["private", "shared"], db, embed,
        write_space="private",
    )
    assert {r.atom.space for r in results} == {"private", "shared"}
    assert _row(db, foreign.id, "sti")["sti"] == 0.0
    assert _row(db, mine.id, "sti")["sti"] > 0.0


def test_resolver_does_not_reinforce_an_atom_in_a_foreign_read_space(atomspace, db, embed):
    """An overlay hit is returned, but the foreign atom is left exactly as found.

    Boosting it would let one agent's mentions steer another space's attention,
    and the evidence row would let this space's epoch rewrite that space's
    truth values on the next consolidation pass.
    """
    foreign = _concept("Kubernetes", "t", "shared", entity_type=EntityType.TECHNOLOGY)
    atomspace.add_atom(foreign)
    before = _row(db, foreign.id, "sti", "probability", "confidence")

    resolved = EntityResolver(db, embed).resolve(
        "Kubernetes", "technology", "t", "private", ["private", "shared"]
    )

    assert resolved == foreign.id  # the overlay read still works
    assert _row(db, foreign.id, "sti", "probability", "confidence") == before
    assert db.fetchall("SELECT id FROM evidence") == []


def test_resolver_still_reinforces_an_atom_in_the_write_space(atomspace, db, embed):
    """The read-only rule applies to foreign spaces only, not to your own."""
    mine = _concept("Kubernetes", "t", "private", entity_type=EntityType.TECHNOLOGY)
    atomspace.add_atom(mine)

    resolved = EntityResolver(db, embed).resolve(
        "Kubernetes", "technology", "t", "private", ["private", "shared"]
    )

    assert resolved == mine.id
    assert _row(db, mine.id, "sti")["sti"] > 0.0
    evidence = db.fetchall("SELECT atom_id, space FROM evidence")
    assert [(e["atom_id"], e["space"]) for e in evidence] == [(mine.id, "private")]


def test_epoch_does_not_apply_evidence_to_an_atom_in_another_space(atomspace, db, embed):
    """Evidence filed here against an atom elsewhere must not rewrite it.

    Legacy databases carry such rows from before entity resolution was scoped;
    consolidating this space must leave the foreign atom's truth value alone.
    """
    foreign = _belief("their belief", "t", "other", probability=0.5, confidence=0.5)
    atomspace.add_atom(foreign)
    _personality(db, "t", "mine")
    atomspace.add_evidence(
        Evidence(atom_id=foreign.id, observed_probability=1.0, tenant_id="t", space="mine")
    )

    before = _row(db, foreign.id, "probability", "confidence")
    result = run_epoch("t", "mine", db, embed)

    assert result.beliefs_updated == 0
    assert _row(db, foreign.id, "probability", "confidence") == before


def test_epoch_retires_evidence_it_cannot_apply(atomspace, db, embed):
    """Unappliable evidence is marked processed so it is not re-scanned forever."""
    foreign = _belief("their belief", "t", "other")
    atomspace.add_atom(foreign)
    _personality(db, "t", "mine")
    atomspace.add_evidence(
        Evidence(atom_id=foreign.id, observed_probability=1.0, tenant_id="t", space="mine")
    )

    run_epoch("t", "mine", db, embed)

    pending = db.fetchall("SELECT id FROM evidence WHERE processed = 0")
    assert pending == []


def test_epoch_contradiction_does_not_weaken_a_belief_in_another_space(atomspace, db, embed):
    """A contradiction edge whose target lives elsewhere adjudicates nothing.

    Bridge edges point out of their own space by construction, so an unscoped
    endpoint lookup lets one space downgrade another's beliefs.
    """
    src = _belief("mine", "t", "mine", confidence=0.9)
    tgt = _belief("theirs", "t", "other", confidence=0.2)
    atomspace.add_atom(src)
    atomspace.add_atom(tgt)
    atomspace.link_atoms(src.id, tgt.id, "contradicts", "t", "mine")
    _personality(db, "t", "mine")

    before = _row(db, tgt.id, "confidence")
    result = run_epoch("t", "mine", db, embed)

    assert result.contradictions_resolved == 0
    assert _row(db, tgt.id, "confidence") == before


def test_epoch_contradiction_still_resolves_within_one_space(atomspace, db, embed):
    strong = _belief("strong", "t", "mine", confidence=0.9)
    weak = _belief("weak", "t", "mine", confidence=0.2)
    atomspace.add_atom(strong)
    atomspace.add_atom(weak)
    atomspace.link_atoms(strong.id, weak.id, "contradicts", "t", "mine")
    _personality(db, "t", "mine", confidence_decay_rate=0.0)

    result = run_epoch("t", "mine", db, embed)

    assert result.contradictions_resolved == 1
    assert _row(db, weak.id, "confidence")["confidence"] < 0.2
    assert _row(db, strong.id, "confidence")["confidence"] == pytest.approx(0.9)


def test_forget_only_softens_atoms_in_the_write_space(db_path):
    Smrti(db_path=db_path, tenant_id="t", write_space="shared").remember(
        "never force-push to the release branch"
    )
    private = Smrti(
        db_path=db_path, tenant_id="t", write_space="private",
        read_spaces=["private", "shared"],
    )
    private.remember("never force-push to the release branch")

    softened = private.forget("force-push release branch", top_k=10)

    assert softened  # something in the write space was softened
    shared_conf = private.db.fetchone(
        "SELECT confidence FROM atoms WHERE tenant_id = 't' AND space = 'shared'"
    )["confidence"]
    assert shared_conf == pytest.approx(0.5)


def test_epoch_in_one_space_does_not_decay_a_sibling_space(atomspace, db, embed):
    mine = _concept("note", "t", "mine", attention={"sti": 1.0, "lti": 0.4})
    sibling = _concept("note", "t", "sibling", attention={"sti": 1.0, "lti": 0.4})
    atomspace.add_atom(mine)
    atomspace.add_atom(sibling)
    _personality(db, "t", "mine")

    before = _row(db, sibling.id, "sti", "lti", "confidence")
    run_epoch("t", "mine", db, embed)

    assert _row(db, sibling.id, "sti", "lti", "confidence") == before
    assert _row(db, mine.id, "sti")["sti"] < 1.0


def test_sti_propagation_across_a_bridge_edge_conserves_activation(atomspace, db, embed):
    """A bridge edge leaves the space, so no activation may leave with it.

    The per-neighbor credit is partitioned by space and silently no-ops on the
    far side; deducting the source's share anyway would delete STI outright.
    """
    bridge = _concept("Redis", "t", "a_x_b", attention={"sti": 1.0})
    parent = _concept("Redis", "t", "a")
    sibling = _concept("Redis cache", "t", "a_x_b")
    for atom in (bridge, parent, sibling):
        atomspace.add_atom(atom)
    atomspace.link_atoms(bridge.id, parent.id, "bridge", "t", "a_x_b")
    atomspace.link_atoms(bridge.id, sibling.id, "mentions", "t", "a_x_b")

    from smrti.evolution.attention import propagate_sti
    propagate_sti(bridge.id, boost=1.0, propagation_factor=0.4, db=db,
                  tenant_id="t", space="a_x_b")

    given_up = 1.0 - _row(db, bridge.id, "sti")["sti"]
    gained = _row(db, sibling.id, "sti")["sti"] + _row(db, parent.id, "sti")["sti"]
    assert _row(db, parent.id, "sti")["sti"] == 0.0  # the far side is untouched
    assert given_up == pytest.approx(gained)


def test_clear_space_cascades_bridge_edges_from_other_spaces(db_path):
    """Clearing a bridged space must not trip the foreign-key constraint.

    A bridge space holds relation atoms whose ``target_id`` points into both
    parents. Those columns are enforced foreign keys, so deleting a parent's
    atoms without first removing the referencing edges aborts the whole delete
    and leaves the space half-cleared.
    """
    a = Smrti(db_path=db_path, tenant_id="t", write_space="a")
    b = Smrti(db_path=db_path, tenant_id="t", write_space="b")
    text = "Postgres is the database we run in production"
    a.remember(text)
    b.remember(text)
    assert a.materialize_bridge("b", threshold=0.8, min_jaccard=0.05) >= 1

    assert a.clear_space() == 1

    assert a.status()["total_atoms"] == 0
    assert b.status()["total_atoms"] == 1
    dangling = a.db.fetchall(
        """SELECT r.id FROM atoms r
           WHERE r.type = 'relation'
             AND r.target_id IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM atoms t WHERE t.id = r.target_id)"""
    )
    assert dangling == []


def test_clear_space_removes_vectors_and_evidence(db_path, embed):
    mem = Smrti(db_path=db_path, tenant_id="t", write_space="s")
    atom_id = mem.believe("the cache is warm", probability=0.7, evidence="observed")
    assert mem.db.fetchone("SELECT COUNT(*) n FROM evidence")["n"] == 1

    mem.clear_space()

    assert mem.db.fetchone("SELECT COUNT(*) n FROM evidence")["n"] == 0
    assert mem.db.fetchone(
        "SELECT COUNT(*) n FROM vec_atoms WHERE atom_id = ?", (atom_id,)
    )["n"] == 0


def test_clear_space_handles_more_atoms_than_one_chunk(db_path, monkeypatch):
    """The chunked delete must cover every atom, not just the first batch."""
    mem = Smrti(db_path=db_path, tenant_id="t", write_space="s")
    monkeypatch.setattr(Smrti, "_CLEAR_CHUNK", 3)
    for i in range(10):
        mem.remember(f"note number {i}")

    assert mem.clear_space() == 10
    assert mem.status()["total_atoms"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# Set-theory laws
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def laws(atomspace, db, embed):
    """Two spaces sharing one label, each with one label of its own."""
    shared = "Postgres connection pooling"
    atomspace.add_atom(_concept(shared, "t", "a"))
    atomspace.add_atom(_concept(shared, "t", "b"))
    atomspace.add_atom(_concept("sourdough starter feeding schedule", "t", "a"))
    atomspace.add_atom(_concept("tide tables for the north sea", "t", "b"))
    return db, embed


def _labels(result):
    return sorted(a.label for a in result.atoms)


def test_intersection_and_difference_partition_the_left_space(laws):
    db, embed = laws
    inter = space_intersection("t", "a", "b", db, 0.85, embed)
    diff = space_difference("t", "a", "b", db, 0.85, embed)

    assert _labels(inter) == ["Postgres connection pooling"]
    assert _labels(diff) == ["sourdough starter feeding schedule"]
    # Every atom of A is in exactly one of the two.
    assert sorted(_labels(inter) + _labels(diff)) == [
        "Postgres connection pooling", "sourdough starter feeding schedule",
    ]


def test_difference_is_asymmetric(laws):
    db, embed = laws
    assert _labels(space_difference("t", "a", "b", db, 0.85, embed)) == [
        "sourdough starter feeding schedule"
    ]
    assert _labels(space_difference("t", "b", "a", db, 0.85, embed)) == [
        "tide tables for the north sea"
    ]


def test_union_deduplicates_the_shared_atom(laws):
    db, embed = laws
    assert _labels(space_union("t", "a", "b", db, 0.85, embed)) == [
        "Postgres connection pooling",
        "sourdough starter feeding schedule",
        "tide tables for the north sea",
    ]


def test_symmetric_difference_is_the_union_minus_the_intersection(laws):
    db, embed = laws
    sym = _labels(space_symmetric_difference("t", "a", "b", db, 0.85, embed))
    union = set(_labels(space_union("t", "a", "b", db, 0.85, embed)))
    inter = set(_labels(space_intersection("t", "a", "b", db, 0.85, embed)))
    assert sorted(sym) == sorted(union - inter)


def test_symmetric_difference_and_union_are_commutative(laws):
    db, embed = laws
    assert _labels(space_symmetric_difference("t", "a", "b", db, 0.85, embed)) == _labels(
        space_symmetric_difference("t", "b", "a", db, 0.85, embed)
    )
    assert _labels(space_union("t", "a", "b", db, 0.85, embed)) == _labels(
        space_union("t", "b", "a", db, 0.85, embed)
    )


def test_overlap_jaccard_is_commutative(laws):
    db, embed = laws
    ab = space_overlap("t", "a", "b", db, 0.85, embed)
    ba = space_overlap("t", "b", "a", db, 0.85, embed)
    assert ab.jaccard == pytest.approx(ba.jaccard)
    assert ab.bridge_space_name == ba.bridge_space_name


def test_identity_laws_for_a_space_against_itself(laws):
    """A∩A = A, A\\A = ∅, AΔA = ∅, A∪A = A."""
    db, embed = laws
    everything = ["Postgres connection pooling", "sourdough starter feeding schedule"]

    assert _labels(space_intersection("t", "a", "a", db, 0.85, embed)) == everything
    assert space_difference("t", "a", "a", db, 0.85, embed).atoms == []
    assert space_symmetric_difference("t", "a", "a", db, 0.85, embed).atoms == []
    assert _labels(space_union("t", "a", "a", db, 0.85, embed)) == everything
    assert space_overlap("t", "a", "a", db, 0.85, embed).jaccard == pytest.approx(1.0)


def test_facade_set_operations_run_against_the_write_space(db_path):
    """The Smrti wrappers pass write_space as the left operand of each op."""
    shared = "Postgres is the database we run in production"
    a = Smrti(db_path=db_path, tenant_id="t", write_space="a")
    b = Smrti(db_path=db_path, tenant_id="t", write_space="b")
    a.remember(shared)
    a.remember("sourdough starter feeding schedule")
    b.remember(shared)
    b.remember("tide tables for the north sea")

    union = a.space_union("b", threshold=0.8)
    assert union.spaces == ["a", "b"]
    assert sorted(atom.label for atom in union.atoms) == sorted(
        [shared, "sourdough starter feeding schedule", "tide tables for the north sea"]
    )

    sym = a.space_symmetric_difference("b", threshold=0.8)
    assert sym.spaces == ["a", "b"]
    assert sorted(atom.label for atom in sym.atoms) == [
        "sourdough starter feeding schedule",
        "tide tables for the north sea",
    ]

    # Left operand is the caller's write space, so b sees the mirror image.
    assert [atom.label for atom in b.space_difference("a", threshold=0.8).atoms] == [
        "tide tables for the north sea"
    ]


def test_a_space_is_not_bridged_to_itself(db_path):
    """A∩A is all of A, so bridging it would duplicate the whole space."""
    mem = Smrti(db_path=db_path, tenant_id="t", write_space="a")
    mem.remember("Postgres is the database we run in production")

    assert mem.materialize_bridge("a", threshold=0.8, min_jaccard=0.05) == 0
    assert mem.list_spaces() == ["a"]


def test_set_operations_on_a_space_that_does_not_exist(laws):
    """An unknown space name behaves as the empty set, never as an error."""
    db, embed = laws
    everything = ["Postgres connection pooling", "sourdough starter feeding schedule"]

    assert space_overlap("t", "a", "ghost", db, 0.85, embed).jaccard == 0.0
    assert space_intersection("t", "a", "ghost", db, 0.85, embed).atoms == []
    assert _labels(space_difference("t", "a", "ghost", db, 0.85, embed)) == everything
    assert _labels(space_union("t", "a", "ghost", db, 0.85, embed)) == everything
    assert _labels(
        space_symmetric_difference("t", "a", "ghost", db, 0.85, embed)
    ) == everything


def test_set_operations_on_two_empty_spaces(db, embed):
    assert space_overlap("t", "x", "y", db, 0.85, embed).jaccard == 0.0
    assert space_union("t", "x", "y", db, 0.85, embed).atoms == []
    assert space_difference("t", "x", "y", db, 0.85, embed).atoms == []
    assert space_intersection("t", "x", "y", db, 0.85, embed).atoms == []
    assert space_symmetric_difference("t", "x", "y", db, 0.85, embed).atoms == []


def test_matching_is_one_to_one(atomspace, db, embed):
    """Two copies in B cannot both match the single atom in A.

    Greedy assignment keeps the pairing injective, so Jaccard stays a ratio of
    distinct atoms rather than counting one atom twice.
    """
    atomspace.add_atom(_concept("Postgres connection pooling", "t", "a"))
    atomspace.add_atom(_concept("Postgres connection pooling", "t", "b"))
    atomspace.add_atom(_concept("Postgres connection pooling", "t", "b"))

    overlap = space_overlap("t", "a", "b", db, 0.85, embed)
    assert len(overlap.pairs) == 1
    assert len({p.atom_a.id for p in overlap.pairs}) == 1
    assert len({p.atom_b.id for p in overlap.pairs}) == 1
    # |A| = 1, |B| = 2, one match → 1 / (1 + 2 - 1)
    assert overlap.jaccard == pytest.approx(0.5)


def test_neighborhood_signal_stays_inside_the_space(atomspace, db, embed):
    """A bridge edge must not import a foreign label into the context vector.

    ``_get_neighbor_labels`` scopes the relation *and* the neighbor it lands
    on; without the second filter an atom's neighborhood would be described by
    labels from a space the caller is not comparing.
    """
    from smrti.spaces.set_ops import _get_neighbor_labels

    subject = _concept("Java", "t", "a")
    local = _concept("JVM", "t", "a")
    foreign = _concept("Bali", "t", "b")
    for atom in (subject, local, foreign):
        atomspace.add_atom(atom)
    atomspace.link_atoms(subject.id, local.id, "associated", "t", "a")
    atomspace.link_atoms(subject.id, foreign.id, "bridge", "t", "a")

    assert _get_neighbor_labels(subject.id, "t", "a", db) == ["JVM"]


def test_relation_atoms_never_appear_in_set_results(atomspace, db, embed):
    atomspace.add_atom(_concept("Postgres", "t", "a"))
    other = _concept("Postgres", "t", "b")
    atomspace.add_atom(other)
    extra = _concept("Redis", "t", "a")
    atomspace.add_atom(extra)
    atomspace.link_atoms(extra.id, other.id, "associated", "t", "a")

    for result in (
        space_union("t", "a", "b", db, 0.85, embed),
        space_difference("t", "a", "b", db, 0.85, embed),
        space_symmetric_difference("t", "a", "b", db, 0.85, embed),
        space_intersection("t", "a", "b", db, 0.85, embed),
    ):
        assert all(a.type != AtomType.RELATION for a in result.atoms)


def test_threshold_controls_what_counts_as_a_match(atomspace, db, embed):
    """A high enough threshold rejects everything; identical atoms always pass."""
    atomspace.add_atom(_concept("Postgres connection pooling", "t", "a"))
    atomspace.add_atom(_concept("Postgres connection pooling", "t", "b"))

    assert space_overlap("t", "a", "b", db, 0.85, embed).pairs
    assert space_overlap("t", "a", "b", db, 1.01, embed).pairs == []
    # With nothing matched, difference returns the whole left space.
    assert len(space_difference("t", "a", "b", db, 1.01, embed).atoms) == 1
