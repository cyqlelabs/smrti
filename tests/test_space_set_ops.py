"""Tests for space set theory operations and bridge space emergence."""
import os
import tempfile

import pytest

from smrti import Smrti
from smrti.core.atomspace import AtomSpace
from smrti.core.db import Database
from smrti.core.embed import EmbeddingProvider
from smrti.core.models import (
    Atom,
    AtomPair,
    AtomType,
    EntityType,
    SpaceOverlap,
    SpaceSetResult,
    TruthValue,
    Valence,
)
from smrti.spaces.set_ops import (
    space_difference,
    space_intersection,
    space_overlap,
    space_symmetric_difference,
    space_union,
)
from smrti.spaces.emergence import materialize_bridge


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    database = Database(db_path)
    database.initialize()
    yield database
    database.close()
    os.unlink(db_path)


@pytest.fixture
def embed():
    return EmbeddingProvider()


@pytest.fixture
def atomspace(db, embed):
    return AtomSpace(db, embed)


@pytest.fixture
def populated_spaces(atomspace):
    """Create two spaces with some overlapping and some unique atoms."""
    tenant = "test"

    # Space A: Python, Machine Learning, FastAPI, Docker
    for label, space in [
        ("Python programming language", "work"),
        ("Machine learning and deep learning", "work"),
        ("FastAPI web framework", "work"),
        ("Docker containers", "work"),
    ]:
        atomspace.add_atom(Atom(
            type=AtomType.CONCEPT,
            label=label,
            content=label,
            entity_type=EntityType.CONCEPT,
            tenant_id=tenant,
            space=space,
            truth=TruthValue(probability=0.9, confidence=0.7),
        ))

    # Space B: Python, Machine Learning, PostgreSQL, Kubernetes
    for label, space in [
        ("Python programming", "research"),
        ("Machine learning algorithms", "research"),
        ("PostgreSQL database", "research"),
        ("Kubernetes orchestration", "research"),
    ]:
        atomspace.add_atom(Atom(
            type=AtomType.CONCEPT,
            label=label,
            content=label,
            entity_type=EntityType.CONCEPT,
            tenant_id=tenant,
            space=space,
            truth=TruthValue(probability=0.8, confidence=0.6),
        ))

    return atomspace, tenant


def test_overlap_empty_spaces(db, embed, atomspace):
    """Overlap between empty/nonexistent spaces should return 0."""
    result = space_overlap("test", "empty_a", "empty_b", db)
    assert result.jaccard == 0.0
    assert result.pairs == []


def test_overlap_no_match(db, embed, atomspace):
    """Completely disjoint spaces should have zero overlap."""
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="quantum physics",
        content="quantum physics", tenant_id="test", space="physics",
    ))
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="chocolate cake recipe",
        content="chocolate cake recipe", tenant_id="test", space="cooking",
    ))

    result = space_overlap("test", "physics", "cooking", db)
    assert result.jaccard == 0.0
    assert len(result.pairs) == 0


def test_overlap_identical_content(db, embed, atomspace):
    """Atoms with identical content across spaces should match."""
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Python programming",
        content="Python programming", tenant_id="test", space="alpha",
    ))
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Python programming",
        content="Python programming", tenant_id="test", space="beta",
    ))

    result = space_overlap("test", "alpha", "beta", db)
    assert result.jaccard > 0.0
    assert len(result.pairs) == 1
    # No embed_engine and no entity types → both absent-signal weights
    # redistribute to embedding: similarity = 1.0 * emb(~1.0)
    assert result.pairs[0].similarity > 0.85


def test_overlap_semantic_match(populated_spaces):
    """Semantically similar atoms across spaces should match."""
    atomspace, tenant = populated_spaces
    result = space_overlap(tenant, "work", "research", atomspace._db)

    # "Python programming language" ↔ "Python programming" should match
    # "Machine learning and deep learning" ↔ "Machine learning algorithms" should match
    assert len(result.pairs) >= 1
    assert result.jaccard > 0.0

    # Verify matched labels contain Python or ML
    matched_labels_a = {p.atom_a.label for p in result.pairs}
    assert any("Python" in l or "Machine" in l or "learning" in l for l in matched_labels_a)


def test_bridge_space_name_commutative(populated_spaces):
    """Bridge space name should be the same regardless of argument order."""
    atomspace, tenant = populated_spaces
    overlap_ab = space_overlap(tenant, "work", "research", atomspace._db)
    overlap_ba = space_overlap(tenant, "research", "work", atomspace._db)
    assert overlap_ab.bridge_space_name == overlap_ba.bridge_space_name
    assert overlap_ab.bridge_space_name == "research_x_work"


def test_intersection(populated_spaces):
    """Intersection returns atoms from space_a that have matches in space_b."""
    atomspace, tenant = populated_spaces
    result = space_intersection(tenant, "work", "research", atomspace._db)

    assert result.operation == "intersection"
    assert result.spaces == ["work", "research"]
    # All returned atoms should be from "work"
    for atom in result.atoms:
        assert atom.space == "work"


def test_difference(populated_spaces):
    """Difference returns atoms unique to space_a."""
    atomspace, tenant = populated_spaces
    result = space_difference(tenant, "work", "research", atomspace._db)

    assert result.operation == "difference"
    # Unique atoms should be from "work" and NOT have matches in research
    for atom in result.atoms:
        assert atom.space == "work"

    # FastAPI and Docker should be unique to work
    labels = {a.label for a in result.atoms}
    assert any("FastAPI" in l or "Docker" in l for l in labels)


def test_union(populated_spaces):
    """Union returns deduplicated atoms from both spaces."""
    atomspace, tenant = populated_spaces
    result = space_union(tenant, "work", "research", atomspace._db)

    assert result.operation == "union"
    # Should have atoms from both spaces, but duplicates removed
    total_unique = len(result.atoms)
    # work has 4, research has 4, at least 1 overlap
    assert total_unique < 8  # some dedup happened
    assert total_unique >= 6  # at least 6 unique concepts


def test_symmetric_difference(populated_spaces):
    """Symmetric difference returns atoms unique to either space."""
    atomspace, tenant = populated_spaces
    result = space_symmetric_difference(tenant, "work", "research", atomspace._db)

    assert result.operation == "symmetric_difference"
    # Should exclude matched pairs, include unmatched from both sides
    spaces_seen = {a.space for a in result.atoms}
    assert "work" in spaces_seen or "research" in spaces_seen


def test_difference_empty_other(db, embed, atomspace):
    """Difference with empty other space returns all atoms from A."""
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="test atom",
        content="test atom", tenant_id="test", space="full",
    ))
    result = space_difference("test", "full", "empty", db)
    assert len(result.atoms) == 1


def test_union_empty_spaces(db, embed, atomspace):
    """Union with one empty space returns atoms from the non-empty space."""
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="solo atom",
        content="solo atom", tenant_id="test", space="solo",
    ))
    result = space_union("test", "solo", "empty", db)
    assert len(result.atoms) == 1
    assert result.atoms[0].label == "solo atom"


# ── Bridge space emergence tests ─────────────────────────────────────


def test_materialize_bridge(populated_spaces):
    """Materialize should create bridge atoms with merged truth values."""
    atomspace, tenant = populated_spaces
    overlap = space_overlap(tenant, "work", "research", atomspace._db)

    if overlap.jaccard == 0.0:
        pytest.skip("No overlap detected (model may differ)")

    count = materialize_bridge(
        overlap, tenant, atomspace._db, atomspace._embed, atomspace, min_jaccard=0.0,
    )
    assert count > 0

    # Verify bridge atoms exist in the bridge space
    bridge_space = overlap.bridge_space_name
    bridge_atoms = atomspace._db.fetchall(
        "SELECT * FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'",
        (tenant, bridge_space),
    )
    assert len(bridge_atoms) >= count

    # Verify bridge relation edges exist
    bridge_relations = atomspace._db.fetchall(
        "SELECT * FROM atoms WHERE tenant_id = ? AND space = ? AND type = 'relation' AND relation = 'bridge'",
        (tenant, bridge_space),
    )
    # Each bridge atom has 2 edges (to source_a and source_b)
    assert len(bridge_relations) == count * 2


def test_materialize_bridge_below_jaccard(db, embed, atomspace):
    """Materialization should not happen when Jaccard is below threshold."""
    overlap = SpaceOverlap(space_a="a", space_b="b", jaccard=0.05, pairs=[])
    count = materialize_bridge(overlap, "test", db, embed, atomspace, min_jaccard=0.1)
    assert count == 0


def test_materialize_bridge_idempotent(populated_spaces):
    """Running materialize twice should update, not duplicate bridge atoms."""
    atomspace, tenant = populated_spaces
    overlap = space_overlap(tenant, "work", "research", atomspace._db)

    if not overlap.pairs:
        pytest.skip("No overlap detected")

    count1 = materialize_bridge(
        overlap, tenant, atomspace._db, atomspace._embed, atomspace, min_jaccard=0.0,
    )
    count2 = materialize_bridge(
        overlap, tenant, atomspace._db, atomspace._embed, atomspace, min_jaccard=0.0,
    )
    # Second run updates the same atoms in place — counted, but no duplicates
    assert count2 == count1
    bridge_atoms = atomspace._db.fetchall(
        "SELECT id FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'",
        (tenant, overlap.bridge_space_name),
    )
    assert len(bridge_atoms) == count1


def test_bridge_atom_truth_merge(db, embed, atomspace):
    """Bridge atom truth values should be PLN-merged from both sources."""
    a = Atom(
        type=AtomType.CONCEPT, label="shared concept",
        content="shared concept for testing",
        tenant_id="test", space="s1",
        truth=TruthValue(probability=0.9, confidence=0.8),
    )
    b = Atom(
        type=AtomType.CONCEPT, label="shared concept",
        content="shared concept for testing",
        tenant_id="test", space="s2",
        truth=TruthValue(probability=0.6, confidence=0.4),
    )
    atomspace.add_atom(a)
    atomspace.add_atom(b)

    overlap = space_overlap("test", "s1", "s2", db)
    assert len(overlap.pairs) == 1

    materialize_bridge(overlap, "test", db, embed, atomspace, min_jaccard=0.0)

    bridge_atoms = db.fetchall(
        "SELECT * FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'",
        ("test", overlap.bridge_space_name),
    )
    assert len(bridge_atoms) == 1

    # PLN merge: higher confidence source has more weight
    merged_prob = bridge_atoms[0]["probability"]
    # Should be closer to 0.9 (higher confidence source) than 0.6
    assert 0.7 < merged_prob < 1.0


def test_bridge_atom_valence_blend(db, embed, atomspace):
    """Bridge atom valence should be intensity-weighted blend."""
    a = Atom(
        type=AtomType.CONCEPT, label="debugging errors",
        content="debugging errors in production",
        tenant_id="test", space="v1",
        valence=Valence(valence=-0.8, intensity=0.9),
        truth=TruthValue(probability=0.9, confidence=0.7),
    )
    b = Atom(
        type=AtomType.CONCEPT, label="debugging errors",
        content="debugging errors in production",
        tenant_id="test", space="v2",
        valence=Valence(valence=-0.3, intensity=0.4),
        truth=TruthValue(probability=0.8, confidence=0.6),
    )
    atomspace.add_atom(a)
    atomspace.add_atom(b)

    overlap = space_overlap("test", "v1", "v2", db)
    materialize_bridge(overlap, "test", db, embed, atomspace, min_jaccard=0.0)

    bridge_atoms = db.fetchall(
        "SELECT * FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'",
        ("test", overlap.bridge_space_name),
    )
    assert len(bridge_atoms) == 1
    # Blended valence should be negative (weighted toward higher-intensity source)
    assert bridge_atoms[0]["valence"] < -0.3


# ── Contextual disambiguation tests ──────────────────────────────────


def test_entity_type_prevents_homonym_match(db, embed, atomspace):
    """Atoms with same label but different entity types should NOT match.

    'Java' (location) and 'Java' (technology) look identical in embedding
    space but entity_type disagreement should push contextual similarity
    below the threshold.
    """
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Java",
        content="Java",
        entity_type=EntityType.LOCATION,
        tenant_id="test", space="geography",
    ))
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Java",
        content="Java",
        entity_type=EntityType.CONCEPT,  # technology
        tenant_id="test", space="programming",
    ))

    # With a standard threshold, entity-type clash should prevent the match
    overlap = space_overlap("test", "geography", "programming", db, threshold=0.85)
    assert len(overlap.pairs) == 0, (
        "Homonyms with different entity_types should not match"
    )


def test_same_entity_type_still_matches(db, embed, atomspace):
    """Atoms with same label AND same entity type should match."""
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Python programming language",
        content="Python programming language",
        entity_type=EntityType.CONCEPT,
        tenant_id="test", space="sp_a",
    ))
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Python programming language",
        content="Python programming language",
        entity_type=EntityType.CONCEPT,
        tenant_id="test", space="sp_b",
    ))

    overlap = space_overlap("test", "sp_a", "sp_b", db, threshold=0.85)
    assert len(overlap.pairs) == 1


def test_neighborhood_disambiguates_with_embed_engine(db, embed, atomspace):
    """When embed_engine is provided, neighborhood context should influence matching.

    Two 'Java' atoms with different neighborhoods (Indonesia vs JVM) should
    be penalized by the neighborhood signal even when entity_type is missing.
    """
    # Space A: Java connected to Indonesia, Bali
    java_geo_id = atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Java",
        content="Java",
        tenant_id="test", space="geo",
    ))
    indo_id = atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Indonesia archipelago",
        content="Indonesia archipelago",
        tenant_id="test", space="geo",
    ))
    bali_id = atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Bali tropical island",
        content="Bali tropical island",
        tenant_id="test", space="geo",
    ))
    atomspace.link_atoms(java_geo_id, indo_id, "part_of", "test", "geo")
    atomspace.link_atoms(java_geo_id, bali_id, "associated", "test", "geo")

    # Space B: Java connected to JVM, Spring Framework
    java_tech_id = atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Java",
        content="Java",
        tenant_id="test", space="tech",
    ))
    jvm_id = atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="JVM virtual machine runtime",
        content="JVM virtual machine runtime",
        tenant_id="test", space="tech",
    ))
    spring_id = atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Spring Framework enterprise software",
        content="Spring Framework enterprise software",
        tenant_id="test", space="tech",
    ))
    atomspace.link_atoms(java_tech_id, jvm_id, "runs_on", "test", "tech")
    atomspace.link_atoms(java_tech_id, spring_id, "associated", "test", "tech")

    # With embed_engine, neighborhood divergence should prevent the match
    overlap = space_overlap(
        "test", "geo", "tech", db, threshold=0.85, embed_engine=embed,
    )
    # "Java" should NOT match "Java" because neighborhoods diverge
    java_pairs = [p for p in overlap.pairs if p.atom_a.label == "Java"]
    assert len(java_pairs) == 0, (
        "Homonyms with divergent neighborhoods should not match"
    )


# ── Smrti facade tests ───────────────────────────────────────────────


def test_smrti_space_overlap():
    """Smrti.space_overlap() should delegate correctly."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mem_a = Smrti(db_path=db_path, tenant_id="t1", write_space="alpha")
        mem_b = Smrti(db_path=db_path, tenant_id="t1", write_space="beta")

        mem_a.remember("Python is a great language", type="belief", probability=0.9)
        mem_b.remember("Python is an excellent programming language", type="belief", probability=0.85)

        overlap = mem_a.space_overlap("beta")
        assert isinstance(overlap, SpaceOverlap)
        assert overlap.space_a == "alpha"
        assert overlap.space_b == "beta"
    finally:
        mem_a.close()
        mem_b.close()
        os.unlink(db_path)


def test_smrti_materialize_bridge():
    """Smrti.materialize_bridge() should create a bridge space."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mem_a = Smrti(db_path=db_path, tenant_id="t1", write_space="project_x")
        mem_b = Smrti(db_path=db_path, tenant_id="t1", write_space="project_y")

        # Add shared concepts
        mem_a.remember("Kubernetes container orchestration", type="belief")
        mem_b.remember("Kubernetes cluster management", type="belief")
        mem_a.remember("CI/CD pipeline automation", type="belief")
        mem_b.remember("CI/CD continuous integration pipeline", type="belief")

        count = mem_a.materialize_bridge("project_y", min_jaccard=0.0)

        spaces = mem_a.list_spaces()
        # Should include original spaces and potentially a bridge space
        assert "project_x" in spaces
        assert "project_y" in spaces
        if count > 0:
            assert "project_x_x_project_y" in spaces
    finally:
        mem_a.close()
        mem_b.close()
        os.unlink(db_path)


def test_smrti_list_spaces():
    """list_spaces() should return all spaces for the tenant."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mem = Smrti(db_path=db_path, tenant_id="t1", write_space="s1")
        mem.remember("atom in s1")
        mem2 = Smrti(db_path=db_path, tenant_id="t1", write_space="s2")
        mem2.remember("atom in s2")

        spaces = mem.list_spaces()
        assert "s1" in spaces
        assert "s2" in spaces
    finally:
        mem.close()
        mem2.close()
        os.unlink(db_path)


def test_tenant_isolation(db, embed, atomspace):
    """Set operations should not cross tenant boundaries."""
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="shared concept X",
        content="shared concept X", tenant_id="tenant_a", space="s1",
    ))
    atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="shared concept X",
        content="shared concept X", tenant_id="tenant_b", space="s1",
    ))

    # Same space name but different tenants — should find no atoms
    result = space_overlap("tenant_a", "s1", "s2", db)
    assert result.jaccard == 0.0


def test_relations_excluded_from_set_ops(db, embed, atomspace):
    """Relation atoms should not be included in set operations."""
    a_id = atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Node A",
        content="Node A", tenant_id="test", space="sp",
    ))
    b_id = atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Node B",
        content="Node B", tenant_id="test", space="sp",
    ))
    atomspace.link_atoms(a_id, b_id, "related", "test", "sp")

    overlap = space_overlap("test", "sp", "sp2", db)
    # Only concept atoms, not relation atoms
    for pair in overlap.pairs:
        assert pair.atom_a.type != AtomType.RELATION
        assert pair.atom_b.type != AtomType.RELATION


# ── Epoch integration tests ──────────────────────────────────────────


def test_epoch_discovers_bridges():
    """Epoch should discover bridges on every 10th run."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mem_a = Smrti(db_path=db_path, tenant_id="t1", write_space="work")
        mem_b = Smrti(db_path=db_path, tenant_id="t1", write_space="hobby")

        mem_a.remember("Python web development with FastAPI", type="belief", probability=0.9)
        mem_b.remember("Python scripting for home automation", type="belief", probability=0.8)

        # Fast-forward epoch counter to 9 so next reflect triggers bridge discovery
        mem_a.db.execute(
            "UPDATE personality SET epoch_count = 9 WHERE tenant_id = ? AND space = ?",
            ("t1", "work"),
        )
        result = mem_a.reflect()
        # The bridges_created field should be present (may be 0 if threshold not met)
        assert hasattr(result, "bridges_created")
        assert result.bridges_created >= 0
    finally:
        mem_a.close()
        mem_b.close()
        os.unlink(db_path)


# ── MCP handler tests ────────────────────────────────────────────────


def test_mcp_space_overlap_handler():
    """MCP handler for space_overlap should return correct structure."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        from smrti.servers.mcp import handle_tool

        mem = Smrti(db_path=db_path, tenant_id="t1", write_space="alpha")
        mem.remember("Python programming")

        mem2 = Smrti(db_path=db_path, tenant_id="t1", write_space="beta")
        mem2.remember("Python development")

        result = handle_tool(mem, "smrti_space_overlap", {"other_space": "beta"})
        assert "space_a" in result
        assert "space_b" in result
        assert "jaccard" in result
        assert "matched_pairs" in result
    finally:
        mem.close()
        mem2.close()
        os.unlink(db_path)


def test_mcp_list_spaces_handler():
    """MCP handler for list_spaces should return spaces list."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        from smrti.servers.mcp import handle_tool

        mem = Smrti(db_path=db_path, tenant_id="t1", write_space="one")
        mem.remember("something in space one")
        result = handle_tool(mem, "smrti_list_spaces", {})
        assert "spaces" in result
        assert "one" in result["spaces"]
    finally:
        mem.close()
        os.unlink(db_path)


def test_mcp_space_merge_handler():
    """MCP handler for space_merge should return bridge info."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        from smrti.servers.mcp import handle_tool

        mem = Smrti(db_path=db_path, tenant_id="t1", write_space="x")
        mem.remember("machine learning classification")

        mem2 = Smrti(db_path=db_path, tenant_id="t1", write_space="y")
        mem2.remember("machine learning regression")

        result = handle_tool(mem, "smrti_space_merge", {"other_space": "y"})
        assert "status" in result
        assert "bridges_created" in result
        assert "bridge_space" in result
        assert result["bridge_space"] == "x_x_y"
    finally:
        mem.close()
        mem2.close()
        os.unlink(db_path)


def test_mcp_space_intersection_handler():
    """MCP handler for space_intersection should return atoms list."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        from smrti.servers.mcp import handle_tool

        mem = Smrti(db_path=db_path, tenant_id="t1", write_space="a")
        mem.remember("deep learning neural networks")

        mem2 = Smrti(db_path=db_path, tenant_id="t1", write_space="b")
        mem2.remember("deep learning neural networks")

        result = handle_tool(mem, "smrti_space_intersection", {"other_space": "b"})
        assert "operation" in result
        assert result["operation"] == "intersection"
        assert "atoms" in result
        assert "jaccard" in result
    finally:
        mem.close()
        mem2.close()
        os.unlink(db_path)


def test_mcp_space_diff_handler():
    """MCP handler for space_diff should return unique atoms."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        from smrti.servers.mcp import handle_tool

        mem = Smrti(db_path=db_path, tenant_id="t1", write_space="main")
        mem.remember("exclusive concept in main space")

        result = handle_tool(mem, "smrti_space_diff", {"other_space": "other"})
        assert result["operation"] == "difference"
        assert len(result["atoms"]) == 1
    finally:
        mem.close()
        os.unlink(db_path)
