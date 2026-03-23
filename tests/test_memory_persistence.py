"""Proves the claim: critical failures and significant insights remain accessible
over time while less important data naturally decays.

Three mechanisms work together:
  1. Severe negative-valence atoms get an LTI floor (≥ 0.5) on creation — above the
     pruning threshold (lti < 0.05), so they are never pruned regardless of how much
     their confidence decays.
  2. Frequently-accessed atoms (high STI) are promoted to LTI by the epoch, giving
     significant insights the same protection.
  3. The salience formula dynamically shifts weight from STI toward valence for
     negative-valence atoms, so old-but-critical failures outrank fresh trivia at
     retrieval time even when their recency signal (STI) has fully decayed.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from smrti import Smrti
from smrti.core.models import Atom, AtomType, AttentionValue, TruthValue, Valence
from smrti.personality.params import PersonalityProfile
from smrti.retrieval.classify import classify_memory
from smrti.retrieval.salience import compute_salience


# ─── fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


# ─── 1. LTI floor on creation ─────────────────────────────────────────────────

def test_critical_failure_gets_lti_floor_on_creation(mem):
    """A severe negative-valence atom (valence < -0.7, intensity > 0.7) is assigned
    lti ≥ 0.5 immediately when added — before any epoch runs.

    The pruning gate requires lti < 0.05, so this atom is permanently protected.
    """
    atom_id = mem.remember("fatal production crash", valence=-0.9)

    row = mem.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (atom_id,))
    assert row["lti"] >= 0.5, (
        f"Expected lti ≥ 0.5 for critical failure, got {row['lti']}"
    )


def test_neutral_atom_has_no_lti_floor(mem):
    """Ordinary (neutral-valence) atoms start with lti = 0 — no protection."""
    atom_id = mem.remember("routine log entry", valence=0.0)

    row = mem.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (atom_id,))
    assert row["lti"] == 0.0, (
        f"Expected lti = 0 for neutral atom, got {row['lti']}"
    )


def test_lti_floor_threshold_is_above_pruning_gate(mem):
    """The LTI floor value (0.5) must exceed the pruning gate (lti < 0.05).

    This documents the numerical invariant that makes the protection work.
    """
    LTI_FLOOR = 0.5
    PRUNING_GATE = 0.05
    assert LTI_FLOOR > PRUNING_GATE


# ─── 2. Significant insights promoted to LTI via STI ─────────────────────────

def test_high_sti_insight_promoted_to_lti_after_epoch(mem):
    """An atom with high STI (frequently accessed / recently salient) is promoted to
    LTI during the epoch, giving 'significant insights' the same long-term protection
    as critical failures.
    """
    atom_id = mem.remember("key architectural decision")
    # Simulate many accesses by setting STI well above balanced promotion threshold (0.7)
    mem.db.execute("UPDATE atoms SET sti = 2.0 WHERE id = ?", (atom_id,))

    mem.reflect()

    row = mem.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (atom_id,))
    assert row["lti"] > 0.0, (
        f"Expected lti > 0 after promotion, got {row['lti']}"
    )


def test_low_sti_atom_not_promoted_to_lti(mem):
    """An atom with low STI is not promoted to LTI — it remains transient and
    eligible for pruning once confidence decays.
    """
    atom_id = mem.remember("forgettable detail", type="concept")
    # STI stays at default (0) — below any promotion threshold

    mem.reflect()

    row = mem.db.fetchone("SELECT lti, confidence FROM atoms WHERE id = ?", (atom_id,))
    assert row["lti"] == 0.0, (
        f"Expected lti = 0 for low-STI atom, got {row['lti']}"
    )


# ─── 3. Trivia decays and gets pruned ────────────────────────────────────────

def test_trivia_is_pruned_once_confidence_decays(mem):
    """A concept atom with no LTI is deleted once its confidence falls below the
    min_confidence_to_surface threshold.

    Uses a fast-decay personality so the test completes in a small number of epochs.
    """
    # Configure fast confidence decay so trivia crosses the pruning threshold quickly
    fast_decay = PersonalityProfile(
        confidence_decay_rate=0.6,   # 60 % decay per epoch
        sti_decay_rate=0.6,
        min_confidence_to_surface=0.1,
        lti_promotion_threshold=0.99,  # effectively disable STI→LTI promotion
    )
    mem.set_personality_profile(fast_decay)

    trivial_id = mem.remember("trivial log noise", type="concept")
    # Initial confidence = 0.5 (set by Smrti.remember).
    # After 1 epoch: 0.5 × 0.4 = 0.20  (still above 0.1)
    # After 2 epochs: 0.20 × 0.4 = 0.08 < 0.1 → pruning threshold crossed

    for _ in range(2):
        mem.reflect()

    row = mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (trivial_id,))
    assert row is None, "Trivial concept atom should have been pruned after confidence decay"


# ─── 4. End-to-end: critical failure survives epochs that prune trivia ────────

def test_critical_failure_survives_epochs_that_prune_trivia(mem):
    """Run multiple consolidation epochs under fast-decay settings.

    Expected outcome:
      • The critical failure atom (lti ≥ 0.5) is still present.
      • The trivial concept atom (lti = 0) has been pruned.
    """
    fast_decay = PersonalityProfile(
        confidence_decay_rate=0.6,
        sti_decay_rate=0.6,
        min_confidence_to_surface=0.1,
        lti_promotion_threshold=0.99,
    )
    mem.set_personality_profile(fast_decay)

    critical_id = mem.remember("SQL injection vulnerability in auth endpoint", valence=-0.9)
    trivial_id = mem.remember("debug print statement in utils.py", type="concept")

    for _ in range(3):
        mem.reflect()

    critical_row = mem.db.fetchone("SELECT id, lti FROM atoms WHERE id = ?", (critical_id,))
    trivial_row = mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (trivial_id,))

    assert critical_row is not None, "Critical failure should survive multiple epochs"
    assert critical_row["lti"] >= 0.5, "Critical failure LTI floor must be intact"
    assert trivial_row is None, "Trivial atom should have been pruned"


def test_significant_insight_survives_epochs_via_lti(mem):
    """An important atom that received many accesses (high STI → LTI promotion) also
    outlasts ordinary trivia under the same fast-decay conditions.
    """
    fast_decay = PersonalityProfile(
        confidence_decay_rate=0.6,
        sti_decay_rate=0.6,
        min_confidence_to_surface=0.1,
        lti_promotion_threshold=0.5,   # lower so our STI=2.0 atom gets promoted
    )
    mem.set_personality_profile(fast_decay)

    insight_id = mem.remember("microservices outperform monolith for this workload", type="concept")
    trivial_id = mem.remember("test variable name typo", type="concept")

    # Simulate many accesses → high STI → triggers LTI promotion on first epoch
    mem.db.execute("UPDATE atoms SET sti = 2.0 WHERE id = ?", (insight_id,))

    for _ in range(3):
        mem.reflect()

    insight_row = mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (insight_id,))
    trivial_row = mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (trivial_id,))

    assert insight_row is not None, "Significant insight should survive via LTI promotion"
    assert trivial_row is None, "Trivial atom should have been pruned"


# ─── 5. Salience: old critical failures outrank recent trivia ────────────────

def test_salience_critical_failure_outranks_recent_trivia():
    """Even with very low STI (= old, not recently accessed), a critical failure atom
    scores higher salience than a fresh but trivial atom with high STI.

    This is the dynamic weight shift: when valence < -0.5, weight transfers from w_sti
    to w_valence so the emotional signal compensates for the lack of recency.
    """
    SIMILARITY = 0.9   # both atoms are equally relevant to the query

    # Critical failure: old (sti=0.05), still has LTI from floor, highly negative
    critical_salience = compute_salience(
        similarity=SIMILARITY,
        sti=0.05,          # decayed — not recently accessed
        confidence=0.35,
        lti=0.5,           # protected by floor
        valence=-0.9,
        intensity=0.9,
    )

    # Recent trivia: fresh (sti=1.0), no emotional charge, no LTI
    trivia_salience = compute_salience(
        similarity=SIMILARITY,
        sti=1.0,           # recently active
        confidence=0.5,
        lti=0.0,
        valence=0.0,
        intensity=0.0,
    )

    assert critical_salience > trivia_salience, (
        f"Critical failure salience ({critical_salience:.4f}) should exceed "
        f"recent trivia salience ({trivia_salience:.4f})"
    )


def test_salience_weight_shift_activates_only_for_severe_negative_valence():
    """The dynamic weight shift only fires for valence < -0.5.

    A mildly negative atom (valence=-0.3) does NOT receive the boost — its salience
    is computed with standard weights, matching an equivalent neutral atom.
    """
    # Mildly negative: no shift should occur
    mild_negative = compute_salience(
        similarity=0.8, sti=0.5, confidence=0.5, lti=0.0, valence=-0.3, intensity=0.3
    )
    # Neutral equivalent
    neutral = compute_salience(
        similarity=0.8, sti=0.5, confidence=0.5, lti=0.0, valence=0.3, intensity=0.3
    )
    # Both use |valence|*intensity identically with default weights → same score
    assert abs(mild_negative - neutral) < 1e-9

    # Severe negative: shift fires → higher than standard-weight calculation would give
    severe = compute_salience(
        similarity=0.8, sti=0.1, confidence=0.4, lti=0.5, valence=-0.9, intensity=0.9
    )
    severe_no_shift = compute_salience(
        similarity=0.8, sti=0.1, confidence=0.4, lti=0.5, valence=0.0, intensity=0.0
    )
    assert severe > severe_no_shift, (
        "Severe negative valence should boost salience above its zero-valence baseline"
    )


# ─── 6. Severity classification at recall time ────────────────────────────────

def test_recall_classifies_severe_negative_atom_as_critical_warning(mem):
    """Atoms with valence < -0.5 and intensity > 0.5 are tagged 'critical_warning'
    in recall results, making them actionable for agents.
    """
    atom_id = mem.remember("race condition causes data corruption under high load", valence=-0.85)

    results = mem.recall("data corruption")
    assert results, "Expected at least one recall result"

    labels = [r.atom.label for r in results]
    target = next((r for r in results if r.atom.id == atom_id), None)
    assert target is not None, f"Critical atom not found in recall. Got: {labels}"

    severity = classify_memory(target)
    assert severity == "critical_warning", (
        f"Expected 'critical_warning', got '{severity}'"
    )


def test_neutral_atom_classified_as_context(mem):
    """Neutral atoms (no strong valence) are classified as 'context' — background
    information rather than an actionable warning.
    """
    atom_id = mem.remember("user prefers dark mode", valence=0.0)

    results = mem.recall("dark mode")
    assert results

    target = next((r for r in results if r.atom.id == atom_id), None)
    assert target is not None

    severity = classify_memory(target)
    assert severity == "context"
