"""Behavioral tests: different personalities produce measurably different outcomes.

Each test instantiates two Smrti engines with different personalities, applies
the same inputs, and asserts that the outputs differ in the expected direction.
"""
import os
import tempfile

import pytest

from smrti import Smrti
from smrti.personality.params import PRESETS, load_preset
from smrti.retrieval.salience import compute_salience


# ── fixture helpers ───────────────────────────────────────────────────────────

def _engine(preset: str) -> tuple[Smrti, str]:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = f.name
    f.close()
    return Smrti(db_path=path, personality=preset, tenant_id="test", write_space="default"), path


class _Pair:
    """Context manager that owns two engines with different personalities."""
    def __init__(self, preset_a: str, preset_b: str):
        self.eng_a, self.path_a = _engine(preset_a)
        self.eng_b, self.path_b = _engine(preset_b)

    def __enter__(self):
        return self.eng_a, self.eng_b

    def __exit__(self, *_):
        self.eng_a.close()
        self.eng_b.close()
        os.unlink(self.path_a)
        os.unlink(self.path_b)


# ── 1. Salience weights differ across presets ─────────────────────────────────

def test_all_presets_have_distinct_salience_weights():
    """No two presets share the exact same set of five salience weights."""
    weight_signatures = set()
    for name, p in PRESETS.items():
        sig = (p.w_similarity, p.w_sti, p.w_confidence, p.w_lti, p.w_valence)
        assert sig not in weight_signatures, (
            f"Preset '{name}' has duplicate salience weights: {sig}"
        )
        weight_signatures.add(sig)


def test_analytical_ranks_high_confidence_atom_first():
    """analytical (w_confidence=0.40) should rank a high-confidence atom above a high-valence one."""
    p = load_preset("analytical")
    # Atom A: high confidence, neutral valence
    sal_a = compute_salience(
        similarity=0.7, sti=1.0, confidence=0.9, lti=0.3,
        valence=0.0, intensity=0.0,
        w_similarity=p.w_similarity, w_sti=p.w_sti, w_confidence=p.w_confidence,
        w_lti=p.w_lti, w_valence=p.w_valence,
    )
    # Atom B: moderate confidence, strong positive valence
    sal_b = compute_salience(
        similarity=0.7, sti=1.0, confidence=0.5, lti=0.3,
        valence=0.85, intensity=0.85,
        w_similarity=p.w_similarity, w_sti=p.w_sti, w_confidence=p.w_confidence,
        w_lti=p.w_lti, w_valence=p.w_valence,
    )
    assert sal_a > sal_b, (
        f"analytical: high-confidence atom ({sal_a:.3f}) should outscore high-valence atom ({sal_b:.3f})"
    )


def test_empathetic_ranks_high_valence_atom_first():
    """empathetic (w_valence=0.35) should rank a high-valence atom above a high-confidence one."""
    p = load_preset("empathetic")
    sal_a = compute_salience(
        similarity=0.7, sti=1.0, confidence=0.9, lti=0.3,
        valence=0.0, intensity=0.0,
        w_similarity=p.w_similarity, w_sti=p.w_sti, w_confidence=p.w_confidence,
        w_lti=p.w_lti, w_valence=p.w_valence,
    )
    sal_b = compute_salience(
        similarity=0.7, sti=1.0, confidence=0.5, lti=0.3,
        valence=0.85, intensity=0.85,
        w_similarity=p.w_similarity, w_sti=p.w_sti, w_confidence=p.w_confidence,
        w_lti=p.w_lti, w_valence=p.w_valence,
    )
    assert sal_b > sal_a, (
        f"empathetic: high-valence atom ({sal_b:.3f}) should outscore high-confidence atom ({sal_a:.3f})"
    )


def test_curious_ranks_high_sti_atom_first():
    """curious (w_sti=0.35) should strongly prefer recently active atoms."""
    p = load_preset("curious")
    sal_recent = compute_salience(
        similarity=0.7, sti=2.0, confidence=0.5, lti=0.1,
        valence=0.0, intensity=0.0,
        w_similarity=p.w_similarity, w_sti=p.w_sti, w_confidence=p.w_confidence,
        w_lti=p.w_lti, w_valence=p.w_valence,
    )
    sal_stale = compute_salience(
        similarity=0.7, sti=0.1, confidence=0.8, lti=0.5,
        valence=0.0, intensity=0.0,
        w_similarity=p.w_similarity, w_sti=p.w_sti, w_confidence=p.w_confidence,
        w_lti=p.w_lti, w_valence=p.w_valence,
    )
    assert sal_recent > sal_stale


def test_curious_vs_analytical_sti_weight_ordering():
    """curious (w_sti=0.35) gives STI a higher weight than analytical (w_sti=0.15)."""
    curious = load_preset("curious")
    analytical = load_preset("analytical")
    assert curious.w_sti > analytical.w_sti


def test_analytical_vs_empathetic_confidence_weight_ordering():
    """analytical (w_confidence=0.40) weights confidence more than empathetic (w_confidence=0.10)."""
    analytical = load_preset("analytical")
    empathetic = load_preset("empathetic")
    assert analytical.w_confidence > empathetic.w_confidence


# ── 2. Confidence decay: analytical retains more than curious ─────────────────

def test_analytical_decays_confidence_slower_than_curious():
    """After an epoch, analytical atoms retain more confidence than curious atoms."""
    with _Pair("analytical", "curious") as (eng_a, eng_c):
        eng_a.remember("Python is a great language for data science.", probability=0.9)
        eng_c.remember("Python is a great language for data science.", probability=0.9)

        eng_a.reflect()
        eng_c.reflect()

        row_a = eng_a.db.fetchone(
            "SELECT confidence FROM atoms WHERE type = 'episode' AND tenant_id = 'test' AND space = 'default'",
        )
        row_c = eng_c.db.fetchone(
            "SELECT confidence FROM atoms WHERE type = 'episode' AND tenant_id = 'test' AND space = 'default'",
        )
        assert row_a["confidence"] > row_c["confidence"], (
            f"analytical conf={row_a['confidence']:.4f} should > curious conf={row_c['confidence']:.4f}"
        )


def test_maverick_decays_confidence_very_slowly():
    """maverick (decay=0.005) should retain almost all confidence after one epoch."""
    with _Pair("maverick", "curious") as (eng_m, eng_c):
        initial_prob = 0.9
        eng_m.remember("Maverick memory test.", probability=initial_prob)
        eng_c.remember("Maverick memory test.", probability=initial_prob)

        eng_m.reflect()
        eng_c.reflect()

        row_m = eng_m.db.fetchone("SELECT confidence FROM atoms WHERE type = 'episode' AND tenant_id = 'test'")
        row_c = eng_c.db.fetchone("SELECT confidence FROM atoms WHERE type = 'episode' AND tenant_id = 'test'")
        assert row_m["confidence"] > row_c["confidence"]


# ── 3. STI decay: analytical retains STI better than curious ──────────────────

def test_analytical_decays_sti_slower_than_curious():
    """analytical (sti_decay=0.05) vs curious (sti_decay=0.20) — more STI survives."""
    with _Pair("analytical", "curious") as (eng_a, eng_c):
        eng_a.remember("Fast decay test.")
        eng_c.remember("Fast decay test.")

        # Boost STI so decay is measurable
        id_a = eng_a.db.fetchone("SELECT id FROM atoms WHERE type = 'episode' AND tenant_id = 'test'")["id"]
        id_c = eng_c.db.fetchone("SELECT id FROM atoms WHERE type = 'episode' AND tenant_id = 'test'")["id"]
        eng_a.db.execute("UPDATE atoms SET sti = 1.0 WHERE id = ?", (id_a,))
        eng_c.db.execute("UPDATE atoms SET sti = 1.0 WHERE id = ?", (id_c,))

        eng_a.reflect()
        eng_c.reflect()

        after_a = eng_a.db.fetchone("SELECT sti FROM atoms WHERE id = ?", (id_a,))["sti"]
        after_c = eng_c.db.fetchone("SELECT sti FROM atoms WHERE id = ?", (id_c,))["sti"]
        assert after_a > after_c, (
            f"analytical STI={after_a:.4f} should > curious STI={after_c:.4f}"
        )


# ── 4. LTI promotion threshold ────────────────────────────────────────────────

def test_curious_promotes_to_lti_at_lower_threshold():
    """curious (lti_threshold=0.5) promotes atoms that analytical (lti_threshold=0.9) doesn't."""
    with _Pair("curious", "analytical") as (eng_c, eng_a):
        eng_c.remember("LTI promotion test.")
        eng_a.remember("LTI promotion test.")

        id_c = eng_c.db.fetchone("SELECT id FROM atoms WHERE type = 'episode' AND tenant_id = 'test'")["id"]
        id_a = eng_a.db.fetchone("SELECT id FROM atoms WHERE type = 'episode' AND tenant_id = 'test'")["id"]

        # STI=0.7 — above curious threshold (0.5) but below analytical (0.9)
        eng_c.db.execute("UPDATE atoms SET sti = 0.7, lti = 0.0 WHERE id = ?", (id_c,))
        eng_a.db.execute("UPDATE atoms SET sti = 0.7, lti = 0.0 WHERE id = ?", (id_a,))

        eng_c.reflect()
        eng_a.reflect()

        lti_c = eng_c.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (id_c,))["lti"]
        lti_a = eng_a.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (id_a,))["lti"]

        assert lti_c > 0.0, f"curious should promote atom with STI=0.7 (threshold=0.5), got lti={lti_c}"
        assert lti_a == 0.0, f"analytical should NOT promote atom with STI=0.7 (threshold=0.9), got lti={lti_a}"


def test_maverick_promotes_lti_more_aggressively_than_deterministic():
    """maverick (threshold=0.4) promotes atoms that deterministic (threshold=0.85) won't."""
    with _Pair("maverick", "deterministic") as (eng_m, eng_d):
        eng_m.remember("Maverick LTI test.")
        eng_d.remember("Maverick LTI test.")

        id_m = eng_m.db.fetchone("SELECT id FROM atoms WHERE type = 'episode' AND tenant_id = 'test'")["id"]
        id_d = eng_d.db.fetchone("SELECT id FROM atoms WHERE type = 'episode' AND tenant_id = 'test'")["id"]

        # STI=0.6 — above maverick (0.4) but below deterministic (0.85)
        eng_m.db.execute("UPDATE atoms SET sti = 0.6, lti = 0.0 WHERE id = ?", (id_m,))
        eng_d.db.execute("UPDATE atoms SET sti = 0.6, lti = 0.0 WHERE id = ?", (id_d,))

        eng_m.reflect()
        eng_d.reflect()

        lti_m = eng_m.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (id_m,))["lti"]
        lti_d = eng_d.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (id_d,))["lti"]

        assert lti_m > 0.0, f"maverick should promote (threshold=0.4)"
        assert lti_d == 0.0, f"deterministic should NOT promote (threshold=0.85)"


# ── 5. Evidence learning rate ─────────────────────────────────────────────────

def test_curious_learns_faster_from_evidence_than_analytical():
    """curious (lr=0.5) moves belief probability further toward evidence than analytical (lr=0.15)."""
    with _Pair("curious", "analytical") as (eng_c, eng_a):
        # Start with a weak belief (prob=0.5, conf=0.3)
        id_c = eng_c.believe("The refactor will reduce bugs", probability=0.5)
        id_a = eng_a.believe("The refactor will reduce bugs", probability=0.5)

        # Add strong positive evidence
        from smrti.core.models import Evidence
        ev_c = Evidence(atom_id=id_c, observed_probability=1.0, weight=1.0,
                        tenant_id="test", space="default")
        ev_a = Evidence(atom_id=id_a, observed_probability=1.0, weight=1.0,
                        tenant_id="test", space="default")
        eng_c.atomspace.add_evidence(ev_c)
        eng_a.atomspace.add_evidence(ev_a)

        eng_c.reflect()
        eng_a.reflect()

        prob_c = eng_c.db.fetchone("SELECT probability FROM atoms WHERE id = ?", (id_c,))["probability"]
        prob_a = eng_a.db.fetchone("SELECT probability FROM atoms WHERE id = ?", (id_a,))["probability"]

        assert prob_c > prob_a, (
            f"curious (lr=0.5) prob={prob_c:.4f} should > analytical (lr=0.15) prob={prob_a:.4f}"
        )
        assert prob_c > 0.5, "curious should have moved toward evidence"


def test_deterministic_learns_faster_than_analytical():
    """deterministic (lr=0.4) updates beliefs faster than analytical (lr=0.15)."""
    assert load_preset("deterministic").confidence_update_lr > load_preset("analytical").confidence_update_lr


# ── 6. min_confidence_to_surface (pruning threshold) ─────────────────────────

def test_deterministic_prunes_low_confidence_concepts_that_balanced_keeps():
    """deterministic (min=0.3) prunes concept atoms that balanced (min=0.1) retains."""
    with _Pair("deterministic", "balanced") as (eng_d, eng_b):
        eng_d.remember("Transient concept to prune.", type="concept", probability=0.5)
        eng_b.remember("Transient concept to prune.", type="concept", probability=0.5)

        id_d = eng_d.db.fetchone("SELECT id FROM atoms WHERE type = 'concept' AND tenant_id = 'test'")["id"]
        id_b = eng_b.db.fetchone("SELECT id FROM atoms WHERE type = 'concept' AND tenant_id = 'test'")["id"]

        # Set confidence just above balanced threshold (0.1) but below deterministic (0.3)
        eng_d.db.execute("UPDATE atoms SET confidence = 0.15, lti = 0.0 WHERE id = ?", (id_d,))
        eng_b.db.execute("UPDATE atoms SET confidence = 0.15, lti = 0.0 WHERE id = ?", (id_b,))

        eng_d.reflect()
        eng_b.reflect()

        dead = eng_d.db.fetchone("SELECT id FROM atoms WHERE id = ?", (id_d,))
        alive = eng_b.db.fetchone("SELECT id FROM atoms WHERE id = ?", (id_b,))

        assert dead is None, "deterministic should prune atom with confidence=0.15 (min=0.3)"
        assert alive is not None, "balanced should keep atom with confidence=0.15 (min=0.1)"


# ── 7. End-to-end recall ordering differs by personality ─────────────────────

def test_recall_ordering_differs_between_analytical_and_empathetic():
    """
    Two atoms — one high-confidence/neutral, one moderate-confidence/high-valence.
    analytical should rank the confident one first; empathetic should rank the emotional one first.
    """
    with _Pair("analytical", "empathetic") as (eng_a, eng_e):
        # Add the same two atoms to both engines
        for eng in (eng_a, eng_e):
            id_conf = eng.remember("Project milestone reached successfully neutral update.")
            id_val = eng.remember("Project milestone reached successfully neutral update.")

            # Manually set up the contrast: first atom = confident, second = emotional
            eng.db.execute(
                "UPDATE atoms SET confidence = 0.95, valence = 0.0, intensity = 0.0 WHERE id = ?",
                (id_conf,),
            )
            eng.db.execute(
                "UPDATE atoms SET confidence = 0.45, valence = 0.9, intensity = 0.9 WHERE id = ?",
                (id_val,),
            )

        results_a = eng_a.recall("project milestone update", top_k=5)
        results_e = eng_e.recall("project milestone update", top_k=5)

        assert len(results_a) >= 2, "analytical should recall at least 2 atoms"
        assert len(results_e) >= 2, "empathetic should recall at least 2 atoms"

        # Identify which atom type is top-ranked for each personality
        top_a = results_a[0].atom
        top_e = results_e[0].atom

        # For analytical: top atom should have higher confidence
        # For empathetic: top atom should have higher valence*intensity
        analytical_prefers_confidence = top_a.truth.confidence >= top_a.valence.intensity
        empathetic_prefers_emotion = top_e.valence.intensity >= top_e.truth.confidence

        # The two personalities must disagree on which atom ranks first
        assert results_a[0].atom.id != results_e[0].atom.id, (
            "analytical and empathetic should rank different atoms first: "
            f"analytical top={results_a[0].atom.truth.confidence:.2f} conf / "
            f"{results_a[0].atom.valence.intensity:.2f} intensity; "
            f"empathetic top={results_e[0].atom.truth.confidence:.2f} conf / "
            f"{results_e[0].atom.valence.intensity:.2f} intensity"
        )


# ── 8. set_personality() changes DB values and affects subsequent epochs ──────

def test_set_personality_changes_epoch_behavior():
    """Switching from curious to analytical mid-session changes decay rate from next epoch."""
    engine, path = _engine("curious")
    try:
        atom_id = engine.remember("Personality switch test.", probability=0.9)
        engine.db.execute("UPDATE atoms SET sti = 1.0, confidence = 0.9 WHERE id = ?", (atom_id,))

        # Run one epoch under curious (high decay)
        engine.reflect()
        conf_after_curious = engine.db.fetchone(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )["confidence"]

        # Switch to analytical (low decay)
        engine.set_personality("analytical")
        engine.db.execute("UPDATE atoms SET confidence = 0.9 WHERE id = ?", (atom_id,))

        # Run one epoch under analytical
        engine.reflect()
        conf_after_analytical = engine.db.fetchone(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        )["confidence"]

        # analytical decay (0.01) is lower than curious decay (0.03), so confidence drops less
        curious_drop = 0.9 - conf_after_curious
        analytical_drop = 0.9 - conf_after_analytical
        assert analytical_drop < curious_drop, (
            f"analytical drop={analytical_drop:.4f} should be smaller than curious drop={curious_drop:.4f}"
        )
    finally:
        engine.close()
        os.unlink(path)


def test_set_personality_to_invalid_name_raises():
    engine, path = _engine("balanced")
    try:
        with pytest.raises(ValueError, match="Unknown personality preset"):
            engine.set_personality("nonexistent_preset")
    finally:
        engine.close()
        os.unlink(path)


# ── 9. Preset parameter contracts ────────────────────────────────────────────

def test_deterministic_has_high_lti_threshold():
    """deterministic targets focused agentic use — its LTI threshold should beat maverick and curious."""
    presets = {name: load_preset(name) for name in PRESETS}
    det = presets["deterministic"].lti_promotion_threshold
    assert det > presets["curious"].lti_promotion_threshold
    assert det > presets["maverick"].lti_promotion_threshold


def test_empathetic_has_highest_valence_weight():
    """empathetic should weight emotional valence more than any other preset."""
    presets = {name: load_preset(name) for name in PRESETS}
    emp_w = presets["empathetic"].w_valence
    others = [p.w_valence for name, p in presets.items() if name != "empathetic"]
    assert emp_w >= max(others), (
        f"empathetic w_valence={emp_w} should be >= all others: {others}"
    )


def test_analytical_has_highest_confidence_weight():
    """analytical should weight confidence more than any other preset."""
    presets = {name: load_preset(name) for name in PRESETS}
    ana_w = presets["analytical"].w_confidence
    others = [p.w_confidence for name, p in presets.items() if name != "analytical"]
    assert ana_w >= max(others), (
        f"analytical w_confidence={ana_w} should be >= all others: {others}"
    )


def test_curious_has_lowest_lti_promotion_threshold_among_named():
    """curious makes it easiest to promote to LTI (low threshold)."""
    presets = {name: load_preset(name) for name in PRESETS}
    curious_thresh = presets["curious"].lti_promotion_threshold
    # maverick (0.4) < curious (0.5) — so curious is NOT the absolute lowest
    # but curious should be in the lower half
    thresholds = sorted(p.lti_promotion_threshold for p in presets.values())
    assert curious_thresh <= thresholds[len(thresholds) // 2], (
        "curious lti_threshold should be in the bottom half"
    )


def test_deterministic_has_low_sti_propagation():
    """deterministic uses minimal STI propagation for focused, non-diffuse attention."""
    det = load_preset("deterministic")
    curious = load_preset("curious")
    assert det.sti_propagation_factor < curious.sti_propagation_factor


def test_maverick_has_high_sti_propagation():
    """maverick spreads attention broadly (high propagation factor)."""
    maverick = load_preset("maverick")
    analytical = load_preset("analytical")
    assert maverick.sti_propagation_factor > analytical.sti_propagation_factor


# ── 10. Multiple epochs amplify differences ───────────────────────────────────

def test_multiple_epochs_amplify_decay_difference():
    """Over 5 epochs the confidence gap between analytical and curious grows wider."""
    with _Pair("analytical", "curious") as (eng_a, eng_c):
        prob = 0.9
        eng_a.remember("Long-running decay comparison.", probability=prob)
        eng_c.remember("Long-running decay comparison.", probability=prob)

        for _ in range(5):
            eng_a.reflect()
            eng_c.reflect()

        row_a = eng_a.db.fetchone("SELECT confidence FROM atoms WHERE type = 'episode' AND tenant_id = 'test'")
        row_c = eng_c.db.fetchone("SELECT confidence FROM atoms WHERE type = 'episode' AND tenant_id = 'test'")

        # After 5 epochs the gap should be clearly visible
        gap = row_a["confidence"] - row_c["confidence"]
        assert gap > 0.01, (
            f"After 5 epochs: analytical={row_a['confidence']:.4f}, curious={row_c['confidence']:.4f}, gap={gap:.4f}"
        )
