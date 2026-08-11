"""Tests for personality presets re-export (personality/presets.py)."""
from smrti.personality.presets import PRESETS, load_preset, PersonalityProfile


def test_presets_exported():
    assert PRESETS is not None
    assert len(PRESETS) > 0


def test_load_preset_exported():
    profile = load_preset("balanced")
    assert isinstance(profile, PersonalityProfile)


def test_personality_profile_exported():
    assert PersonalityProfile is not None


# ── schema / dataclass / town parameter-list consistency ──────────────────────

def _profile_fields():
    from dataclasses import fields
    return {f.name for f in fields(PersonalityProfile)} - {"preset_name"}


def test_every_profile_field_has_a_personality_column(tmp_path):
    """A field with no column makes every INSERT on that table raise."""
    from smrti.core.db import get_database

    db = get_database(str(tmp_path / "cols.db"))
    columns = {r["name"] for r in db.fetchall("PRAGMA table_info(personality)")}
    assert _profile_fields() <= columns


def test_profile_round_trips_through_the_database(tmp_path):
    """Persisting a profile must not silently drop the newer parameters."""
    from dataclasses import replace

    from smrti import Smrti

    mem = Smrti(
        db_path=str(tmp_path / "rt.db"), tenant_id="t1", write_space="s1",
    )
    profile = replace(load_preset("balanced"), lti_decay_rate=0.037, agent_source_trust=0.21)
    mem.set_personality_profile(profile, "custom")

    row = mem.db.fetchone(
        "SELECT * FROM personality WHERE tenant_id = 't1' AND space = 's1'"
    )
    for field in _profile_fields():
        assert row[field] == getattr(profile, field), f"{field} did not round-trip"
    mem.db.close()


def test_town_inherits_every_personality_parameter():
    """smrti-town blends personalities across generations from its own list.

    A parameter missing here is never inherited or mutated — children silently
    fall back to the schema default forever, which is invisible at runtime.
    """
    from smrti_town.config import PARAM_BOUNDS, PERSONALITY_PARAMS

    assert _profile_fields() == set(PERSONALITY_PARAMS)
    assert _profile_fields() <= set(PARAM_BOUNDS)


def test_town_bounds_admit_every_preset_value():
    """Clamping a preset's own value would silently rewrite that personality."""
    from smrti_town.config import PARAM_BOUNDS

    for name, profile in PRESETS.items():
        for param, (lo, hi) in PARAM_BOUNDS.items():
            value = getattr(profile, param)
            assert lo <= value <= hi, f"{name}.{param}={value} outside {(lo, hi)}"
