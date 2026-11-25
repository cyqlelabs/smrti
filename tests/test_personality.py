"""Tests for personality presets."""
import pytest

from smrti import Smrti
from smrti.personality.params import PRESETS, PersonalityProfile, load_preset


def test_all_presets_load():
    for name in ["balanced", "analytical", "curious", "empathetic", "maverick", "deterministic"]:
        profile = load_preset(name)
        assert isinstance(profile, PersonalityProfile)
        assert profile.preset_name == name


def test_weights_sum_to_one():
    for name, profile in PRESETS.items():
        total = (
            profile.w_similarity
            + profile.w_sti
            + profile.w_confidence
            + profile.w_lti
            + profile.w_valence
        )
        assert abs(total - 1.0) < 0.01, f"Preset {name} weights sum to {total}, expected 1.0"


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        load_preset("nonexistent")


@pytest.mark.parametrize(
    "params",
    [{"w_sti": "abc"}, {"min_confidence_to_surface": 7}, {"sti_decay_rate": -0.1},
     {"w_lti": float("nan")}, {"mood_inertia": True}],
)
def test_a_profile_refuses_a_value_outside_its_range(params):
    with pytest.raises(ValueError):
        PersonalityProfile(**params)


def test_a_stored_value_outside_its_range_reads_as_the_default(tmp_path):
    """One bad write through /personality used to make every later recall in
    the space fail, across restarts."""
    mem = Smrti(db_path=str(tmp_path / "p.db"))
    mem.remember("The billing service runs on Postgres")
    mem.db.execute(
        "UPDATE personality SET w_sti = 'abc', min_confidence_to_surface = 7 "
        "WHERE tenant_id = ? AND space = ?",
        (mem.tenant_id, mem.write_space),
    )
    assert mem.recall("billing database")
    mem.reflect()
    mem.close()
