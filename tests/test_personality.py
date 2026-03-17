"""Tests for personality presets."""
import pytest

from smrti.personality.params import PRESETS, PersonalityProfile, load_preset


def test_all_presets_load():
    for name in ["balanced", "analytical", "curious", "empathetic", "maverick"]:
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
