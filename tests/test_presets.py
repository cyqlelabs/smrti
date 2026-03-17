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
