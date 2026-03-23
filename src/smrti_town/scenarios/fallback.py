"""Fallback council — hardcoded 5-member founding council for LLM-offline mode."""

from __future__ import annotations

from smrti_town.config import PRESET_TRAITS


def create_fallback_council() -> tuple[list[dict], list[dict]]:
    """Return (council_specs, citizen_specs) for 5 hardcoded council members.

    council_specs: dicts with ``name``, ``role``, ``personality``, ``governing_style``, ``traits``
    citizen_specs: dicts with ``name``, ``age``, ``personality``, ``skills``, ``bio``, ``council_role``
    """
    council_specs = [
        {
            "name": "Eleanor Blackwood",
            "role": "mayor",
            "personality": "balanced",
            "governing_style": "moderate",
            "traits": {
                "shyness": 0.2, "proactivity": 0.6, "leadership": 0.8, "laziness": 0.1,
                "adventurous": 0.3, "nurturing": 0.5, "stubbornness": 0.4, "creativity": 0.5,
            },
        },
        {
            "name": "Marcus Stone",
            "role": "sheriff",
            "personality": "deterministic",
            "governing_style": "strict",
            "traits": {
                "shyness": 0.3, "proactivity": 0.5, "leadership": 0.7, "laziness": 0.1,
                "adventurous": 0.2, "nurturing": 0.2, "stubbornness": 0.8, "creativity": 0.2,
            },
        },
        {
            "name": "Sofia Chen",
            "role": "superintendent",
            "personality": "curious",
            "governing_style": "progressive",
            "traits": {
                "shyness": 0.3, "proactivity": 0.7, "leadership": 0.5, "laziness": 0.2,
                "adventurous": 0.6, "nurturing": 0.6, "stubbornness": 0.3, "creativity": 0.8,
            },
        },
        {
            "name": "James Abbott",
            "role": "doctor",
            "personality": "empathetic",
            "governing_style": "careful",
            "traits": {
                "shyness": 0.4, "proactivity": 0.5, "leadership": 0.4, "laziness": 0.2,
                "adventurous": 0.2, "nurturing": 0.9, "stubbornness": 0.3, "creativity": 0.4,
            },
        },
        {
            "name": "Helena Voss",
            "role": "treasurer",
            "personality": "analytical",
            "governing_style": "conservative",
            "traits": {
                "shyness": 0.5, "proactivity": 0.4, "leadership": 0.5, "laziness": 0.1,
                "adventurous": 0.1, "nurturing": 0.3, "stubbornness": 0.7, "creativity": 0.4,
            },
        },
    ]

    # Map council members to citizen specs.
    _skill_map = {
        "mayor": {"leadership": 0.5, "literacy": 0.3, "commerce": 0.2},
        "sheriff": {"leadership": 0.3, "craftsmanship": 0.2},
        "superintendent": {"teaching": 0.5, "literacy": 0.4},
        "doctor": {"medicine": 0.5, "literacy": 0.3},
        "treasurer": {"commerce": 0.5, "literacy": 0.3},
    }
    _age_map = {
        "mayor": 45,
        "sheriff": 38,
        "superintendent": 34,
        "doctor": 42,
        "treasurer": 50,
    }
    _bio_map = {
        "mayor": "A pragmatic administrator who believes in steady, measured growth.",
        "sheriff": "A disciplined former guard who values order and the rule of law.",
        "superintendent": "A passionate educator determined to build schools for every child.",
        "doctor": "A compassionate healer who left the city to serve a community in need.",
        "treasurer": "A meticulous accountant who keeps every coin accounted for.",
    }

    citizen_specs = []
    for cs in council_specs:
        role = cs["role"]
        citizen_specs.append({
            "name": cs["name"],
            "age": _age_map.get(role, 35),
            "personality": cs["personality"],
            "skills": _skill_map.get(role, {}),
            "bio": _bio_map.get(role, ""),
            "council_role": role,
            "traits": cs["traits"],
        })

    return council_specs, citizen_specs
