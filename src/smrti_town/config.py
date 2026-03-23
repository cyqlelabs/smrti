"""All constants for the smrti-town simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Calendar ─────────────────────────────────────────────────────────
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 28  # 4 seasons of 7 days each
HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR  # 672
DAYS_PER_SEASON = 7

SEASONS = ["spring", "summer", "autumn", "winter"]

TIME_OF_DAY_RANGES = {
    "night": (0, 6),
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
}

# ── Director tick deltas (sim-hours) ─────────────────────────────────
TICK_SCENE = 0.25       # 15 minutes — multi-agent interaction
TICK_ROUTINE = 2.0      # default daily activities
TICK_MONTAGE = 8.0      # everyone sleeping or solo
TICK_SKIP = 168.0       # 1 week fast-forward

# ── Citizen needs rates (per sim-hour) ───────────────────────────────
# 9-level Maslow hierarchy
HUNGER_RATE = 2
SHELTER_RATE = 0         # binary: 0 if housed, rises only if homeless
HEALTH_RATE = 0.5
SAFETY_RATE = 1          # function of crime rate
SOCIAL_RATE = 1
EDUCATION_RATE = 1
PURPOSE_RATE = 3         # only during work hours
CULTURE_RATE = 0.5
ACTUALIZATION_RATE = 0.3

# ── Needs thresholds (trigger action consideration) ──────────────────
HUNGER_THRESHOLD = 70
SHELTER_THRESHOLD = 50
HEALTH_THRESHOLD = 60
SAFETY_THRESHOLD = 50
SOCIAL_THRESHOLD = 60
EDUCATION_THRESHOLD = 50
PURPOSE_THRESHOLD = 40
CULTURE_THRESHOLD = 50
ACTUALIZATION_THRESHOLD = 60

# ── Needs clamps ─────────────────────────────────────────────────────
NEED_MAX = 100
NEED_MIN = 0

# ── Needs resets (value set after satisfying action) ─────────────────
HUNGER_RESET = 0
ENERGY_RESET = 100
SOCIAL_RESET_AMOUNT = 30
EDUCATION_RESET_AMOUNT = 25
CULTURE_RESET_AMOUNT = 20

# ── Life stages ──────────────────────────────────────────────────────
LIFE_STAGES = {
    "infant": {
        "age_range": (0, 5),
        "needs": ["hunger", "shelter", "health"],
        "can_move": False,
        "can_talk": False,
        "can_work": False,
        "can_reproduce": False,
        "energy_decay_mult": 0.5,
        "schedule": None,
    },
    "child": {
        "age_range": (5, 18),
        "needs": ["hunger", "shelter", "health", "safety", "social", "education"],
        "can_move": True,
        "can_talk": True,
        "can_work": False,
        "can_reproduce": False,
        "energy_decay_mult": 0.8,
        "schedule": {"school": (8, 14)},
    },
    "adult": {
        "age_range": (18, 65),
        "needs": ["hunger", "shelter", "health", "safety", "social", "education",
                  "purpose", "culture", "actualization"],
        "can_move": True,
        "can_talk": True,
        "can_work": True,
        "can_reproduce": True,
        "energy_decay_mult": 1.0,
        "schedule": {"work": (8, 17)},
    },
    "elder": {
        "age_range": (65, 200),
        "needs": ["hunger", "shelter", "health", "safety", "social", "culture"],
        "can_move": True,
        "can_talk": True,
        "can_work": False,
        "can_reproduce": False,
        "energy_decay_mult": 2.0,
        "schedule": None,
    },
}

# ── Milestones (age in years -> event type) ──────────────────────────
MILESTONES = {
    5: "school_enrollment",
    13: "adolescence",
    18: "graduation",
    22: "career_start",
    65: "retirement",
}

# ── Relationship gates ───────────────────────────────────────────────
RELATIONSHIP_GATES = {
    "acquaintance": {},
    "friend": {"interaction_count": 5, "valence": 0.2},
    "close_friend": {"friend_lti": 0.5, "valence": 0.4, "shared_episodes": 3},
    "romantic": {"close_friend_lti": 0.6, "mutual_valence": 0.5},
    "married": {"romantic_lti": 0.7, "mutual_valence": 0.6, "cohabitation_years": 1},
}

# ── Reproduction gate ────────────────────────────────────────────────
REPRODUCTION_GATE = {
    "min_relationship": "romantic",
    "also_allow_close_friend": True,
    "both_energy": 70,
    "life_stage": "adult",
    "min_relationship_years": 1,
}

# ── Death parameters ─────────────────────────────────────────────────
ELDER_DEATH_PROB_PER_TICK = 0.0003
STARVATION_HOURS = 48
DEATH_LOW_ENERGY_MULT = 2.0

# ── Personality inheritance ──────────────────────────────────────────
PERSONALITY_PARAMS = [
    "confidence_decay_rate", "confidence_update_lr", "min_confidence_to_surface",
    "sti_decay_rate", "sti_boost_on_access", "sti_propagation_factor",
    "lti_promotion_threshold", "valence_weight", "valence_propagation",
    "mood_inertia", "w_similarity", "w_sti", "w_confidence", "w_lti", "w_valence",
]

PARAM_BOUNDS = {
    "confidence_decay_rate": (0.001, 0.1),
    "confidence_update_lr": (0.05, 0.6),
    "min_confidence_to_surface": (0.05, 0.5),
    "sti_decay_rate": (0.01, 0.3),
    "sti_boost_on_access": (0.1, 1.0),
    "sti_propagation_factor": (0.01, 0.5),
    "lti_promotion_threshold": (0.3, 0.95),
    "valence_weight": (0.01, 0.5),
    "valence_propagation": (0.01, 0.4),
    "mood_inertia": (0.2, 0.99),
    "w_similarity": (0.1, 0.5),
    "w_sti": (0.05, 0.4),
    "w_confidence": (0.05, 0.5),
    "w_lti": (0.05, 0.2),
    "w_valence": (0.05, 0.4),
}

STRESS_VARIANCE_BASE = 1.0
STRESS_VARIANCE_MAX_MULT = 3.0

# ── Behavioural personality traits ───────────────────────────────────
TRAIT_NAMES = [
    "shyness", "proactivity", "leadership", "laziness",
    "adventurous", "nurturing", "stubbornness", "creativity",
]

TRAIT_BOUNDS = {t: (0.0, 1.0) for t in TRAIT_NAMES}

PRESET_TRAITS = {
    "balanced": {
        "shyness": 0.3, "proactivity": 0.5, "leadership": 0.4, "laziness": 0.3,
        "adventurous": 0.4, "nurturing": 0.5, "stubbornness": 0.3, "creativity": 0.5,
    },
    "analytical": {
        "shyness": 0.6, "proactivity": 0.4, "leadership": 0.3, "laziness": 0.2,
        "adventurous": 0.2, "nurturing": 0.3, "stubbornness": 0.7, "creativity": 0.6,
    },
    "curious": {
        "shyness": 0.2, "proactivity": 0.7, "leadership": 0.3, "laziness": 0.4,
        "adventurous": 0.8, "nurturing": 0.4, "stubbornness": 0.2, "creativity": 0.8,
    },
    "empathetic": {
        "shyness": 0.4, "proactivity": 0.6, "leadership": 0.5, "laziness": 0.3,
        "adventurous": 0.3, "nurturing": 0.9, "stubbornness": 0.2, "creativity": 0.5,
    },
    "maverick": {
        "shyness": 0.1, "proactivity": 0.8, "leadership": 0.6, "laziness": 0.5,
        "adventurous": 0.9, "nurturing": 0.3, "stubbornness": 0.6, "creativity": 0.7,
    },
    "deterministic": {
        "shyness": 0.5, "proactivity": 0.4, "leadership": 0.7, "laziness": 0.1,
        "adventurous": 0.1, "nurturing": 0.3, "stubbornness": 0.8, "creativity": 0.3,
    },
}

PERSONALITY_ACTION_BIAS = {
    "balanced":      {"social": 0.5, "education": 0.5, "purpose": 0.5, "culture": 0.5, "wander": 0.3},
    "analytical":    {"social": 0.2, "education": 0.9, "purpose": 0.7, "culture": 0.3, "wander": 0.1},
    "curious":       {"social": 0.6, "education": 0.9, "purpose": 0.3, "culture": 0.6, "wander": 0.7},
    "empathetic":    {"social": 0.9, "education": 0.4, "purpose": 0.4, "culture": 0.5, "wander": 0.3},
    "maverick":      {"social": 0.5, "education": 0.7, "purpose": 0.2, "culture": 0.7, "wander": 0.9},
    "deterministic": {"social": 0.3, "education": 0.6, "purpose": 0.9, "culture": 0.2, "wander": 0.1},
}

# ── Action types ─────────────────────────────────────────────────────
ACTION_MOVE = "MOVE"
ACTION_TALK = "TALK"
ACTION_EAT = "EAT"
ACTION_SLEEP = "SLEEP"
ACTION_WORK = "WORK"
ACTION_STUDY = "STUDY"
ACTION_INTERACT = "INTERACT"
ACTION_PROPOSE = "PROPOSE"
ACTION_REPRODUCE = "REPRODUCE"
ACTION_WAIT = "WAIT"
ACTION_WANDER = "WANDER"
ACTION_SHOP = "SHOP"
ACTION_PRAY = "PRAY"
ACTION_PLAY = "PLAY"

# ── Skill categories ─────────────────────────────────────────────────
SKILL_CATEGORIES = {
    "literacy": {
        "learned_at": ["school", "library"],
        "enables": ["teacher", "bookstore_staff", "library_staff"],
        "xp_per_hour": 0.01,
    },
    "medicine": {
        "learned_at": ["university", "clinic", "hospital"],
        "enables": ["doctor", "hospital_staff"],
        "xp_per_hour": 0.005,
    },
    "commerce": {
        "learned_at": ["market", "store", "bakery", "butcher", "tavern", "inn"],
        "enables": ["merchant", "business_owner"],
        "xp_per_hour": 0.008,
    },
    "craftsmanship": {
        "learned_at": ["blacksmith", "tailor"],
        "enables": ["blacksmith_staff", "tailor_staff", "builder"],
        "xp_per_hour": 0.007,
    },
    "teaching": {
        "learned_at": ["university", "library"],
        "enables": ["teacher", "professor"],
        "xp_per_hour": 0.005,
    },
    "leadership": {
        "learned_at": [],  # experience-based
        "enables": ["council_candidate", "manager"],
        "xp_per_hour": 0.003,
    },
    "agriculture": {
        "learned_at": ["farm"],
        "enables": ["farmer", "farm_efficiency"],
        "xp_per_hour": 0.01,
    },
    "arts": {
        "learned_at": ["theater", "museum"],
        "enables": ["artist", "cultural_contribution"],
        "xp_per_hour": 0.006,
    },
}

# ── Building catalog ─────────────────────────────────────────────────

@dataclass
class BuildingDef:
    key: str
    category: str          # residential, commercial, civic, infrastructure, cultural, industrial
    cost: int
    maintenance: int       # per year
    capacity: int          # residents for housing, 0 for non-residential
    staff_required: int
    staff_skill: str       # skill category needed
    staff_min_level: float  # 0.0-1.0
    unlock_population: int
    unlock_buildings: list[str] = field(default_factory=list)
    effects: dict = field(default_factory=dict)
    sprite_key: str = ""
    variants: int = 1
    description: str = ""
    provides_food: bool = False
    provides_goods: bool = False
    provides_housing: bool = False
    revenue_per_hour: int = 0


BUILDING_CATALOG: dict[str, BuildingDef] = {}


def _register(b: BuildingDef) -> BuildingDef:
    BUILDING_CATALOG[b.key] = b
    return b


# ── Residential ──────────────────────────────────────────────────────
_register(BuildingDef(
    key="town_hall", category="civic", cost=0, maintenance=200, capacity=10,
    staff_required=0, staff_skill="", staff_min_level=0, unlock_population=0,
    sprite_key="town_hall", description="Center of governance. Temporary quarters for the founding council.",
))
_register(BuildingDef(
    key="cottage", category="residential", cost=2000, maintenance=100, capacity=2,
    staff_required=0, staff_skill="", staff_min_level=0, unlock_population=0,
    sprite_key="cottage_1", variants=3, provides_housing=True,
    description="Small home for a young couple or single worker.",
))
_register(BuildingDef(
    key="house", category="residential", cost=4000, maintenance=200, capacity=4,
    staff_required=0, staff_skill="", staff_min_level=0, unlock_population=5,
    sprite_key="house_1", variants=3, provides_housing=True,
    description="Family home with room for 2 adults and 1-2 children.",
))
_register(BuildingDef(
    key="apartment", category="residential", cost=8000, maintenance=400, capacity=8,
    staff_required=0, staff_skill="", staff_min_level=0, unlock_population=15,
    sprite_key="apartment_1", variants=2, provides_housing=True,
    description="Multi-family dwelling for singles, couples, and small families.",
))
_register(BuildingDef(
    key="manor", category="residential", cost=15000, maintenance=750, capacity=2,
    staff_required=0, staff_skill="", staff_min_level=0, unlock_population=30,
    sprite_key="manor", provides_housing=True,
    effects={"tax_bonus": 0.2},
    description="Luxury residence that attracts wealthy individuals.",
))

# ── Commercial ───────────────────────────────────────────────────────
_register(BuildingDef(
    key="general_store", category="commercial", cost=3000, maintenance=150, capacity=0,
    staff_required=1, staff_skill="commerce", staff_min_level=0.1, unlock_population=5,
    sprite_key="store", provides_goods=True, revenue_per_hour=3,
    description="Basic goods shop serving the community.",
))
_register(BuildingDef(
    key="bakery", category="commercial", cost=4000, maintenance=200, capacity=0,
    staff_required=1, staff_skill="commerce", staff_min_level=0.1, unlock_population=8,
    unlock_buildings=["farm"],
    sprite_key="bakery", provides_food=True, revenue_per_hour=4,
    description="Produces bread and baked goods.",
))
_register(BuildingDef(
    key="butcher", category="commercial", cost=4000, maintenance=200, capacity=0,
    staff_required=1, staff_skill="commerce", staff_min_level=0.1, unlock_population=10,
    sprite_key="butcher", provides_food=True, revenue_per_hour=4,
    description="Meat shop.",
))
_register(BuildingDef(
    key="market", category="commercial", cost=6000, maintenance=300, capacity=0,
    staff_required=2, staff_skill="commerce", staff_min_level=0.2, unlock_population=15,
    sprite_key="market", provides_food=True, provides_goods=True, revenue_per_hour=8,
    description="Central food and goods hub.",
))
_register(BuildingDef(
    key="tavern", category="commercial", cost=5000, maintenance=250, capacity=0,
    staff_required=1, staff_skill="commerce", staff_min_level=0.1, unlock_population=10,
    sprite_key="tavern", provides_food=True, revenue_per_hour=5,
    effects={"social": 0.2},
    description="Social gathering place with food and drink.",
))
_register(BuildingDef(
    key="inn", category="commercial", cost=6000, maintenance=300, capacity=4,
    staff_required=1, staff_skill="commerce", staff_min_level=0.2, unlock_population=12,
    sprite_key="inn", provides_housing=True, revenue_per_hour=6,
    description="Temporary housing for newcomers. Generates income.",
))
_register(BuildingDef(
    key="blacksmith", category="commercial", cost=5000, maintenance=250, capacity=0,
    staff_required=1, staff_skill="craftsmanship", staff_min_level=0.2, unlock_population=12,
    sprite_key="blacksmith", provides_goods=True, revenue_per_hour=5,
    description="Tools, repairs, and metalwork.",
))
_register(BuildingDef(
    key="tailor", category="commercial", cost=4000, maintenance=200, capacity=0,
    staff_required=1, staff_skill="craftsmanship", staff_min_level=0.1, unlock_population=10,
    sprite_key="tailor", provides_goods=True, revenue_per_hour=4,
    description="Clothing and textile shop.",
))
_register(BuildingDef(
    key="bookstore", category="commercial", cost=3500, maintenance=175, capacity=0,
    staff_required=1, staff_skill="literacy", staff_min_level=0.3, unlock_population=15,
    sprite_key="bookstore", provides_goods=True, revenue_per_hour=3,
    effects={"education": 0.1},
    description="Books and educational materials.",
))

# ── Civic ────────────────────────────────────────────────────────────
_register(BuildingDef(
    key="school", category="civic", cost=8000, maintenance=400, capacity=0,
    staff_required=1, staff_skill="teaching", staff_min_level=0.2, unlock_population=10,
    sprite_key="school",
    effects={"education": 0.3},
    description="Basic education for children.",
))
_register(BuildingDef(
    key="library", category="civic", cost=6000, maintenance=300, capacity=0,
    staff_required=1, staff_skill="literacy", staff_min_level=0.3, unlock_population=20,
    unlock_buildings=["school"],
    sprite_key="library",
    effects={"education": 0.2},
    description="Advanced education and research.",
))
_register(BuildingDef(
    key="university", category="civic", cost=20000, maintenance=1000, capacity=0,
    staff_required=3, staff_skill="teaching", staff_min_level=0.5, unlock_population=40,
    unlock_buildings=["school", "library"],
    sprite_key="university",
    effects={"education": 0.5},
    description="Professional training and higher learning.",
))
_register(BuildingDef(
    key="clinic", category="civic", cost=6000, maintenance=300, capacity=0,
    staff_required=1, staff_skill="medicine", staff_min_level=0.2, unlock_population=10,
    sprite_key="clinic",
    effects={"health": 0.3},
    description="Basic healthcare.",
))
_register(BuildingDef(
    key="hospital", category="civic", cost=15000, maintenance=750, capacity=0,
    staff_required=2, staff_skill="medicine", staff_min_level=0.4, unlock_population=30,
    unlock_buildings=["clinic"],
    sprite_key="hospital",
    effects={"health": 0.5},
    description="Advanced healthcare facility.",
))
_register(BuildingDef(
    key="church", category="civic", cost=7000, maintenance=350, capacity=0,
    staff_required=1, staff_skill="leadership", staff_min_level=0.1, unlock_population=15,
    sprite_key="church",
    effects={"social": 0.2, "culture": 0.1},
    description="Spiritual needs and community gathering.",
))
_register(BuildingDef(
    key="courthouse", category="civic", cost=10000, maintenance=500, capacity=0,
    staff_required=2, staff_skill="leadership", staff_min_level=0.3, unlock_population=25,
    sprite_key="courthouse",
    effects={"safety": 0.2},
    description="Law and dispute resolution.",
))
_register(BuildingDef(
    key="fire_station", category="civic", cost=8000, maintenance=400, capacity=0,
    staff_required=2, staff_skill="", staff_min_level=0, unlock_population=20,
    sprite_key="fire_station",
    effects={"safety": 0.3},
    description="Fire prevention and disaster response.",
))
_register(BuildingDef(
    key="constabulary", category="civic", cost=5000, maintenance=250, capacity=0,
    staff_required=2, staff_skill="", staff_min_level=0, unlock_population=12,
    sprite_key="constabulary",
    effects={"safety": 0.4},
    description="Law enforcement and patrol.",
))
_register(BuildingDef(
    key="jail", category="civic", cost=4000, maintenance=200, capacity=0,
    staff_required=1, staff_skill="", staff_min_level=0, unlock_population=15,
    unlock_buildings=["constabulary"],
    sprite_key="jail",
    effects={"safety": 0.2},
    description="Crime reduction through incarceration.",
))

# ── Infrastructure ───────────────────────────────────────────────────
_register(BuildingDef(
    key="well", category="infrastructure", cost=1500, maintenance=75, capacity=0,
    staff_required=0, staff_skill="", staff_min_level=0, unlock_population=0,
    sprite_key="well",
    effects={"health": 0.1},
    description="Basic water supply.",
))
_register(BuildingDef(
    key="water_tower", category="infrastructure", cost=5000, maintenance=250, capacity=0,
    staff_required=0, staff_skill="", staff_min_level=0, unlock_population=20,
    unlock_buildings=["well"],
    sprite_key="water_tower",
    effects={"health": 0.2},
    description="Scaled water supply for growing population.",
))
_register(BuildingDef(
    key="granary", category="infrastructure", cost=3000, maintenance=150, capacity=0,
    staff_required=0, staff_skill="", staff_min_level=0, unlock_population=10,
    sprite_key="granary",
    description="Food storage. Famine buffer.",
))
_register(BuildingDef(
    key="warehouse", category="infrastructure", cost=4000, maintenance=200, capacity=0,
    staff_required=0, staff_skill="", staff_min_level=0, unlock_population=15,
    sprite_key="warehouse",
    description="Goods storage.",
))
_register(BuildingDef(
    key="trading_post", category="infrastructure", cost=8000, maintenance=400, capacity=0,
    staff_required=1, staff_skill="commerce", staff_min_level=0.3, unlock_population=20,
    sprite_key="trading_post", revenue_per_hour=10,
    description="External trade generates passive income.",
))

# ── Cultural ─────────────────────────────────────────────────────────
_register(BuildingDef(
    key="park", category="cultural", cost=2000, maintenance=100, capacity=0,
    staff_required=0, staff_skill="", staff_min_level=0, unlock_population=5,
    sprite_key="park",
    effects={"social": 0.1, "culture": 0.1},
    description="Recreation space. Free to visit.",
))
_register(BuildingDef(
    key="theater", category="cultural", cost=12000, maintenance=600, capacity=0,
    staff_required=2, staff_skill="arts", staff_min_level=0.3, unlock_population=30,
    sprite_key="theater", revenue_per_hour=5,
    effects={"culture": 0.4},
    description="Entertainment and cultural enrichment.",
))
_register(BuildingDef(
    key="museum", category="cultural", cost=10000, maintenance=500, capacity=0,
    staff_required=1, staff_skill="arts", staff_min_level=0.2, unlock_population=25,
    sprite_key="museum", revenue_per_hour=3,
    effects={"culture": 0.3, "education": 0.1},
    description="Culture and tourism attraction.",
))
_register(BuildingDef(
    key="festival_grounds", category="cultural", cost=5000, maintenance=250, capacity=0,
    staff_required=0, staff_skill="", staff_min_level=0, unlock_population=15,
    sprite_key="festival_grounds",
    effects={"social": 0.2, "culture": 0.2},
    description="Venue for festivals and community events.",
))

# ── Industrial ───────────────────────────────────────────────────────
_register(BuildingDef(
    key="farm", category="industrial", cost=5000, maintenance=250, capacity=0,
    staff_required=2, staff_skill="agriculture", staff_min_level=0.1, unlock_population=0,
    sprite_key="farm_1", provides_food=True, revenue_per_hour=4,
    description="Large-scale food production.",
))
_register(BuildingDef(
    key="lumber_mill", category="industrial", cost=6000, maintenance=300, capacity=0,
    staff_required=2, staff_skill="craftsmanship", staff_min_level=0.1, unlock_population=10,
    sprite_key="lumber_mill", revenue_per_hour=5,
    description="Raw materials: wood.",
))
_register(BuildingDef(
    key="quarry", category="industrial", cost=7000, maintenance=350, capacity=0,
    staff_required=2, staff_skill="craftsmanship", staff_min_level=0.1, unlock_population=15,
    sprite_key="quarry_1", revenue_per_hour=5,
    description="Raw materials: stone.",
))
_register(BuildingDef(
    key="windmill", category="industrial", cost=4000, maintenance=200, capacity=0,
    staff_required=1, staff_skill="agriculture", staff_min_level=0.1, unlock_population=10,
    unlock_buildings=["farm"],
    sprite_key="windmill", revenue_per_hour=3,
    description="Grain processing.",
))

# ── Sporadic event definitions ───────────────────────────────────────
SPORADIC_EVENTS = [
    {"id": "weather_rain", "prob": 0.04, "outdoor_only": True, "affects_all": True,
     "templates": ["A sudden rainstorm sweeps through {location}.",
                   "Dark clouds gather and rain begins to pour over {location}."]},
    {"id": "weather_sunny", "prob": 0.03, "outdoor_only": True, "affects_all": True,
     "templates": ["The sun breaks through the clouds, bathing {location} in warm light."]},
    {"id": "weather_wind", "prob": 0.02, "outdoor_only": True, "affects_all": True,
     "templates": ["A strong gust of wind blows through {location}, scattering leaves."]},
    {"id": "accident_trip", "prob": 0.005, "outdoor_only": False, "affects_all": False,
     "templates": ["{agent} tripped and scraped their knee at {location}."]},
    {"id": "found_item", "prob": 0.008, "outdoor_only": False, "affects_all": False,
     "templates": ["{agent} found a shiny coin on the ground at {location}."]},
    {"id": "illness_mild", "prob": 0.003, "outdoor_only": False, "affects_all": False,
     "templates": ["{agent} is feeling a bit under the weather."]},
    {"id": "surprise_visitor", "prob": 0.006, "outdoor_only": False, "affects_all": True,
     "templates": ["A traveling merchant arrives at {location} with exotic goods."]},
    {"id": "animal_encounter", "prob": 0.01, "outdoor_only": True, "affects_all": True,
     "templates": ["A stray cat wanders into {location} and rubs against {agent}'s leg."]},
    {"id": "gossip", "prob": 0.007, "outdoor_only": False, "affects_all": False,
     "templates": ["{agent} overheard an interesting rumor at {location}."]},
]

# ── Crisis definitions ───────────────────────────────────────────────
CRISIS_EVENTS = [
    {"id": "fire", "prob": 0.002, "mitigated_by": "fire_station",
     "description": "A fire breaks out! Without a fire station, a building may be destroyed."},
    {"id": "epidemic", "prob": 0.001, "mitigated_by": "clinic",
     "description": "Disease spreads through the town. Citizens fall ill."},
    {"id": "drought", "prob": 0.001, "mitigated_by": "granary",
     "description": "A drought reduces farm output. Food prices spike."},
    {"id": "crime_wave", "prob": 0.002, "mitigated_by": "constabulary",
     "description": "Crime rises. Theft and unrest increase."},
    {"id": "economic_downturn", "prob": 0.001, "mitigated_by": "trading_post",
     "description": "Trade income drops. Businesses may close."},
]

# ── Council ──────────────────────────────────────────────────────────
COUNCIL_ROLES = {
    "mayor": {"domain": "governance", "title": "Mayor"},
    "sheriff": {"domain": "security", "title": "Sheriff"},
    "superintendent": {"domain": "education", "title": "Superintendent"},
    "doctor": {"domain": "health", "title": "Doctor"},
    "treasurer": {"domain": "finances", "title": "Treasurer"},
}

COUNCIL_MEETING_INTERVAL_HOURS = 168  # 1 sim-week

# ── Economy ──────────────────────────────────────────────────────────
STARTING_TREASURY = 50000
STARTING_WALLET = 100

TAX_RATES_DEFAULT = {
    "property": 0.05,   # per housing unit per year
    "business": 0.08,   # per commercial building per year
    "income": 0.10,     # percentage of citizen earnings
}

FOOD_COST = 10
RENT_COST = 5
GOODS_COST = 15

INCOME_EMPLOYED = 8       # per sim-hour
INCOME_ODD_JOBS = 2       # per sim-hour (unemployed adult)
INCOME_ELDER = 3          # per sim-hour (retired)

COUNCIL_SALARY = 100      # per sim-day

ENTREPRENEURSHIP_SAVINGS_THRESHOLD = 500
ENTREPRENEURSHIP_COMMERCE_SKILL = 0.4

# ── Population ───────────────────────────────────────────────────────
MAX_POPULATION = 200
IMMIGRATION_CHECK_INTERVAL_HOURS = 48
IMMIGRATION_BASE_PROBABILITY = 0.3

# Attractors: building types that attract specific immigrant profiles
HOUSING_IMMIGRANT_PROFILES = {
    "cottage": {"adults": 2, "children": 0, "description": "Young couple or single worker"},
    "house": {"adults": 2, "children": (1, 2), "description": "Small family"},
    "apartment": {"adults": (1, 2), "children": (0, 2), "description": "Mixed: singles, couples, small families"},
    "manor": {"adults": 2, "children": 0, "description": "Wealthy individual/couple with specific skills"},
    "inn": {"adults": 1, "children": 0, "description": "Temporary visitor who may settle"},
}

# ── NavGrid / movement ─────────────────────────────────────────────
CELL_SIZE = 16
WORLD_WIDTH = 2400
WORLD_HEIGHT = 1600
GRID_WIDTH = WORLD_WIDTH // CELL_SIZE    # 150
GRID_HEIGHT = WORLD_HEIGHT // CELL_SIZE  # 100

AGENT_SPEED_DEFAULT = 3.0
AGENT_SPEED_CHILD = 2.0
AGENT_SPEED_ELDER = 1.5

# ── Epoch tuning ─────────────────────────────────────────────────────
EPOCH_INTERVAL_HOURS = 24.0
BRIDGE_THRESHOLD = 0.3
CULTURE_CONFIDENCE_MIN = 0.5

# ── Petition thresholds ────────────────────────────────────────────
PETITION_SIMILARITY_THRESHOLD = 0.6
PETITION_CONFIDENCE_THRESHOLD = 0.4
PETITION_MAX_AGE_HOURS = 720

# ── Game state phases ────────────────────────────────────────────────
PHASE_OPENING_PLACE_HALL = "opening_place_hall"
PHASE_OPENING_CHOOSE_MAYOR = "opening_choose_mayor"
PHASE_OPENING_COUNCIL = "opening_council"
PHASE_GAMEPLAY = "gameplay"
PHASE_GAME_OVER = "game_over"

# ── Failure conditions ───────────────────────────────────────────────
BANKRUPTCY_THRESHOLD = 0
MIN_POPULATION_SURVIVAL = 5
SATISFACTION_EXODUS_THRESHOLD = 0.2
SATISFACTION_EXODUS_HOURS = 168  # 1 sim-week below threshold triggers exodus
