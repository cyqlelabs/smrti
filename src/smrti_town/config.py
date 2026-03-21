"""All constants for the smrti-town simulation."""

from __future__ import annotations

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

# ── Drive rates (per sim-hour) ───────────────────────────────────────
HUNGER_RATE = 2
ENERGY_DRAIN_RATE = 1
SOCIAL_RATE = 1
CURIOSITY_RATE = 1
DUTY_RATE = 3           # only during work hours
ROMANCE_RATE = 1        # adults only

# ── Drive thresholds (trigger action consideration) ──────────────────
HUNGER_THRESHOLD = 70
ENERGY_LOW_THRESHOLD = 20
SOCIAL_THRESHOLD = 60
CURIOSITY_THRESHOLD = 50
DUTY_THRESHOLD = 40
ROMANCE_THRESHOLD = 50

# ── Drive clamps ─────────────────────────────────────────────────────
DRIVE_MAX = 100
DRIVE_MIN = 0

# ── Drive resets (value set after satisfying action) ─────────────────
HUNGER_RESET = 0
ENERGY_RESET = 100
SOCIAL_RESET_AMOUNT = 30   # subtracted after conversation
CURIOSITY_RESET_AMOUNT = 25
ROMANCE_RESET_AMOUNT = 30

# ── Life stages ──────────────────────────────────────────────────────
LIFE_STAGES = {
    "infant": {
        "age_range": (0, 5),
        "drives": ["hunger", "energy", "social"],
        "can_move": False,
        "can_talk": False,
        "can_reproduce": False,
        "energy_decay_mult": 0.5,
        "schedule": None,
    },
    "child": {
        "age_range": (5, 18),
        "drives": ["hunger", "energy", "social", "curiosity"],
        "can_move": True,
        "can_talk": True,
        "can_reproduce": False,
        "energy_decay_mult": 0.8,
        "schedule": {"school": (8, 14)},
    },
    "adult": {
        "age_range": (18, 65),
        "drives": ["hunger", "energy", "social", "curiosity", "duty", "romance"],
        "can_move": True,
        "can_talk": True,
        "can_reproduce": True,
        "energy_decay_mult": 1.0,
        "schedule": {"work": (8, 17)},
    },
    "elder": {
        "age_range": (65, 200),
        "drives": ["hunger", "energy", "social", "curiosity"],
        "can_move": True,
        "can_talk": True,
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
# NOTE: No marriage requirement for reproduction. Romantic or close_friend
# with romance_drive is enough.
RELATIONSHIP_GATES = {
    "acquaintance": {},
    "friend": {"interaction_count": 5, "valence": 0.2},
    "close_friend": {"friend_lti": 0.5, "valence": 0.4, "shared_episodes": 3},
    "romantic": {"close_friend_lti": 0.6, "mutual_valence": 0.5, "romance_drive": 50},
    "married": {"romantic_lti": 0.7, "mutual_valence": 0.6, "cohabitation_years": 1},
}

# ── Reproduction gate ────────────────────────────────────────────────
# No marriage requirement — romantic relationship or close_friend + romance
# drive is sufficient.
REPRODUCTION_GATE = {
    "min_relationship": "romantic",      # romantic OR close_friend with romance_drive
    "also_allow_close_friend_romance": True,
    "both_energy": 70,
    "life_stage": "adult",
    "min_relationship_years": 1,
}

# ── Death parameters ─────────────────────────────────────────────────
ELDER_DEATH_PROB_PER_TICK = 0.002   # per year past 65, per tick
STARVATION_HOURS = 48               # hours at energy=0 before death
DEATH_LOW_ENERGY_MULT = 2.0

# ── Personality inheritance ──────────────────────────────────────────
PERSONALITY_PARAMS = [
    "confidence_decay_rate",
    "confidence_update_lr",
    "min_confidence_to_surface",
    "sti_decay_rate",
    "sti_boost_on_access",
    "sti_propagation_factor",
    "lti_promotion_threshold",
    "valence_weight",
    "valence_propagation",
    "mood_inertia",
    "w_similarity",
    "w_sti",
    "w_confidence",
    "w_lti",
    "w_valence",
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

# Stress-boosted mutation variance
STRESS_VARIANCE_BASE = 1.0
STRESS_VARIANCE_MAX_MULT = 3.0

# ── Behavioural personality traits ───────────────────────────────────
# Heritable traits (0.0–1.0) that modulate the rule-based decision
# system. They sit alongside the Smrti personality hyperparameters
# but govern *behaviour*, not memory-engine tuning.
#
# Each trait modulates specific decision paths:
#   shyness      — inverts social drive probability; shy agents avoid crowds
#   proactivity  — increases probability of acting on low-urgency drives
#   leadership   — when co-located, leader initiates conversations first
#   laziness     — reduces probability of duty/work actions
#   adventurous  — increases wander probability; seeks unvisited places
#   nurturing    — boosts positive valence of social interactions
#   stubbornness — reduces probability of changing current action/location
#   creativity   — boosts curiosity-driven actions and diverse dialogue topics
TRAIT_NAMES = [
    "shyness",
    "proactivity",
    "leadership",
    "laziness",
    "adventurous",
    "nurturing",
    "stubbornness",
    "creativity",
]

TRAIT_BOUNDS = {t: (0.0, 1.0) for t in TRAIT_NAMES}

# Default trait profiles derived from personality presets
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

# ── Personality → agent behaviour mapping ────────────────────────────
# Maps personality presets to behavioural biases used by the rule-based
# decision system. Higher value = stronger preference for that action
# category.
PERSONALITY_ACTION_BIAS = {
    "balanced": {
        "social": 0.5, "curiosity": 0.5, "duty": 0.5,
        "romance": 0.5, "wander": 0.3,
    },
    "analytical": {
        "social": 0.2, "curiosity": 0.9, "duty": 0.7,
        "romance": 0.2, "wander": 0.1,
    },
    "curious": {
        "social": 0.6, "curiosity": 0.9, "duty": 0.3,
        "romance": 0.4, "wander": 0.7,
    },
    "empathetic": {
        "social": 0.9, "curiosity": 0.4, "duty": 0.4,
        "romance": 0.7, "wander": 0.3,
    },
    "maverick": {
        "social": 0.5, "curiosity": 0.7, "duty": 0.2,
        "romance": 0.5, "wander": 0.9,
    },
    "deterministic": {
        "social": 0.3, "curiosity": 0.6, "duty": 0.9,
        "romance": 0.2, "wander": 0.1,
    },
}

# ── Sporadic event definitions ───────────────────────────────────────
# Each entry: (event_id, base probability per routine tick, description template)
SPORADIC_EVENTS = [
    {
        "id": "weather_rain",
        "prob": 0.04,
        "outdoor_only": True,
        "affects_all": True,
        "templates": [
            "A sudden rainstorm sweeps through {location}.",
            "Dark clouds gather and rain begins to pour over {location}.",
        ],
    },
    {
        "id": "weather_sunny",
        "prob": 0.03,
        "outdoor_only": True,
        "affects_all": True,
        "templates": [
            "The sun breaks through the clouds, bathing {location} in warm light.",
            "A beautiful clear sky stretches over {location}.",
        ],
    },
    {
        "id": "weather_wind",
        "prob": 0.02,
        "outdoor_only": True,
        "affects_all": True,
        "templates": [
            "A strong gust of wind blows through {location}, scattering leaves.",
        ],
    },
    {
        "id": "accident_trip",
        "prob": 0.005,
        "outdoor_only": False,
        "affects_all": False,
        "templates": [
            "{agent} tripped and scraped their knee at {location}.",
            "{agent} stumbled over a loose stone at {location}.",
        ],
    },
    {
        "id": "found_item",
        "prob": 0.008,
        "outdoor_only": False,
        "affects_all": False,
        "templates": [
            "{agent} found a shiny coin on the ground at {location}.",
            "{agent} discovered a forgotten book on a bench at {location}.",
            "{agent} picked up a pretty wildflower at {location}.",
        ],
    },
    {
        "id": "illness_mild",
        "prob": 0.003,
        "outdoor_only": False,
        "affects_all": False,
        "templates": [
            "{agent} is feeling a bit under the weather.",
            "{agent} has come down with a mild headache.",
        ],
    },
    {
        "id": "surprise_visitor",
        "prob": 0.006,
        "outdoor_only": False,
        "affects_all": True,
        "templates": [
            "A traveling merchant arrives at {location} with exotic goods.",
            "An old friend nobody expected shows up at {location}.",
            "A wandering musician sets up at {location} and begins to play.",
        ],
    },
    {
        "id": "animal_encounter",
        "prob": 0.01,
        "outdoor_only": True,
        "affects_all": True,
        "templates": [
            "A stray cat wanders into {location} and rubs against {agent}'s leg.",
            "A flock of birds lands in {location}, chirping loudly.",
            "A friendly dog appears at {location}, wagging its tail.",
            "A squirrel darts across {location}, cheeks stuffed with acorns.",
        ],
    },
    {
        "id": "gossip",
        "prob": 0.007,
        "outdoor_only": False,
        "affects_all": False,
        "templates": [
            "{agent} overheard an interesting rumor at {location}.",
            "Someone whispered a piece of gossip to {agent} at {location}.",
        ],
    },
    {
        "id": "power_outage",
        "prob": 0.002,
        "outdoor_only": False,
        "affects_all": True,
        "templates": [
            "The lights flicker and go out at {location}. A power outage!",
        ],
    },
    {
        "id": "festival",
        "prob": 0.005,
        "outdoor_only": True,
        "affects_all": True,
        "templates": [
            "Festive decorations appear at {location}. A spontaneous celebration breaks out!",
            "Someone starts playing music at {location} and others join in dancing.",
        ],
    },
    {
        "id": "strange_noise",
        "prob": 0.004,
        "outdoor_only": False,
        "affects_all": True,
        "templates": [
            "A strange noise echoes through {location}.",
            "An unexplained rumble shakes the walls of {location}.",
        ],
    },
]

# ── Spatial / topology ───────────────────────────────────────────────
OUTDOOR_PLACES = {"Central_Park", "Elm_Street", "Main_Street", "Town_Millbrook"}

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

# ── Epoch tuning ─────────────────────────────────────────────────────
EPOCH_INTERVAL_HOURS = 24.0    # run epoch every 24 sim-hours
BRIDGE_THRESHOLD = 0.3         # higher than default 0.1 to prevent base-ontology explosion
CULTURE_CONFIDENCE_MIN = 0.5   # only promote bridge atoms above this confidence

# ── Population cap ───────────────────────────────────────────────────
MAX_POPULATION = 20            # soft cap — reduces fertility when exceeded

# ── Conversation generation ──────────────────────────────────────────
GREETINGS = [
    "Hey {target}, how are you?",
    "Good to see you, {target}!",
    "Hello {target}! What's new?",
    "{target}! I was hoping to run into you.",
]

SMALL_TALK = [
    "The weather has been interesting lately, hasn't it?",
    "I've been thinking a lot about things recently.",
    "Have you heard anything new around town?",
    "This is a nice spot, isn't it?",
    "I wonder what's happening at the market today.",
    "It's been a while since we last talked.",
]

ROMANTIC_LINES = [
    "I really enjoy spending time with you, {target}.",
    "Being here with you makes everything better, {target}.",
    "You know, {target}, I feel so comfortable around you.",
    "I was thinking about you earlier, {target}.",
]

FOOD_TOPICS = [
    "I'm starving! Time for a good meal.",
    "The food here always hits the spot.",
    "I could really go for something to eat.",
]

CURIOSITY_TOPICS = [
    "I read the most fascinating thing today.",
    "I've been curious about how things work around here.",
    "Do you know anything about the history of this place?",
    "I learned something interesting recently.",
]
