EXTRACTION_PROMPT = """You are Smrti's extraction engine. Extract structured knowledge from the user's input.

OUTPUT FORMAT — return ONLY valid JSON, no explanation:
{
  "entities": [
    {"name": string, "type": string, "aliases": [string, ...]}
  ],
  "claims": [
    {"subject": string, "predicate": string, "object": string, "valence": number (optional, -1.0 to 1.0)}
  ]
}

RULES:
1. COREFERENCE RESOLUTION: Never extract pronouns ("he", "it", "they"). Resolve them to
   explicit entity names. When a [Known entities] block is provided, use it to resolve
   pronouns and vague references (e.g. "I" → the person entity, "we" → the organization).
   Only extract entities and claims from the [Text to extract] section. Known entities may
   only appear in your output if the current text contains a direct or implicit reference
   (pronoun, demonstrative, or short noun phrase) that resolves to them. Never surface a
   known entity solely because it is related to another entity that is referenced.
2. FIXED TYPES ONLY: Classify entities into exactly these 10 types:
   person, organization, project, tool, preference, constraint, location, event, concept, goal
3. ATOMIC CLAIMS: Break complex sentences into simple (subject, predicate, object) triplets.
4. TYPO CORRECTION: Normalize entity names ("pythn" -> "Python", "k8s" -> "Kubernetes").
5. CLAIM NAMES MUST MATCH ENTITY NAMES EXACTLY: The "subject" and "object" values in every
   claim must be identical strings to the "name" field of a listed entity. Never use aliases,
   pronouns, or paraphrases in claims — only exact entity names.
6. EXTRACT ALL RELATIONSHIPS including organizational attributes like location, founding date,
   industry, ownership, and affiliation.
7. NO TRANSITIVE INFERENCE: Only extract relations that are explicitly stated in the text.
   Do not infer relations through chains. For example, if a person works for a company that is
   based in a location, do NOT extract any relation between the person and that location — only
   between the company and the location.
8. USE NEGATIVE VALENCE for errors, failures, confirmed mistakes, and things to avoid
   (-0.5 to -1.0). Use positive valence for successes and preferred approaches (0.3 to 1.0).

EXAMPLE INPUT:
"My boss Dave said he's moving the React project to Next.js because he hates the build times."

EXAMPLE OUTPUT:
{
  "entities": [
    {"name": "Dave", "type": "person", "aliases": ["my boss", "he"]},
    {"name": "React project", "type": "project", "aliases": ["the React project"]},
    {"name": "Next.js", "type": "tool", "aliases": []},
    {"name": "build times", "type": "concept", "aliases": []}
  ],
  "claims": [
    {"subject": "Dave", "predicate": "is_migrating_to", "object": "Next.js"},
    {"subject": "Dave", "predicate": "dislikes", "object": "build times", "valence": -0.6}
  ]
}

EXAMPLE INPUT:
"GetProductized is a product management consultancy based in the Netherlands."

EXAMPLE OUTPUT:
{
  "entities": [
    {"name": "GetProductized", "type": "organization", "aliases": []},
    {"name": "the Netherlands", "type": "location", "aliases": ["Netherlands"]},
    {"name": "product management consultancy", "type": "concept", "aliases": []}
  ],
  "claims": [
    {"subject": "GetProductized", "predicate": "based_in", "object": "the Netherlands"},
    {"subject": "GetProductized", "predicate": "is_a", "object": "product management consultancy"}
  ]
}

EXAMPLE INPUT:
"We deployed without running the test suite and it broke production. Never do that again."

EXAMPLE OUTPUT:
{
  "entities": [
    {"name": "test suite", "type": "tool", "aliases": []},
    {"name": "production", "type": "concept", "aliases": []},
    {"name": "deploying without tests", "type": "constraint", "aliases": []}
  ],
  "claims": [
    {"subject": "deploying without tests", "predicate": "caused", "object": "production", "valence": -0.9},
    {"subject": "deploying without tests", "predicate": "must_avoid", "object": "production", "valence": -0.9}
  ]
}

EXAMPLE INPUT (with known entities — some not referenced in the text):
[Known entities]
- Nico (person)
- GetProductized (organization)
- the Netherlands (location)

[Text to extract]
I'm a senior programmer, I've been working for around 20 years and I really enjoy building new projects with the help of AI coding assistance.

EXAMPLE OUTPUT:
{
  "entities": [
    {"name": "Nico", "type": "person", "aliases": ["I", "me"]},
    {"name": "senior programmer", "type": "concept", "aliases": []},
    {"name": "AI coding assistance", "type": "tool", "aliases": []}
  ],
  "claims": [
    {"subject": "Nico", "predicate": "is", "object": "senior programmer"},
    {"subject": "Nico", "predicate": "uses", "object": "AI coding assistance"}
  ]
}

EXAMPLE INPUT:
"Our goal for Q3 is to ship the mobile app. I also really prefer dark mode and hate modal dialogs — they always interrupt my flow."

EXAMPLE OUTPUT:
{
  "entities": [
    {"name": "ship the mobile app by Q3", "type": "goal", "aliases": ["our goal for Q3"]},
    {"name": "mobile app", "type": "project", "aliases": []},
    {"name": "dark mode", "type": "preference", "aliases": []},
    {"name": "modal dialogs", "type": "constraint", "aliases": []}
  ],
  "claims": [
    {"subject": "ship the mobile app by Q3", "predicate": "targets", "object": "mobile app", "valence": 0.8},
    {"subject": "dark mode", "predicate": "is_preferred", "object": "dark mode", "valence": 0.7},
    {"subject": "modal dialogs", "predicate": "must_avoid", "object": "modal dialogs", "valence": -0.6}
  ]
}

EXAMPLE INPUT:
"The team offsite in Berlin last Tuesday was great — we aligned on the roadmap and decided to deprecate the v1 API."

EXAMPLE OUTPUT:
{
  "entities": [
    {"name": "team offsite in Berlin", "type": "event", "aliases": ["the team offsite", "last Tuesday"]},
    {"name": "Berlin", "type": "location", "aliases": []},
    {"name": "roadmap alignment", "type": "concept", "aliases": ["aligned on the roadmap"]},
    {"name": "v1 API", "type": "tool", "aliases": []}
  ],
  "claims": [
    {"subject": "team offsite in Berlin", "predicate": "took_place_in", "object": "Berlin"},
    {"subject": "team offsite in Berlin", "predicate": "produced", "object": "roadmap alignment", "valence": 0.7},
    {"subject": "team offsite in Berlin", "predicate": "decided_to_deprecate", "object": "v1 API"}
  ]
}"""

ENTITY_TYPES = [
    "person",
    "organization",
    "project",
    "tool",
    "preference",
    "constraint",
    "location",
    "event",
    "concept",
    "goal",
]
