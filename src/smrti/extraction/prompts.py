EXTRACTION_PROMPT = """Extract knowledge from user text. Return ONLY valid JSON — no prose, no fences.

FORMAT:
{"entities":[{"name":string,"type":string,"aliases":[string]}],
 "claims":[{"subject":string,"predicate":string,"object":string,"valence":number}]}

ENTITY TYPES (exactly these 10): person · organization · project · tool · preference · constraint · location · event · concept · goal
"valence" is optional; omit when neutral. Pets/animals → "concept".

━━━ RULES — follow all; top rules are highest priority ━━━

GOAL EXTRACTION — ALWAYS check for these markers FIRST before extracting anything else.
When the text contains durable-objective language ("my goal is", "my primary goal", "I aim to",
"I plan to", "I intend to", "I'm working toward", "my objective is"), you MUST:
  1. Create a SEPARATE entity with type "goal" whose label captures the objective concisely
     (e.g. "launch open-source platform for artists" — NOT "open-source platform" alone)
  2. Also extract the underlying project/tool as its own entity (type "project" or "tool")
  3. Produce these two claims:
       person→has_goal→goal_entity        (if a person is known)
       goal_entity→targets→project_entity
  ✓ TWO entities: {"name":"launch X for Y","type":"goal"} + {"name":"X","type":"project"}
  ✗ ONE entity: {"name":"X","type":"project"}   ← WRONG — you lost the goal
Not for transient desires: "I want a coffee", "I'd like to understand X".

PERSON ANCHORING — "I/my/me" resolves to the known person entity. Every trait, belief,
opinion, value, hobby, fear, or biographical fact MUST become a claim with the person as subject.
  ✓ Marco→has_hobby→cycling     ✗ cycling→is_hobby→cycling
  ✓ Sam→believes→simplicity     ✗ simplicity→is_believed→simplicity

BELIEF/VALUE TYPING — When the text expresses a belief, value, conviction, or personal
principle ("I believe", "I value", "I think that", "I hold that", "X is important to me",
"X is essential", "I'm a firm believer"), the object of that belief MUST be type "preference",
NOT "concept". Things someone avoids or rejects ("I hate X", "never do X", "X is harmful")
→ type "constraint". Only use "concept" for neutral, factual entities with no expressed stance.
  ✓ {"name":"intellectual humility","type":"preference"}  ← person believes in it
  ✗ {"name":"intellectual humility","type":"concept"}     ← WRONG, lost the belief

FIRST MESSAGE — When NO [Known entities] block is present and the text
introduces a name ("I am X", "my name is X"), the entity name MUST be
the actual name (X), not pronouns. Put pronouns in "aliases".

EXACT NAMES IN CLAIMS — subject and object must be character-identical to a listed entity name.
Never use pronouns, aliases, or paraphrases in claims.

KNOWN ENTITIES — Surface a known entity only when the text directly references it (pronoun,
demonstrative, noun phrase). Never surface it merely because it relates to something mentioned.

VALENCE — negative (−0.5 to −1.0): errors, fears, avoidance, displeasure.
           positive (0.3 to 1.0): preferences, successes, enjoyment.

PRONOUNS — Never output pronouns as entity names. Resolve he/she/it/they to the named entity.
ATOMIC — one fact per claim triplet.
NO INFERENCE — extract only explicitly stated facts; no transitive chains, no invented entities.
NO METAPHOR — figurative language ("she's the real boss" about a cat) ≠ organizational fact.
NORMALIZE — fix typos in entity names ("pythn"→"Python", "k8s"→"Kubernetes").

━━━ EXAMPLES ━━━

IN: "My boss Dave said he's moving the React project to Next.js because he hates the build times."
OUT:
{"entities":[
  {"name":"Dave","type":"person","aliases":["my boss","he"]},
  {"name":"React project","type":"project","aliases":["the React project"]},
  {"name":"Next.js","type":"tool","aliases":[]},
  {"name":"build times","type":"concept","aliases":[]}
],"claims":[
  {"subject":"Dave","predicate":"is_migrating_to","object":"Next.js"},
  {"subject":"Dave","predicate":"dislikes","object":"build times","valence":-0.6}
]}

IN: "We deployed without running the test suite and it broke production. Never do that again."
OUT:
{"entities":[
  {"name":"test suite","type":"tool","aliases":[]},
  {"name":"production","type":"concept","aliases":[]},
  {"name":"deploying without tests","type":"constraint","aliases":[]}
],"claims":[
  {"subject":"deploying without tests","predicate":"caused","object":"production","valence":-0.9},
  {"subject":"deploying without tests","predicate":"must_avoid","object":"production","valence":-0.9}
]}

IN:
[Known entities]
- Elias (person)
[Text to extract]
My primary goal right now is to launch an open-source platform that helps independent artists manage their workflows.
OUT:
{"entities":[
  {"name":"Elias","type":"person","aliases":["I","my"]},
  {"name":"launch open-source platform for independent artists","type":"goal","aliases":["my primary goal"]},
  {"name":"open-source platform","type":"project","aliases":[]}
],"claims":[
  {"subject":"Elias","predicate":"has_goal","object":"launch open-source platform for independent artists","valence":0.8},
  {"subject":"launch open-source platform for independent artists","predicate":"targets","object":"open-source platform","valence":0.8}
]}

IN:
[Known entities]
- Nico (person) · GetProductized (organization) · the Netherlands (location)
[Text to extract]
I'm a senior programmer, I've been working for around 20 years and I really enjoy building new projects with the help of AI coding assistance.
OUT:
{"entities":[
  {"name":"Nico","type":"person","aliases":["I","me"]},
  {"name":"senior programmer","type":"concept","aliases":[]},
  {"name":"AI coding assistance","type":"tool","aliases":[]}
],"claims":[
  {"subject":"Nico","predicate":"is","object":"senior programmer"},
  {"subject":"Nico","predicate":"uses","object":"AI coding assistance"}
]}

IN:
[Known entities]
- Alice (person)
[Text to extract]
"Our goal for Q3 is to ship the mobile app. I also really prefer dark mode and hate modal dialogs — they always interrupt my flow."
OUT:
{"entities":[
  {"name":"Alice","type":"person","aliases":["I","my"]},
  {"name":"ship the mobile app by Q3","type":"goal","aliases":["our goal for Q3"]},
  {"name":"mobile app","type":"project","aliases":[]},
  {"name":"dark mode","type":"preference","aliases":[]},
  {"name":"modal dialogs","type":"constraint","aliases":[]}
],"claims":[
  {"subject":"Alice","predicate":"has_goal","object":"ship the mobile app by Q3","valence":0.8},
  {"subject":"ship the mobile app by Q3","predicate":"targets","object":"mobile app","valence":0.8},
  {"subject":"Alice","predicate":"prefers","object":"dark mode","valence":0.7},
  {"subject":"Alice","predicate":"dislikes","object":"modal dialogs","valence":-0.6}
]}

IN:
[Known entities]
- Sam (person)
[Text to extract]
"I'm a firm believer that simplicity is the ultimate sophistication. I think most problems come down to poor communication. I also try not to waste people's time."
OUT:
{"entities":[
  {"name":"Sam","type":"person","aliases":["I","my"]},
  {"name":"simplicity","type":"preference","aliases":[]},
  {"name":"ultimate sophistication","type":"concept","aliases":[]},
  {"name":"poor communication","type":"concept","aliases":[]},
  {"name":"people's time","type":"preference","aliases":[]}
],"claims":[
  {"subject":"Sam","predicate":"believes","object":"simplicity","valence":0.6},
  {"subject":"simplicity","predicate":"is","object":"ultimate sophistication"},
  {"subject":"Sam","predicate":"believes_root_cause_is","object":"poor communication"},
  {"subject":"Sam","predicate":"values","object":"people's time","valence":0.5}
]}

IN:
[Known entities]
- Marco (person)
[Text to extract]
"My hobbies are cycling and photography. I have a cat named Luna — she's a bit judgmental. I have an irrational fear of escalators."
OUT:
{"entities":[
  {"name":"Marco","type":"person","aliases":["I","my","me"]},
  {"name":"cycling","type":"concept","aliases":[]},
  {"name":"photography","type":"concept","aliases":[]},
  {"name":"Luna","type":"concept","aliases":["she"]},
  {"name":"escalators","type":"constraint","aliases":[]}
],"claims":[
  {"subject":"Marco","predicate":"has_hobby","object":"cycling","valence":0.5},
  {"subject":"Marco","predicate":"has_hobby","object":"photography","valence":0.5},
  {"subject":"Marco","predicate":"owns_pet","object":"Luna"},
  {"subject":"Marco","predicate":"has_fear_of","object":"escalators","valence":-0.7}
]}

IN: "Hi! I'm Elara, a systems strategist focused on organizational design."
OUT:
{"entities":[
  {"name":"Elara","type":"person","aliases":["I","I'm"]},
  {"name":"systems strategist","type":"concept","aliases":[]},
  {"name":"organizational design","type":"concept","aliases":[]}
],"claims":[
  {"subject":"Elara","predicate":"is","object":"systems strategist"},
  {"subject":"Elara","predicate":"focuses_on","object":"organizational design"}
]}"""

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

AGENT_EXTRACTION_PROMPT = """Extract agent self-knowledge from agent responses. Return ONLY valid JSON.

{"entities":[{"name":string,"type":string,"aliases":[string]}],
 "claims":[{"subject":string,"predicate":string,"object":string,"valence":number}]}

DEFAULT: return {"entities":[],"claims":[]} — the correct output for most agent turns.

EXTRACT ONLY (past-tense, confirmed facts about the agent itself):
- Mistakes acknowledged: "I was wrong about X", "I gave incorrect advice on Y"
- Self-corrections: "I should not have...", "that was an error"
- Executed actions: "I created file X", "I deleted Y", "I ran command Z"

NEVER EXTRACT — return empty for ANY of these patterns:
- Offers: "I can help...", "I'd be happy to...", "I can assist..."
- Capability lists: "Here's what I can do:", "Here's what I can help with:", "Here's how I can..."
- Proposals or questions: "Would you like...", "Shall we...", "Let's..."
- Numbered or bulleted action plans describing future steps
- Acknowledgements of user goals ("I understand your goal...", "These are meaningful aspirations...")
- Claims about the user, praise, or general commentary

RULE: When in doubt, return empty. Only extract past-tense mistakes and completed actions.

IN: "I see. I incorrectly told you that Python 2 was still maintained — that's wrong, Python 2 reached end-of-life in 2020. I've now updated the documentation."
OUT:
{"entities":[
  {"name":"Python 2 maintenance claim","type":"constraint","aliases":[]},
  {"name":"documentation","type":"tool","aliases":[]}
],"claims":[
  {"subject":"Python 2 maintenance claim","predicate":"was_incorrect","object":"Python 2 maintenance claim","valence":-0.8}
]}

IN: "I understand your goal to restore the lighthouse and achieve fluency in a third language. These are meaningful aspirations. Here's what I can help with:
1. Language learning tools and resources.
2. Project management assistance."
OUT: {"entities":[],"claims":[]}

IN: "I'd be happy to help you with your project. Here's what I can do:
1. Research best practices.
2. Break it into manageable milestones."
OUT: {"entities":[],"claims":[]}"""


CLAIMS_ONLY_PROMPT = """Extract relationship claims between pre-extracted entities. You MAY also emit NEW goal entities that the NER missed. Return ONLY valid JSON — no prose, no fences.

FORMAT:
{"entities":[{"name":string,"type":"goal"}],
 "claims":[{"subject":string,"predicate":string,"object":string,"valence":number}]}

"entities" is OPTIONAL — include it ONLY when you detect a durable objective not already
listed below. The only allowed type for new entities is "goal".

PRE-EXTRACTED ENTITIES:
{entities_block}

RULES:
- subject and object in claims MUST be character-identical to a pre-extracted entity name
  OR a new goal entity you emitted in "entities"
- One fact per claim triplet (atomic)
- "valence" is optional; omit when neutral
  negative (−0.5 to −1.0): errors, fears, avoidance, displeasure
  positive (0.3 to 1.0): preferences, successes, enjoyment
- Extract only explicitly stated facts; no inference or invention
- No metaphor — figurative language ≠ literal fact
- If no clear relationships exist, return {"claims":[]}
- PERSON ANCHORING — when a person entity is listed, they must be the subject of claims
  that describe their goals, intentions, preferences, or actions. Never leave the person
  entity disconnected if the text describes something they intend, want, or are doing.
- GOAL EXTRACTION — when the text expresses a durable objective (goal, aim, plan, intention,
  ambition, mission, aspiration) and no goal entity is already listed above:
  1. Emit a NEW entity with type "goal" whose name captures the objective as an action phrase
     (e.g. "establish community permaculture project", "automate data workflows")
  2. Emit a `has_goal` claim from person→goal
  3. If a related project/tool entity exists, also emit goal→targets→project
  Not for transient desires ("I want a coffee", "I'd like to understand X").
- GOAL CLAIMS — when a goal or project entity is already listed alongside a person, emit a
  `has_goal` claim from person→goal and/or a `works_on` claim from person→project.

EXAMPLES:

Entities: Dave (person), Next.js (tool), build times (concept)
Text: "Dave said he's moving to Next.js because he hates the build times."
OUT:
{"claims":[
  {"subject":"Dave","predicate":"is_migrating_to","object":"Next.js"},
  {"subject":"Dave","predicate":"dislikes","object":"build times","valence":-0.6}
]}

Entities: test suite (tool), production (concept), deploying without tests (constraint)
Text: "We deployed without running the test suite and it broke production."
OUT:
{"claims":[
  {"subject":"deploying without tests","predicate":"caused","object":"production","valence":-0.9},
  {"subject":"deploying without tests","predicate":"must_avoid","object":"production","valence":-0.9}
]}

Entities: Elias (person), community-led permaculture project (project), food-insecure neighborhoods (location)
Text: "My primary goal is to establish a community-led permaculture project that provides fresh produce to food-insecure neighborhoods."
OUT:
{"entities":[
  {"name":"establish community permaculture project","type":"goal"}
],"claims":[
  {"subject":"Elias","predicate":"has_goal","object":"establish community permaculture project","valence":0.8},
  {"subject":"establish community permaculture project","predicate":"targets","object":"community-led permaculture project","valence":0.7}
]}

Entities: Alice (person), mobile app (project), dark mode (preference)
Text: "Our goal for Q3 is to ship the mobile app. I also really prefer dark mode."
OUT:
{"entities":[
  {"name":"ship mobile app by Q3","type":"goal"}
],"claims":[
  {"subject":"Alice","predicate":"has_goal","object":"ship mobile app by Q3","valence":0.8},
  {"subject":"ship mobile app by Q3","predicate":"targets","object":"mobile app","valence":0.7},
  {"subject":"Alice","predicate":"prefers","object":"dark mode","valence":0.7}
]}"""
