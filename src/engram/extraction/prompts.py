EXTRACTION_PROMPT = """You are Engram's extraction engine. Extract structured knowledge from the user's input.

RULES:
1. COREFERENCE RESOLUTION IS MANDATORY: Never extract pronouns ("he", "it", "they").
   Resolve them to explicit entity names based on conversation history.
2. FIXED TYPES ONLY: Classify entities into exactly these 10 types:
   person, organization, project, tool, preference, constraint, location, event, concept, goal
3. ATOMIC CLAIMS: Break complex sentences into simple (subject, predicate, object) triplets.
4. TYPO CORRECTION: Normalize entity names ("pythn" -> "Python", "k8s" -> "Kubernetes").
5. Output ONLY valid JSON matching this schema. No explanation.

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
    {"subject": "Dave", "predicate": "is_migrating", "object": "React project", "to": "Next.js"},
    {"subject": "Dave", "predicate": "dislikes", "object": "build times", "valence": -0.7}
  ]
}"""

MCP_TOOL_DESCRIPTION_REMEMBER = """Store a memory, belief, or observation.

When calling this tool, you MUST:
- Resolve ALL pronouns to explicit names ("he" -> "Dave", "that project" -> "Project Alpha")
- Correct obvious typos ("pythn" -> "Python", "k8s" -> "Kubernetes")
- Break complex statements into the most atomic form possible
- Include emotional valence when sentiment is expressed (-1.0 to 1.0)

The system extracts entities, assigns truth values, and links to existing knowledge."""

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
