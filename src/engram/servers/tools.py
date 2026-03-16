"""Shared MCP/REST tool definitions."""

TOOLS = [
    {
        "name": "engram_remember",
        "description": """Store a memory, belief, or observation.

When calling this tool, you MUST:
- Resolve ALL pronouns to explicit names ("he" → "Dave", "that project" → "Project Alpha")
- Correct obvious typos ("pythn" → "Python", "k8s" → "Kubernetes")
- Break complex statements into the most atomic form possible
- Include emotional valence when sentiment is expressed (-1.0 to 1.0)

The system extracts entities, assigns truth values, and links to existing knowledge.""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory or observation to store"},
                "type": {"type": "string", "enum": ["belief", "episode", "goal"], "default": "episode"},
                "probability": {"type": "number", "description": "How true is this (0-1)", "default": 0.8},
                "valence": {"type": "number", "description": "Emotional tone (-1 to 1)", "default": 0.0},
            },
            "required": ["content"],
        },
    },
    {
        "name": "engram_recall",
        "description": "Retrieve relevant memories using salience-scored search. Returns memories with their truth values, confidence, and emotional context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to recall"},
                "top_k": {"type": "integer", "default": 10},
                "min_confidence": {"type": "number", "default": 0.1},
            },
            "required": ["query"],
        },
    },
    {
        "name": "engram_reflect",
        "description": "Trigger a consolidation pass. Updates beliefs based on evidence, decays attention, discovers new connections. Returns a summary of changes.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "engram_believe",
        "description": "Assert or update a specific belief with a truth value. If the belief contradicts existing knowledge, creates a contradiction link.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "probability": {"type": "number"},
                "evidence": {"type": "string", "description": "Why you believe this"},
            },
            "required": ["statement", "probability"],
        },
    },
    {
        "name": "engram_forget",
        "description": "Lower confidence on a memory or belief. Does not hard-delete — the consolidation epoch handles pruning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to forget"},
                "reason": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "engram_personality",
        "description": "Get or set the agent's personality profile. Affects how memories are scored, how fast beliefs change, and emotional reactivity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "set", "preset"]},
                "preset": {"type": "string", "enum": ["balanced", "analytical", "curious", "empathetic", "maverick"]},
                "params": {"type": "object", "description": "Custom personality parameters"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "engram_status",
        "description": "Get memory statistics: total atoms, active beliefs, emotional state, attention distribution.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]
