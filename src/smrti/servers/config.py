"""Shared environment-variable configuration for all Smrti server modes."""
from __future__ import annotations

import os

DB: str = os.environ.get("SMRTI_DB", "~/.smrti/memory.db")
PERSONALITY: str = os.environ.get("SMRTI_PERSONALITY", "balanced")
TENANT_ID: str = os.environ.get("SMRTI_TENANT_ID", "default")
SPACE: str = os.environ.get("SMRTI_SPACE", "default")

# Optional API key — when set, REST/proxy/viz require it on every request
# (Authorization: Bearer <key> or X-Api-Key: <key>).
API_KEY: str = os.environ.get("SMRTI_API_KEY", "")

# Extra SQLite paths the visualizer's ?db= param may open — os.pathsep-separated.
# Empty by default, so the only browsable DB is the one the server was started
# with; the endpoints are unauthenticated unless SMRTI_API_KEY is set, so an
# ungated ?db= would let a GET open (or create) any file as a SQLite database.
_viz_dbs_raw: str = os.environ.get("SMRTI_VIZ_DBS", "")
VIZ_DBS: list[str] = [p.strip() for p in _viz_dbs_raw.split(os.pathsep) if p.strip()]

# CORS origins for the proxy — comma-separated; CORS middleware is only added when set
_cors_raw: str = os.environ.get("SMRTI_CORS_ORIGINS", "")
CORS_ORIGINS: list[str] = [o.strip() for o in _cors_raw.split(",") if o.strip()]

# Max characters of a single memory text injected into the system prompt
INJECT_MAX_CHARS: int = int(os.environ.get("SMRTI_INJECT_MAX_CHARS", "500"))

_read_raw: str = os.environ.get("SMRTI_READ_SPACES", "")
READ_SPACES: list[str] | None = [s.strip() for s in _read_raw.split(",") if s.strip()] or None

# Ignore patterns — newline-separated regex patterns; matching content is silently dropped
_ignore_raw: str = os.environ.get("SMRTI_IGNORE_PATTERNS", "")
IGNORE_PATTERNS: list[str] = [p.strip() for p in _ignore_raw.splitlines() if p.strip()]

# Extraction — entity/claim extraction after remember() calls
EXTRACT: bool = os.environ.get("SMRTI_EXTRACT", "1") == "1"
EXTRACT_MODE: str = os.environ.get("SMRTI_EXTRACT_MODE", "hybrid")
EXTRACT_URL: str = (
    os.environ.get("SMRTI_EXTRACT_URL")
    or os.environ.get("SMRTI_UPSTREAM_URL", "https://api.openai.com")
)
EXTRACT_MODEL: str = os.environ.get("SMRTI_EXTRACT_MODEL", "")
# Thinking mode for extraction LLM calls.
# "auto"     — don't modify the request (default)
# "disabled" — pass chat_template_kwargs={"enable_thinking":false} to suppress
#              chain-of-thought (llama.cpp / vLLM Qwen3 style); faster and avoids
#              token-budget exhaustion on thinking models
# "enabled"  — pass chat_template_kwargs={"enable_thinking":true} to force thinking on
EXTRACT_THINKING: str = os.environ.get("SMRTI_EXTRACT_THINKING", "disabled")
