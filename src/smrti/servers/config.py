"""Shared environment-variable configuration for all Smrti server modes."""
from __future__ import annotations

import os

DB: str = os.environ.get("SMRTI_DB", "~/.smrti/memory.db")
PERSONALITY: str = os.environ.get("SMRTI_PERSONALITY", "balanced")
TENANT_ID: str = os.environ.get("SMRTI_TENANT_ID", "default")
SPACE: str = os.environ.get("SMRTI_SPACE", "default")

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
