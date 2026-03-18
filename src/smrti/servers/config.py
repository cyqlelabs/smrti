"""Shared environment-variable configuration for all Smrti server modes."""
from __future__ import annotations

import os

DB: str = os.environ.get("SMRTI_DB", "~/.smrti/memory.db")
PERSONALITY: str = os.environ.get("SMRTI_PERSONALITY", "balanced")
TENANT_ID: str = os.environ.get("SMRTI_TENANT_ID", "default")
SPACE: str = os.environ.get("SMRTI_SPACE", "default")

_read_raw: str = os.environ.get("SMRTI_READ_SPACES", "")
READ_SPACES: list[str] | None = [s.strip() for s in _read_raw.split(",") if s.strip()] or None
