"""Miscellaneous coverage: truth.pln_merge, connections, __init__ edge cases."""
from __future__ import annotations

import os
import re
import tempfile
from unittest.mock import patch

import pytest

from smrti import Smrti
from smrti.core.models import TruthValue
from smrti.evolution.truth import pln_merge


# ── truth.pln_merge ───────────────────────────────────────────────────────────

def test_pln_merge_returns_truth_value():
    a = TruthValue(probability=0.8, confidence=0.6)
    b = TruthValue(probability=0.6, confidence=0.4)
    result = pln_merge(a, b)
    assert isinstance(result, TruthValue)
    assert 0.0 <= result.probability <= 1.0
    assert 0.0 <= result.confidence <= 1.0


# ── connections.discover_connections ─────────────────────────────────────────

def test_discover_connections_creates_associations():
    from smrti.evolution.connections import discover_connections

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        mem = Smrti(db_path=db_path, tenant_id="test", write_space="default")
        # Add atoms with LTI > 0.3 to trigger discovery
        a = mem.remember("machine learning algorithms")
        b = mem.remember("deep learning neural networks")
        # Boost LTI for both
        mem.db.execute("UPDATE atoms SET lti=0.8 WHERE id=?", (a,))
        mem.db.execute("UPDATE atoms SET lti=0.8 WHERE id=?", (b,))

        count = discover_connections("test", "default", mem.db, mem.embed)
        assert isinstance(count, int)
        mem.close()
    finally:
        os.unlink(db_path)


# ── Smrti.__init__ edge cases ─────────────────────────────────────────────────

def test_ignore_patterns_invalid_regex_raises():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        with pytest.raises(ValueError, match="invalid regex"):
            Smrti(db_path=db_path, ignore_patterns=["[invalid"])
    finally:
        os.unlink(db_path)


def test_is_ignored_returns_true_when_pattern_matches():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mem = Smrti(db_path=db_path, ignore_patterns=[r"SECRET"])
        assert mem.is_ignored("This contains SECRET data")
        assert not mem.is_ignored("This is fine")
        mem.close()
    finally:
        os.unlink(db_path)


def test_remember_skips_ignored_content():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mem = Smrti(db_path=db_path, ignore_patterns=[r"SKIP_ME"])
        result = mem.remember("SKIP_ME please")
        assert result == ""
        mem.close()
    finally:
        os.unlink(db_path)


def test_personality_env_var_forces_update():
    """When SMRTI_PERSONALITY is set, a mismatched preset triggers set_personality."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # First init with balanced
        mem1 = Smrti(db_path=db_path, personality="balanced")
        mem1.close()

        # Re-init with analytical + env var set → should update
        with patch.dict(os.environ, {"SMRTI_PERSONALITY": "analytical"}):
            mem2 = Smrti(db_path=db_path, personality="analytical")
            row = mem2.db.fetchone(
                "SELECT preset_name FROM personality WHERE tenant_id=? AND space=?",
                ("default", "default"),
            )
            assert row["preset_name"] == "analytical"
            mem2.close()
    finally:
        os.unlink(db_path)


def test_db_execute_many():
    """Database.execute_many must commit a batch."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        from smrti.core.db import Database
        db = Database(db_path)
        db.initialize()
        import uuid
        rows = [
            (str(uuid.uuid4()), "episode", f"batch {i}", "test", "default")
            for i in range(3)
        ]
        db.execute_many(
            "INSERT INTO atoms (id, type, label, tenant_id, space) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        count = db.fetchone("SELECT COUNT(*) as n FROM atoms WHERE tenant_id='test'")
        assert count["n"] >= 3
        db.close()
    finally:
        os.unlink(db_path)


def test_db_context_manager():
    """Database can be used as a context manager."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        from smrti.core.db import Database
        with Database(db_path) as db:
            db.initialize()
            row = db.fetchone("SELECT 1 as n")
            assert row["n"] == 1
    finally:
        os.unlink(db_path)
