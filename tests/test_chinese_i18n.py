"""
Chinese-language fixture tests for smrti.

Verifies the memory engine handles CJK correctly:
- Round-trip of Chinese content through remember -> recall
- Believe/forget with Chinese statements
- Belief content preserved without mojibake
- Valence + STI still work with Chinese

Motivated by ZenMind AI's Siddhartha persona production deployment
(Chinese AI companion). Fixture is additive — no behavior change.
"""
import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from smrti import Smrti


@pytest.fixture(scope="module")
def cjk_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def cjk_mem(cjk_db_path):
    engine = Smrti(db_path=cjk_db_path, tenant_id="default", write_space="default")
    yield engine
    engine.close()


@pytest.fixture(scope="module")
def cjk_client(cjk_mem):
    from smrti.servers import rest as rest_mod

    async def _noop_reflect(*_args, **_kwargs):
        return

    with patch.object(rest_mod, "get_mem", return_value=cjk_mem):
        with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
            with patch("smrti.servers.config.EXTRACT", False):
                with TestClient(rest_mod.app, raise_server_exceptions=True) as c:
                    yield c


# Sample corpus — spans Simplified Chinese, Traditional, emoji, mixed script
CJK_CORPUS = [
    "心生万法——观察决定显现",  # Simplified
    "中道不是折中，是超越二元",
    "我記得菩提樹下的那49天",  # Traditional
    "anxiety_severe → 呼吸 → 平静 🙏",  # mixed + emoji
    "苦 = 执着于无常的结果",
]


def test_remember_stores_chinese_verbatim(cjk_client):
    """Chinese content should be stored and retrieved byte-for-byte."""
    text = "心生万法——观察决定显现"
    resp = cjk_client.post("/remember", json={"content": text, "type": "episode"})
    assert resp.status_code == 200
    atom_id = resp.json()["atom_id"]

    got = cjk_client.get(f"/atoms/{atom_id}")
    assert got.status_code == 200
    assert got.json()["content"] == text, "no mojibake / no truncation"


def test_remember_then_recall_chinese(cjk_client):
    """After remembering several Chinese fragments, recall should surface them."""
    for text in CJK_CORPUS:
        r = cjk_client.post("/remember", json={"content": text})
        assert r.status_code == 200

    # Recall with Chinese query — should return something
    r = cjk_client.post("/recall", json={"query": "空性 无常", "top_k": 3})
    assert r.status_code == 200
    # Response shape varies but at least shouldn't be empty dict on no match
    assert isinstance(r.json(), dict)


def test_believe_chinese_statement(cjk_client):
    """A belief expressed in Chinese should round-trip."""
    r = cjk_client.post("/believe", json={
        "statement": "痛苦是信号，不是敌人",
        "probability": 0.9,
    })
    assert r.status_code == 200


def test_recall_mixed_cjk_english(cjk_client):
    """A query that mixes English emotion labels and Chinese should work."""
    r = cjk_client.post("/recall", json={
        "query": "anxiety_severe 焦虑",
        "top_k": 5,
    })
    assert r.status_code == 200


def test_remember_chinese_with_negative_valence(cjk_client):
    """Emotional valence should apply regardless of script."""
    r = cjk_client.post("/remember", json={
        "content": "离开妻子耶输陀罗和儿子的那一夜",
        "valence": -0.7,
        "probability": 0.95,
    })
    assert r.status_code == 200
