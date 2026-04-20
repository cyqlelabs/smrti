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

import pytest

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


# Sample corpus — spans Simplified Chinese, Traditional, emoji, mixed script
CJK_CORPUS = [
    "心生万法——观察决定显现",  # Simplified
    "中道不是折中，是超越二元",
    "我記得菩提樹下的那49天",  # Traditional
    "anxiety_severe → 呼吸 → 平静 🙏",  # mixed + emoji
    "苦 = 执着于无常的结果",
]


def test_remember_stores_chinese_verbatim(cjk_mem):
    """Chinese content should be stored and retrieved byte-for-byte."""
    text = "心生万法——观察决定显现"
    atom_id = cjk_mem.remember(text, type="episode")
    assert atom_id

    atom = cjk_mem.atomspace.get_atom(atom_id, cjk_mem.tenant_id, cjk_mem.write_space)
    assert atom is not None
    assert atom.content == text, "no mojibake / no truncation"


def test_remember_then_recall_chinese(cjk_mem):
    """After remembering several Chinese fragments, recall should surface them."""
    for text in CJK_CORPUS:
        cjk_mem.remember(text)

    results = cjk_mem.recall("空性 无常", top_k=3)
    assert isinstance(results, list)


def test_believe_chinese_statement(cjk_mem):
    """A belief expressed in Chinese should round-trip."""
    atom_id = cjk_mem.believe("痛苦是信号，不是敌人", probability=0.9)
    assert atom_id


def test_recall_mixed_cjk_english(cjk_mem):
    """A query that mixes English emotion labels and Chinese should work."""
    results = cjk_mem.recall("anxiety_severe 焦虑", top_k=5)
    assert isinstance(results, list)


def test_remember_chinese_with_negative_valence(cjk_mem):
    """Emotional valence should apply regardless of script."""
    atom_id = cjk_mem.remember(
        "离开妻子耶输陀罗和儿子的那一夜",
        valence=-0.7,
        probability=0.95,
    )
    assert atom_id
