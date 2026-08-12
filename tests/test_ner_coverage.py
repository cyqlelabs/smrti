"""Coverage tests for the NER provider plumbing (extraction/ner.py).

GLiNER2 is never installed in CI, so every test here drives `NERProvider`
with a stand-in model — what is under test is the chunking, the lazy-load
fallback, the priority/pronoun merge and the verb-phrase filter, none of
which depend on the real weights.
"""
from __future__ import annotations

import builtins
import sys
from unittest.mock import MagicMock, patch

from smrti.extraction import ner as ner_mod
from smrti.extraction.ner import NERProvider, _chunk_text, _is_verb_phrase, get_ner


# ── _chunk_text ───────────────────────────────────────────────────────────────

def test_short_text_is_not_chunked():
    assert _chunk_text("a short sentence") == ["a short sentence"]


def test_long_text_is_split_into_chunks():
    text = "Sentence number one. " * 200  # comfortably over the byte budget
    chunks = _chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c.encode()) <= ner_mod._NER_CHUNK_BYTES for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_long_text_falls_back_to_one_chunk_without_chonkie():
    text = "x" * (ner_mod._NER_CHUNK_BYTES + 10)
    real_import = builtins.__import__

    def _no_chonkie(name, *args, **kwargs):
        if name == "chonkie_core":
            raise ImportError("no chonkie here")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", _no_chonkie):
        assert _chunk_text(text) == [text]


def test_extract_merges_entities_across_chunks():
    provider = NERProvider()
    model = MagicMock()
    model.extract_entities.side_effect = [
        {"entities": {"person": [{"text": "Ada", "confidence": 0.9}]}},
        {"entities": {"organization": [{"text": "Cyqle", "confidence": 0.8}]}},
    ]
    provider._model = model
    provider._has_classify = False

    with patch("smrti.extraction.ner._chunk_text", return_value=["chunk one", "chunk two"]):
        results = provider.extract("ignored")

    assert {r["name"] for r in results} == {"Ada", "Cyqle"}
    assert model.extract_entities.call_count == 2


# ── _get_model ────────────────────────────────────────────────────────────────

def _install_fake_gliner(monkeypatch, from_pretrained):
    module = type(sys)("gliner2")
    module.GLiNER2 = MagicMock()
    module.GLiNER2.from_pretrained = from_pretrained
    monkeypatch.setitem(sys.modules, "gliner2", module)
    return module


def test_get_model_prefers_the_local_snapshot(monkeypatch):
    local_model = MagicMock(spec=["extract_entities", "classify_text"])
    from_pretrained = MagicMock(return_value=local_model)
    _install_fake_gliner(monkeypatch, from_pretrained)

    provider = NERProvider(model_name="some/model")
    assert provider._get_model() is local_model
    from_pretrained.assert_called_once_with("some/model", local_files_only=True)
    assert provider._has_classify is True
    # Second call reuses the cached instance.
    provider._get_model()
    assert from_pretrained.call_count == 1


def test_get_model_downloads_when_no_local_snapshot_exists(monkeypatch):
    downloaded = MagicMock(spec=["extract_entities"])

    def from_pretrained(name, local_files_only=False):
        if local_files_only:
            raise OSError("not cached")
        return downloaded

    _install_fake_gliner(monkeypatch, MagicMock(side_effect=from_pretrained))
    provider = NERProvider(model_name="some/model")
    assert provider._get_model() is downloaded
    assert provider._has_classify is False


def test_model_name_defaults_to_the_environment(monkeypatch):
    monkeypatch.setenv("SMRTI_NER_MODEL", "acme/custom-ner")
    assert NERProvider()._model_name == "acme/custom-ner"


# ── extract: span normalisation ───────────────────────────────────────────────

def test_extract_accepts_a_bare_span_instead_of_a_list():
    provider = NERProvider()
    model = MagicMock()
    model.extract_entities.return_value = {"entities": {"person": {"text": "Ada", "confidence": 0.7}}}
    provider._model = model
    provider._has_classify = False

    results = provider.extract("Ada wrote the notes")
    assert results == [{"name": "Ada", "type": "person", "score": 0.7}]


def test_extract_drops_empty_span_text():
    provider = NERProvider()
    model = MagicMock()
    model.extract_entities.return_value = {"entities": {"person": [{"text": "", "confidence": 0.9}, ""]}}
    provider._model = model
    provider._has_classify = False

    assert provider.extract("nothing nameable here") == []


def test_extract_ignores_a_non_dict_response():
    provider = NERProvider()
    model = MagicMock()
    model.extract_entities.return_value = "not a dict"
    provider._model = model
    provider._has_classify = False

    assert provider.extract("whatever") == []


def test_extract_restores_the_pronoun_type_after_priority_merge():
    """A span tagged both `person` and `pronoun` keeps `pronoun` in the output."""
    provider = NERProvider()
    model = MagicMock()
    model.extract_entities.return_value = {
        "entities": {"person": ["she"], "pronoun": ["she"]}
    }
    provider._model = model
    provider._has_classify = False

    results = provider.extract("she shipped it")
    assert results == [{"name": "she", "type": "pronoun", "score": 1.0}]


def test_extract_filters_verb_phrases_when_the_classifier_is_available():
    provider = NERProvider()
    model = MagicMock()
    model.extract_entities.return_value = {
        "entities": {"constraint": ["Never delete the volume"], "technology": ["Postgres"]}
    }
    model.classify_text.return_value = {"type": "verb_phrase"}
    provider._model = model
    provider._has_classify = True

    results = provider.extract("Never delete the volume; we run Postgres")
    assert [r["name"] for r in results] == ["Postgres"]


# ── _is_verb_phrase ───────────────────────────────────────────────────────────

def test_is_verb_phrase_only_applies_to_preferences_and_constraints():
    model = MagicMock()
    assert _is_verb_phrase({"type": "person", "name": "Ada Lovelace Byron"}, model) is False
    model.classify_text.assert_not_called()


def test_is_verb_phrase_skips_short_multiword_spans():
    model = MagicMock()
    assert _is_verb_phrase({"type": "preference", "name": "dark mode"}, model) is False
    model.classify_text.assert_not_called()


def test_is_verb_phrase_checks_a_single_unsegmented_run():
    """CJK/Thai spans arrive as one long word — they must still reach the classifier."""
    model = MagicMock()
    model.classify_text.return_value = {"type": "verb_phrase"}
    assert _is_verb_phrase({"type": "constraint", "name": "絶対に削除しない"}, model) is True


def test_is_verb_phrase_keeps_noun_phrases():
    model = MagicMock()
    model.classify_text.return_value = {"type": "noun_phrase"}
    assert _is_verb_phrase({"type": "preference", "name": "strong black coffee"}, model) is False


def test_is_verb_phrase_survives_a_classifier_error():
    model = MagicMock()
    model.classify_text.side_effect = RuntimeError("model exploded")
    assert _is_verb_phrase({"type": "constraint", "name": "avoid at all costs"}, model) is False


def test_is_verb_phrase_ignores_a_non_dict_verdict():
    model = MagicMock()
    model.classify_text.return_value = "verb_phrase"
    assert _is_verb_phrase({"type": "constraint", "name": "avoid at all costs"}, model) is False


# ── classify_pronoun ──────────────────────────────────────────────────────────

def test_classify_pronoun_rejects_blank_input():
    provider = NERProvider()
    assert provider.classify_pronoun("") is False
    assert provider.classify_pronoun("   ") is False


def test_classify_pronoun_returns_false_when_the_model_cannot_load():
    provider = NERProvider()
    with patch.object(NERProvider, "_get_model", side_effect=ImportError("no gliner2")):
        assert provider.classify_pronoun("she") is False


def test_classify_pronoun_returns_false_without_classify_text():
    provider = NERProvider()
    provider._model = MagicMock(spec=["extract_entities"])
    provider._has_classify = False
    assert provider.classify_pronoun("she") is False


def test_classify_pronoun_ignores_a_non_dict_verdict():
    provider = NERProvider()
    model = MagicMock()
    model.classify_text.return_value = ["pronoun"]
    provider._model = model
    provider._has_classify = True
    assert provider.classify_pronoun("she") is False


def test_classify_pronoun_survives_a_classifier_error():
    provider = NERProvider()
    model = MagicMock()
    model.classify_text.side_effect = RuntimeError("boom")
    provider._model = model
    provider._has_classify = True
    assert provider.classify_pronoun("she") is False


# ── singleton ─────────────────────────────────────────────────────────────────

def test_get_ner_returns_a_singleton():
    assert get_ner() is get_ner()
