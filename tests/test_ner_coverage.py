"""Coverage tests for the NER provider plumbing (extraction/ner.py).

The model weights are never present in CI, so every test here drives
`NERProvider` with a stand-in — what is under test is the chunking, the
lazy load, the priority/pronoun merge, the verb-phrase filter and the
pronoun lexicon, none of which depend on the real weights.
"""
from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from smrti.extraction import ner as ner_mod
from smrti.extraction.ner import NERProvider, _chunk_text, _is_verb_phrase, get_ner


def _span(text: str, label: str, score: float = 1.0) -> SimpleNamespace:
    """One entity as the ONNX runtime returns it."""
    return SimpleNamespace(text=text, label=label, score=score, start=0, end=len(text))


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
        [_span("Ada", "person", 0.9)],
        [_span("Cyqle", "organization", 0.8)],
    ]
    provider._model = model

    with patch("smrti.extraction.ner._chunk_text", return_value=["chunk one", "chunk two"]):
        results = provider.extract("ignored")

    assert {r["name"] for r in results} == {"Ada", "Cyqle"}
    assert model.extract_entities.call_count == 2


# ── _get_model ────────────────────────────────────────────────────────────────

def _install_fake_runtime(monkeypatch, from_pretrained):
    module = type(sys)("gliner2_onnx")
    module.GLiNER2ONNXRuntime = MagicMock()
    module.GLiNER2ONNXRuntime.from_pretrained = from_pretrained
    monkeypatch.setitem(sys.modules, "gliner2_onnx", module)
    return module


def test_get_model_loads_once_and_caches(monkeypatch):
    runtime = MagicMock(spec=["extract_entities", "classify"])
    from_pretrained = MagicMock(return_value=runtime)
    _install_fake_runtime(monkeypatch, from_pretrained)

    provider = NERProvider(model_name="some/model-onnx")
    assert provider._get_model() is runtime
    from_pretrained.assert_called_once_with("some/model-onnx")
    # Second call reuses the cached instance.
    provider._get_model()
    assert from_pretrained.call_count == 1


def test_model_name_defaults_to_the_environment(monkeypatch):
    monkeypatch.setenv("SMRTI_NER_MODEL", "acme/custom-ner")
    assert NERProvider()._model_name == "acme/custom-ner"


def test_model_name_defaults_to_the_onnx_export(monkeypatch):
    monkeypatch.delenv("SMRTI_NER_MODEL", raising=False)
    assert NERProvider()._model_name == ner_mod._DEFAULT_MODEL


# ── extract: span handling ────────────────────────────────────────────────────

def test_extract_surfaces_the_real_confidence():
    provider = NERProvider()
    model = MagicMock()
    model.extract_entities.return_value = [_span("Ada", "person", 0.7)]
    provider._model = model

    assert provider.extract("Ada wrote the notes") == [
        {"name": "Ada", "type": "person", "score": 0.7}
    ]


def test_extract_drops_empty_span_text():
    provider = NERProvider()
    model = MagicMock()
    model.extract_entities.return_value = [_span("", "person", 0.9)]
    provider._model = model

    assert provider.extract("nothing nameable here") == []


def test_extract_keeps_the_most_specific_type():
    """A span tagged both `person` and `topic` keeps `person` — the lower index."""
    provider = NERProvider()
    model = MagicMock()
    model.extract_entities.return_value = [
        _span("Ada", "topic", 0.9), _span("Ada", "person", 0.8),
    ]
    provider._model = model

    assert provider.extract("Ada again") == [
        {"name": "Ada", "type": "person", "score": 0.8}
    ]


def test_extract_restores_the_pronoun_type_after_priority_merge():
    """A span tagged both `person` and `pronoun` keeps `pronoun` in the output."""
    provider = NERProvider()
    model = MagicMock()
    model.extract_entities.return_value = [
        _span("she", "person"), _span("she", "pronoun"),
    ]
    provider._model = model

    results = provider.extract("she shipped it")
    assert results == [{"name": "she", "type": "pronoun", "score": 1.0}]


def test_extract_keeps_the_real_type_when_pronoun_tag_is_noise():
    """A proper name tagged both `technology` and `pronoun` keeps `technology`.

    The model noise-tags product names as pronoun; only lexicon membership
    may turn a span into one.
    """
    provider = NERProvider()
    model = MagicMock()
    model.extract_entities.return_value = [
        _span("Factor", "technology", 0.6), _span("Factor", "pronoun", 0.5),
    ]
    provider._model = model

    assert provider.extract("Factor runs on the desktop") == [
        {"name": "Factor", "type": "technology", "score": 0.6}
    ]


def test_extract_filters_verb_phrases():
    provider = NERProvider()
    model = MagicMock()

    def entities(text, labels, threshold=0.4):
        if labels == ["noun_phrase", "verb_phrase"]:
            return [_span(text, "verb_phrase", 0.9), _span(text, "noun_phrase", 0.2)]
        return [_span("Never delete the volume", "constraint"), _span("Postgres", "technology")]

    model.extract_entities.side_effect = entities
    provider._model = model

    results = provider.extract("Never delete the volume; we run Postgres")
    assert [r["name"] for r in results] == ["Postgres"]


# ── _is_verb_phrase ───────────────────────────────────────────────────────────

def test_is_verb_phrase_only_applies_to_preferences_and_constraints():
    model = MagicMock()
    assert _is_verb_phrase({"type": "person", "name": "Ada Lovelace Byron"}, model) is False
    model.extract_entities.assert_not_called()


def test_is_verb_phrase_skips_short_multiword_spans():
    model = MagicMock()
    assert _is_verb_phrase({"type": "preference", "name": "dark mode"}, model) is False
    model.extract_entities.assert_not_called()


def test_is_verb_phrase_checks_a_single_unsegmented_run():
    """CJK/Thai spans arrive as one long word — they must still reach the model."""
    model = MagicMock()
    model.extract_entities.return_value = [_span("絶対に削除しない", "verb_phrase", 0.8)]
    assert _is_verb_phrase({"type": "constraint", "name": "絶対に削除しない"}, model) is True
    model.extract_entities.assert_called_once()


def test_is_verb_phrase_keeps_noun_phrases():
    model = MagicMock()
    model.extract_entities.return_value = [
        _span("strong black coffee", "noun_phrase", 0.6),
        _span("strong black coffee", "verb_phrase", 0.03),
    ]
    assert _is_verb_phrase({"type": "preference", "name": "strong black coffee"}, model) is False


def test_is_verb_phrase_needs_a_verb_span_to_win():
    """Neither label matching leaves the entity in place."""
    model = MagicMock()
    model.extract_entities.return_value = []
    assert _is_verb_phrase({"type": "constraint", "name": "avoid at all costs"}, model) is False


# ── classify_pronoun ──────────────────────────────────────────────────────────

def test_classify_pronoun_rejects_blank_input():
    provider = NERProvider()
    assert provider.classify_pronoun("") is False
    assert provider.classify_pronoun("   ") is False


def test_classify_pronoun_is_a_lexicon_not_a_model_call():
    """No weights are consulted — the model is never even loaded."""
    provider = NERProvider()
    with patch.object(NERProvider, "_get_model", side_effect=AssertionError("loaded the model")):
        assert provider.classify_pronoun("she") is True
        assert provider.classify_pronoun("Elara") is False


def test_classify_pronoun_ignores_case_and_surrounding_space():
    provider = NERProvider()
    assert provider.classify_pronoun("  THEY  ") is True


def test_classify_pronoun_covers_every_listed_language():
    provider = NERProvider()
    for pronoun in ("i", "ella", "eu", "nous", "ich", "loro", "彼女", "我们"):
        assert provider.classify_pronoun(pronoun) is True, pronoun


# ── singleton ─────────────────────────────────────────────────────────────────

def test_get_ner_returns_a_singleton():
    assert get_ner() is get_ner()
