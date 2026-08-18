"""Zero-shot NER via GLiNER2 on ONNX Runtime — lazy-loaded, thread-safe singleton.

ONNX Runtime is the only backend. The PyTorch build of GLiNER2 executes SSE4.1
instructions inside oneDNN the moment it is imported, which is an illegal
instruction — not a missing feature — on pre-2012 x86 hardware, and SIGILL is a
signal Python cannot catch: one import takes the whole engine down. ONNX Runtime
carries no such floor, so the same model runs everywhere, and the image sheds
torch's ~800MB with it.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

# The ONNX export of the multilingual GLiNER2 model, published alongside the
# runtime. Override with SMRTI_NER_MODEL to use another export (fp16 variants,
# the larger English model, or a local path).
_DEFAULT_MODEL = "lmo3/gliner2-multi-v1-onnx"

# GLiNER transformer context window is ~386–512 sub-word tokens.
# At UTF-8 encoding rates this maps to roughly 1 500 bytes for mixed-script
# text — conservative enough for CJK/Arabic while rarely splitting typical
# conversation turns (< 500 bytes).
_NER_CHUNK_BYTES: int = 1500


def _chunk_text(text: str) -> list[str]:
    """Return *text* split into NER-safe byte-sized chunks.

    Uses chonkie-core (SIMD-accelerated, splits at sentence boundaries) when
    available.  Falls back to the full text when the library is not installed
    or the text already fits within the context window.
    """
    if len(text.encode("utf-8")) <= _NER_CHUNK_BYTES:
        return [text]
    try:
        from chonkie_core import Chunker  # type: ignore[import-untyped]

        return [
            bytes(chunk).decode("utf-8", errors="replace")
            for chunk in Chunker(text, size=_NER_CHUNK_BYTES)
        ]
    except ImportError:
        return [text]


class NERProvider:
    """Wraps GLiNER2's ONNX runtime with lazy initialization. Thread-safe singleton."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.environ.get("SMRTI_NER_MODEL", _DEFAULT_MODEL)
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from gliner2_onnx import GLiNER2ONNXRuntime

                    self._model = GLiNER2ONNXRuntime.from_pretrained(self._model_name)
        return self._model

    def extract(
        self,
        text: str,
        labels: list[str] | None = None,
        threshold: float = 0.4,
    ) -> list[dict]:
        """Extract entity spans from text.

        Long texts are split into NER-safe chunks so that entities near the end
        of the input are not silently dropped by the transformer context window.
        Results from all chunks are merged before post-processing.

        `threshold` is the minimum span confidence; real confidence scores are
        surfaced in the returned dicts.

        Returns list of {"name": str, "type": str, "score": float}.
        """
        if labels is None:
            labels = _DEFAULT_LABELS
        model = self._get_model()

        # Collect entities across all chunks, merging as we go. Track every span
        # GLiNER tagged as "pronoun" before priority deduplication discards that
        # label. Deduplicate: keep one entry per name_lower, preferring the
        # highest-priority (most specific) type.
        pronoun_tagged: set[str] = set()
        best: dict[str, dict] = {}

        for chunk in _chunk_text(text):
            for span in model.extract_entities(chunk, labels, threshold=threshold):
                name = span.text
                if not name:
                    continue
                etype = span.label
                key = name.lower()
                if etype == "pronoun":
                    pronoun_tagged.add(key)
                priority = _TYPE_PRIORITY.get(etype, len(_DEFAULT_LABELS))
                existing = best.get(key)
                if existing is None or priority < _TYPE_PRIORITY.get(existing["type"], len(_DEFAULT_LABELS)):
                    best[key] = {"name": name, "type": etype, "score": float(span.score)}

        # Restore pronoun type for any span GLiNER labelled as pronoun,
        # regardless of what the priority system selected as the winning type.
        result = list(best.values())
        for ent in result:
            if ent["name"].lower() in pronoun_tagged:
                ent["type"] = "pronoun"
        # Filter out verb phrases misidentified as preference/constraint entities.
        # Uses the span head (multilingual) to distinguish noun phrases from
        # imperative clauses like "Avoid at all costs" / "Niemals löschen".
        return [ent for ent in result if not _is_verb_phrase(ent, model)]

    def classify_pronoun(self, name: str) -> bool:
        """Return True if `name` is a pronoun rather than a proper name.

        A lexicon rather than a model call. Pronouns are a closed class — every
        language has a few dozen and coins no more — so a lookup is exact where
        zero-shot inference is not: asked to choose between `proper_name` and
        `pronoun`, the model answers `proper_name` for "she", "he", "they" and
        "my" alike, on both the ONNX and the PyTorch build. Membership needs no
        weights, no context and no inference call.
        """
        return name.strip().casefold() in _PRONOUNS


def _is_verb_phrase(ent: dict, model) -> bool:
    """Return True if an entity span is a verb/imperative phrase rather than a noun phrase.

    Only checks preference/constraint entities — these are the types most
    prone to matching imperative clauses ("Avoid X", "Niemals Y"). The
    model runs when the span has 3+ whitespace-separated words OR is a
    single space-free run of 4+ characters, so unsegmented scripts
    (CJK/Thai) still reach it.

    Asks the span head rather than the classifier head: given the two labels as
    entity types it separates "Avoid at all costs" (verb 0.86 / noun 0.45) from
    "strong black coffee" (verb 0.03 / noun 0.58), while the classifier head
    answers `noun_phrase` for both. Language-agnostic — the same margin holds
    for German imperatives.
    """
    etype = ent.get("type", "")
    name = ent.get("name", "")
    if etype not in ("preference", "constraint"):
        return False
    words = len(name.split())
    if not (words >= 3 or (words == 1 and len(name) >= 4)):
        return False
    best: dict[str, float] = {}
    for span in model.extract_entities(name, ["noun_phrase", "verb_phrase"], threshold=0.0):
        best[span.label] = max(best.get(span.label, 0.0), float(span.score))
    return best.get("verb_phrase", 0.0) > best.get("noun_phrase", 0.0)


# Personal, possessive and reflexive pronouns across the languages the
# multilingual model covers. Closed class: this list is complete per language
# by definition, which is what makes a lookup the right instrument.
_PRONOUNS: frozenset[str] = frozenset(
    # English
    "i me my mine myself you your yours yourself yourselves he him his himself "
    "she her hers herself it its itself we us our ours ourselves they them "
    "their theirs themselves"
    # Spanish
    " yo me mi mis mío mía míos mías tú te tu tus tuyo tuya usted ustedes él "
    "ella ello lo la le su sus suyo suya nosotros nosotras nos nuestro nuestra "
    "vosotros vosotras os ellos ellas les"
    # Portuguese
    " eu mim meu minha meus minhas tu ti teu tua você vocês ele ela dele dela "
    "nós nosso nossa eles elas deles delas"
    # French
    " je moi mon ma mes tu toi ton ta tes vous votre vos il elle lui son sa ses "
    "nous notre nos ils elles leur leurs"
    # German
    " ich mich mir mein meine du dich dir dein deine er ihn ihm sein seine sie "
    "ihr ihre es wir uns unser unsere ihnen"
    # Italian
    " io me mio mia miei mie tu te tuo tua lui lei suo sua noi nostro nostra "
    "voi vostro vostra loro"
    # Japanese
    " 私 わたし 僕 ぼく 俺 おれ 彼 彼女 彼ら 私たち あなた"
    # Chinese
    " 我 你 您 他 她 它 我们 你们 他们 她们 它们".split()
)

_DEFAULT_LABELS = [
    "person",
    "organization",
    "project",
    "role",          # job titles and occupations ("software engineer", "CEO")
    "technology",    # languages, frameworks, platforms, tools ("Python", "Docker", "Kubernetes")
    "skill",         # abilities and competencies ("public speaking", "cooking", "piano")
    "preference",
    "constraint",
    "location",
    "event",
    "topic",         # subject domains and disciplines ("machine learning", "DevOps")
    "media",         # books, shows, podcasts, courses, articles ("Atomic Habits", "Lex Fridman")
    "health",        # conditions, symptoms, treatments, wellness ("insomnia", "therapy")
    "concept",
    "goal",
    "pronoun",
]


# When GLiNER tags the same span under multiple labels, keep the most specific.
# Lower index = higher priority.
_TYPE_PRIORITY: dict[str, int] = {t: i for i, t in enumerate(_DEFAULT_LABELS)}

_instance: Optional[NERProvider] = None
_instance_lock = threading.Lock()


def get_ner() -> NERProvider:
    """Return the module-level NERProvider singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NERProvider()
    return _instance
