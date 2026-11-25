"""Folding the pronouns one text uses into the person it names."""
from __future__ import annotations


def merge_pronoun_entities_in_batch(entities: list[dict]) -> list[dict]:
    """Fold the pronouns NER found in one text into the person it names.

    A pronoun is an entity the tagger typed ``pronoun`` (see
    ``NERProvider.extract``); no word list decides it.

    - one named person: the pronouns become other names for that person in
      this text, so the claims read from it resolve to them
    - none, or several: the pronouns are dropped, since who they stand for
      cannot be told from the text, and an atom called "she" is nobody

    The alias table is never consulted. It used to be, for a text with no
    named person, and another text's "she" was stored there as Bob's: "She
    moved to Berlin." was then Bob's move. Other entities are never touched.
    """
    # A name the list also carries typed as a pronoun is one wherever else
    # it appears.
    pronoun_names = {
        (ent.get("name") or "").strip().lower() for ent in entities if ent.get("type") == "pronoun"
    }
    pronouns = [ent for ent in entities if (ent.get("name") or "").strip().lower() in pronoun_names]
    if not pronouns:
        return entities
    kept = [ent for ent in entities if (ent.get("name") or "").strip().lower() not in pronoun_names]
    named = [ent for ent in kept if ent.get("type") == "person"]
    if len(named) == 1:
        _fold_into(named[0], pronouns)
    return kept


def _fold_into(target: dict, pronouns: list[dict]) -> None:
    aliases = list(target.get("aliases", []))
    seen = {alias.lower() for alias in aliases} | {(target.get("name") or "").lower()}
    for pronoun in pronouns:
        for name in [pronoun.get("name") or "", *pronoun.get("aliases", [])]:
            if name and name.lower() not in seen:
                aliases.append(name)
                seen.add(name.lower())
    target["aliases"] = aliases
