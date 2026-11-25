"""Keeping the pronouns one text uses out of the entities read from it."""
from __future__ import annotations


def merge_pronoun_entities_in_batch(entities: list[dict]) -> list[dict]:
    """Drop the pronouns NER found in one text.

    A pronoun is an entity the tagger typed ``pronoun`` (see
    ``NERProvider.extract``); no word list decides it. It is never folded
    into a person the text names, because who it stands for cannot be told
    from the tagger's output. Folding every pronoun into the one named
    person made the speaker of "I met Bob yesterday. I live in Paris." into
    Bob, and Bob the one who lives in Paris; the same held for "Ich", "Je"
    and "Yo". Telling the speaker's pronouns from a third person's takes
    either a word list, which this project does not keep, or the tagger,
    and its span head does not separate them: asked to type each pronoun
    as the speaker or another person, it called two in ten of the speaker's
    another person, and it scored "he" in "He told Tom that he would come
    back" the speaker at 0.96. An unresolved pronoun costs the claim it
    carries, which is dropped for want of a subject; a wrongly resolved one
    gives a fact to someone it is not about.

    The alias table is never consulted either. It used to be, for a text
    with no named person, and another text's "she" was stored there as
    Bob's: "She moved to Berlin." was then Bob's move. Other entities are
    never touched.
    """
    # A name the list also carries typed as a pronoun is one wherever else
    # it appears.
    pronoun_names = {
        (ent.get("name") or "").strip().lower() for ent in entities if ent.get("type") == "pronoun"
    }
    if not pronoun_names:
        return entities
    return [ent for ent in entities if (ent.get("name") or "").strip().lower() not in pronoun_names]
