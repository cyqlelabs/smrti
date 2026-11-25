"""Entity resolution: exact -> alias -> accent folding -> fuzzy -> embedding -> create."""
from __future__ import annotations

import logging
import struct
import unicodedata
import uuid

from rapidfuzz import fuzz, process

from smrti.core.atomspace import AtomSpace
from smrti.core.models import (
    INITIAL_CONFIDENCE,
    STRUCTURAL_RELATIONS,
    Atom,
    AtomType,
    AttentionValue,
    EntityType,
    TruthValue,
)
from smrti.core.provenance import (
    ATOM_FORGOTTEN,
    ATOM_METADATA_JSON,
    ATOM_SOURCE,
    SOURCE_AGENT,
    SOURCE_USER,
    claim_current_sql,
    forgotten_sql,
)
from smrti.decisions import DecisionEngine
from smrti.decisions.extraction import verify_entity


logger = logging.getLogger("smrti.resolve")


def _is_latin(char: str) -> bool:
    return "A" <= char <= "ɏ" or "Ḁ" <= char <= "ỿ"


def _fold(name: str) -> str:
    """*name* case-folded, with the accents on Latin letters dropped.

    "José" and "Jose" are one name written with and without its accent.
    Only Latin letters lose theirs: in Devanagari or kana the combining mark
    makes another letter (क़ is not क, が is not か).
    """
    kept: list[str] = []
    for char in unicodedata.normalize("NFD", name.casefold()):
        if unicodedata.combining(char) and kept and _is_latin(kept[-1]):
            continue
        kept.append(char)
    return unicodedata.normalize("NFC", "".join(kept))


class EntityResolver:
    """Resolves extracted entity names to existing atoms or creates new ones.

    Reads are scoped to read_spaces (overlay); writes go to write_space only.

    Resolution tiers (ordered by cost):
      0. Exact label match      — indexed, < 1ms
      1. Alias table lookup     — indexed, < 1ms
      1b. Accent-folded label   — in-process, < 5ms
      2. Fuzzy match (RapidFuzz) — in-process, < 5ms
      3. Embedding cosine sim   — ONNX inference, < 20ms
      4. Create new atom        — write path

    Tiers 0 to 1b are unambiguous and never questioned. Tiers 2 and 3 only
    propose: a name that looks like a known one is as often a different
    entity (Paul and Paula, Java and JavaScript, React and React Native).
    When the ``entity`` decision task is on, the candidates they found are
    put to the provider with the sentence the mention came from and what
    the graph records about each, plus "none" and "ambiguous", and only a
    confident choice of one of them is an identity link. Anything else, and
    every uncertain match when the task is off, makes a provisional
    duplicate, which a later mention can still merge, where a wrong link
    cannot be undone.
    """

    def __init__(
        self,
        db,
        embed_engine,
        fuzzy_threshold: float = 85.0,
        # Cosine distance: 0.2 requires similarity >= 0.8 — loose enough for
        # semantic variants ("Postgres"/"PostgreSQL"), tight enough that
        # distinct names ("Alice"/"Alicia" ~ 0.75 sim) never silently merge.
        cosine_threshold: float = 0.2,
        source: str = SOURCE_USER,
        agent_trust: float = 0.5,
        episode_id: str = "",
        decisions: DecisionEngine | None = None,
        context: str = "",
    ) -> None:
        self.db = db
        self.embed_engine = embed_engine
        self.fuzzy_threshold = fuzzy_threshold
        self.cosine_threshold = cosine_threshold
        self.source = source
        self.episode_id = episode_id
        self.decisions = decisions if isinstance(decisions, DecisionEngine) else None
        self.context = context or ""
        # Atoms extracted from an agent turn start proportionally weaker and
        # corroborate proportionally less, so the graph reflects what the user
        # said unless the model's contribution is picked up later.
        self.trust = agent_trust if source == SOURCE_AGENT else 1.0

        from smrti.extraction.aliases import AliasManager
        self.aliases = AliasManager(db)
        # Write spaces whose stored pronoun aliases this resolver has purged.
        self._purged: set[tuple[str, str]] = set()

    def resolve(
        self,
        name: str,
        entity_type: str,
        tenant_id: str,
        write_space: str,
        read_spaces: list[str],
    ) -> str:
        """Return atom_id for the named entity, creating one if needed.

        Searches across read_spaces; new atoms are created in write_space.
        Only atoms in write_space are ever written to — a match found in a
        space this session merely reads is returned untouched. A forgotten
        atom is never a match on any tier: forgetting is final, and an
        entity mentioned again after being forgotten is a new one.
        """
        # Tier 3 probes once per space, so a repeated name is repeated ONNX work.
        read_spaces = list(dict.fromkeys(read_spaces))
        spaces_ph = ",".join("?" * len(read_spaces))

        # Tier 0: exact label match across read_spaces (u_lower: Unicode-aware
        # case folding — SQLite's LOWER() only folds ASCII)
        atom_id = self._exact_match(name, entity_type, tenant_id, read_spaces)
        if atom_id:
            self._boost_sti(atom_id, tenant_id, write_space)
            return atom_id

        # Tier 0b: the same label under a compatible entity type. GLiNER puts
        # several labels on one span, so "Python" arrives as technology in
        # one message and concept in the next. A person, an organization and
        # a place are each a distinct kind of referent, and a label shared
        # with one of them is a homonym: Java the island is not Java the
        # language, nor Apple the company the fruit. A goal only ever matches
        # a goal. A preference or constraint also matches a concept of the
        # same name, which it makes the belief it says it is, and a later
        # mention of the concept finds that belief rather than making the
        # concept again.
        atom_type = self._ENTITY_TYPE_TO_ATOM_TYPE.get(entity_type, "concept")
        if entity_type not in self._DISTINCT_KINDS:
            targets = self._CROSS_TYPE_TARGETS[atom_type]
            row = self.db.fetchone(
                f"""SELECT id, type, space FROM atoms
                    WHERE u_lower(label) = u_lower(?) AND type IN ({",".join("?" * len(targets))})
                      AND COALESCE(entity_type, '') NOT IN ({",".join("?" * len(self._DISTINCT_KINDS))})
                      AND tenant_id = ? AND space IN ({spaces_ph}) AND NOT {ATOM_FORGOTTEN}
                    ORDER BY type != ?, (sti + lti) DESC LIMIT 1""",
                (name, *targets, *self._DISTINCT_KINDS, tenant_id, *read_spaces, atom_type),
            )
            if row:
                if atom_type == "belief" and row["type"] == "concept" and row["space"] == write_space:
                    self._make_belief(row["id"], entity_type, tenant_id, write_space)
                self._boost_sti(row["id"], tenant_id, write_space)
                return row["id"]

        # Tier 1: alias table across read_spaces. Pronouns older builds stored
        # there are never a match, and are removed from the write space the
        # first time this resolver reads it; spaces it only reads are left as
        # they are.
        if (tenant_id, write_space) not in self._purged:
            self.aliases.purge_pronouns(tenant_id, write_space)
            self._purged.add((tenant_id, write_space))
        atom_id = self.aliases.lookup(name, tenant_id, read_spaces)
        if atom_id and not self._forgotten(atom_id):
            self._boost_sti(atom_id, tenant_id, write_space)
            return atom_id

        # The tiers below compare against the most salient atoms of the same
        # entity type, bounded so the scan cannot blow up.
        candidates = self.db.fetchall(
            f"SELECT id, label FROM atoms WHERE entity_type = ? AND tenant_id = ? AND space IN ({spaces_ph}) AND type != 'relation' AND NOT {ATOM_FORGOTTEN} ORDER BY (sti + lti) DESC LIMIT 500",
            (entity_type, tenant_id, *read_spaces),
        )

        # Tier 1b: the same spelling with or without its accents.
        folded = _fold(name)
        for candidate in candidates:
            if _fold(candidate["label"]) == folded:
                self._boost_sti(candidate["id"], tenant_id, write_space)
                return candidate["id"]

        # Without the entity task nothing can confirm what tiers 2 and 3
        # propose, so the mention becomes an atom of its own.
        if not self._verifying():
            return self._create_atom(name, entity_type, tenant_id, write_space)

        # Tier 2: fuzzy match within the same entity type.
        if candidates:
            names_map = {r["id"]: r["label"] for r in candidates}
            proposed = [
                m[2] for m in process.extract(
                    name, names_map, scorer=fuzz.WRatio,
                    limit=self._verify_candidates(),
                )
                if m[1] >= self.fuzzy_threshold
            ]
            if proposed:
                return self._confirmed_or_new(
                    name, entity_type, proposed, tenant_id, write_space, read_spaces,
                )

        # Tier 3: embedding cosine similarity via sqlite-vec. KNN filters
        # support equality only, so probe each read space and keep the best.
        query_vec = self.embed_engine.embed(name)
        vec_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)
        vec_rows: list = []
        for space in read_spaces:
            vec_rows.extend(
                self.db.fetchall(
                    """SELECT atom_id, distance FROM vec_atoms
                       WHERE embedding MATCH ? AND tenant_id = ? AND space = ?
                       ORDER BY distance LIMIT ?""",
                    (vec_bytes, tenant_id, space, self._verify_candidates()),
                )
            )
        vec_rows.sort(key=lambda r: r["distance"])
        vec_ids = []
        for row in vec_rows:
            if row["distance"] >= self.cosine_threshold:
                break
            atom_row = self.db.fetchone(
                f"SELECT entity_type FROM atoms WHERE id = ? AND NOT {ATOM_FORGOTTEN}",
                (row["atom_id"],),
            )
            if atom_row and atom_row["entity_type"] == entity_type:
                vec_ids.append(row["atom_id"])
        if vec_ids:
            return self._confirmed_or_new(
                name, entity_type, vec_ids, tenant_id, write_space, read_spaces, vec=query_vec,
            )

        # Tier 4: create new atom in write_space (reusing the probe vector)
        return self._create_atom(name, entity_type, tenant_id, write_space, vec=query_vec)

    _EXACT_MATCH_SQL = f"""SELECT id FROM atoms WHERE u_lower(label) = u_lower(?) AND entity_type = ?
        AND tenant_id = ? AND space IN ({{spaces}}) AND NOT {ATOM_FORGOTTEN}"""

    def _exact_match(self, name: str, entity_type: str, tenant_id: str, spaces: list[str]) -> str | None:
        row = self.db.fetchone(
            self._EXACT_MATCH_SQL.format(spaces=",".join("?" * len(spaces))),
            (name, entity_type, tenant_id, *spaces),
        )
        return row["id"] if row else None

    def _confirmed_or_new(
        self,
        name: str,
        entity_type: str,
        proposed: list[str],
        tenant_id: str,
        write_space: str,
        read_spaces: list[str],
        vec: list[float] | None = None,
    ) -> str:
        """The proposed atom the entity task confirms, else a new atom."""
        verdict = self._verify(name, entity_type, proposed, tenant_id, write_space, read_spaces)
        if verdict is not None and verdict.applied and verdict.matched:
            self._boost_sti(verdict.atom_id, tenant_id, write_space)
            return verdict.atom_id
        return self._create_atom(name, entity_type, tenant_id, write_space, vec=vec)

    def _make_belief(self, atom_id: str, entity_type: str, tenant_id: str, space: str) -> None:
        """Turn a concept into the preference or constraint a mention says it is."""
        self.db.execute(
            """UPDATE atoms SET type = 'belief', entity_type = ?
               WHERE id = ? AND tenant_id = ? AND space = ? AND type = 'concept'""",
            (entity_type, atom_id, tenant_id, space),
        )

    # A re-mention asserts the entity is real and still relevant, but it is a
    # weaker signal than an explicit belief assertion — hence short of 1.0.
    _MENTION_PROBABILITY = 0.9

    # How many facts each candidate carries into a verification.
    _VERIFY_FACTS = 4

    def _verifying(self) -> bool:
        return self.decisions is not None and self.decisions.enabled("entity")

    def _verify_candidates(self) -> int:
        """How many matches a tier proposes for verification."""
        return max(1, self.decisions.policy.entity_candidates)

    def _candidate_facts(self, atom_ids: list[str], tenant_id: str, spaces: list[str]) -> list[dict]:
        """Each candidate with what the graph currently records about it."""
        if not atom_ids:
            return []
        ph = ",".join("?" * len(atom_ids))
        spaces_ph = ",".join("?" * len(spaces))
        rows = self.db.fetchall(
            f"""SELECT id, label, entity_type FROM atoms
                WHERE id IN ({ph}) AND tenant_id = ? AND space IN ({spaces_ph}) AND NOT {ATOM_FORGOTTEN}""",
            (*atom_ids, tenant_id, *spaces),
        )
        by_id = {r["id"]: {"id": r["id"], "label": r["label"], "entity_type": r["entity_type"], "facts": []}
                 for r in rows}
        rel_ph = ",".join("?" * len(STRUCTURAL_RELATIONS))
        claims = self.db.fetchall(
            f"""SELECT r.source_id AS subject_id, r.relation AS predicate, t.label AS object
                FROM atoms r JOIN atoms t ON t.id = r.target_id
                WHERE r.type = 'relation' AND r.tenant_id = ? AND r.space IN ({spaces_ph})
                  AND r.source_id IN ({ph}) AND r.relation NOT IN ({rel_ph})
                  AND {claim_current_sql('r')} AND NOT {forgotten_sql('t')}
                ORDER BY r.confidence DESC, r.created_at DESC""",
            (tenant_id, *spaces, *atom_ids, *STRUCTURAL_RELATIONS),
        )
        for claim in claims:
            facts = by_id.get(claim["subject_id"], {}).get("facts")
            if facts is not None and len(facts) < self._VERIFY_FACTS:
                facts.append(f"{claim['predicate']} {claim['object']}")
        return [by_id[atom_id] for atom_id in atom_ids if atom_id in by_id]

    def _verify(
        self,
        name: str,
        entity_type: str,
        candidate_ids: list[str],
        tenant_id: str,
        write_space: str,
        read_spaces: list[str],
    ):
        """Put an uncertain match to the ``entity`` decision task.

        None when the task is off or could not answer, in which case the
        mention becomes an atom of its own. A verdict naming an
        atom is checked against the graph again after the call — the
        provider was given candidates, and only one of them, still present
        and not forgotten, can come back.
        """
        if not self._verifying():
            return None
        candidates = self._candidate_facts(candidate_ids, tenant_id, read_spaces)
        if not candidates:
            return None
        verdict = verify_entity(
            self.decisions,
            name=name,
            entity_type=entity_type,
            context=self.context,
            candidates=candidates,
            tenant_id=tenant_id,
            space=write_space,
        )
        if verdict is None:
            return None
        if verdict.atom_id is not None and (
            verdict.atom_id not in candidate_ids or self._forgotten(verdict.atom_id)
        ):
            logger.warning("entity verification named an atom that was not offered; ignoring it")
            return None
        return verdict

    def _boost_sti(self, atom_id: str, tenant_id: str, space: str) -> None:
        """Reinforce an atom on re-mention and log the mention as evidence.

        The evidence row is what separates a fact the user keeps returning to
        from one the model raised once and nobody picked up: user mentions
        carry full weight, agent mentions carry ``agent_trust``, so PLN builds
        confidence for the former roughly twice as fast as for the latter.

        Only atoms in the write space are reinforced. Resolution reads across
        the whole overlay, so a match may live in a space this session merely
        reads — boosting it there would let one agent's mentions drive another
        space's attention weights and, via the evidence row, its epoch would
        rewrite that space's truth values. Reads never mutate what they read.
        """
        if not self._in_write_space(atom_id, tenant_id, space):
            return
        self.db.execute(
            "UPDATE atoms SET sti = MIN(sti + ?, 3.0) WHERE id = ? AND tenant_id = ? AND space = ?",
            (0.5 * self.trust, atom_id, tenant_id, space),
        )
        self._adopt_if_user_mention(atom_id, tenant_id, space)
        self.db.execute(
            """INSERT INTO evidence
                   (id, atom_id, observed_probability, weight, source_episode_id,
                    text, source, tenant_id, space)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()), atom_id, self._MENTION_PROBABILITY,
                self.trust, self.episode_id or None, "mentioned again",
                self.source, tenant_id, space,
            ),
        )

    def _forgotten(self, atom_id: str) -> bool:
        return (
            self.db.fetchone(
                f"SELECT 1 FROM atoms WHERE id = ? AND {ATOM_FORGOTTEN}", (atom_id,)
            )
            is not None
        )

    def _in_write_space(self, atom_id: str, tenant_id: str, space: str) -> bool:
        return (
            self.db.fetchone(
                "SELECT 1 FROM atoms WHERE id = ? AND tenant_id = ? AND space = ?",
                (atom_id, tenant_id, space),
            )
            is not None
        )

    def _adopt_if_user_mention(self, atom_id: str, tenant_id: str, space: str) -> None:
        """Transfer an agent-authored atom to the user when the user mentions it.

        This is the one way model output earns permanence. Something the
        assistant volunteered starts weak and decays to nothing, but once the
        user brings it up themselves they have incorporated it, and from that
        point it is theirs — durable on the same terms as anything else they
        said. Without this, an entity the assistant introduced and the user
        adopted would keep evaporating no matter how often it came up.
        """
        if self.source != SOURCE_USER:
            return
        self.db.execute(
            f"""UPDATE atoms
                   SET metadata = json_set({ATOM_METADATA_JSON}, '$.source', ?)
                 WHERE id = ? AND tenant_id = ? AND space = ?
                   AND {ATOM_SOURCE} = ?""",
            (SOURCE_USER, atom_id, tenant_id, space, SOURCE_AGENT),
        )

    # Entity types naming a distinct kind of referent (see tier 0b).
    _DISTINCT_KINDS = ("person", "organization", "location")

    # The atom types a mention of each atom type may match under another
    # entity type (see tier 0b).
    _CROSS_TYPE_TARGETS = {"goal": ("goal",), "belief": ("belief", "concept"), "concept": ("concept", "belief")}

    _ENTITY_TYPE_TO_ATOM_TYPE = {
        "goal": "goal",
        "preference": "belief",
        "constraint": "belief",
    }

    def _create_atom(
        self,
        name: str,
        entity_type: str,
        tenant_id: str,
        space: str,
        vec: list[float] | None = None,
    ) -> str:
        atom_type = self._ENTITY_TYPE_TO_ATOM_TYPE.get(entity_type, "concept")
        try:
            typed = EntityType(entity_type)
        except ValueError:
            typed = EntityType.CONCEPT
        # Provenance is recorded on the derived atom, not just the episode it
        # came from: the epoch decays and prunes atoms, and without a source of
        # its own an agent-extracted concept is indistinguishable from a fact
        # the user stated. Truth and attention start scaled by trust.
        #
        # Written through AtomSpace like every other atom, so the row carries
        # everything a row must — the intrinsic valence pair above all. A
        # direct INSERT used to leave those columns NULL, and an atom with no
        # intrinsic tone is judged on the mood it absorbs from its neighbours,
        # which for the most-mentioned concepts meant the mood of every
        # complaint that ever mentioned them.
        atom = Atom(
            id=str(uuid.uuid4()),
            type=AtomType(atom_type),
            label=name,
            entity_type=typed,
            truth=TruthValue(
                probability=0.8,
                confidence=INITIAL_CONFIDENCE[atom_type] * self.trust,
            ),
            attention=AttentionValue(sti=1.0 * self.trust, lti=0.3 * self.trust),
            tenant_id=tenant_id,
            space=space,
            metadata={"source": SOURCE_AGENT} if self.source == SOURCE_AGENT else {},
        )
        # Extraction runs in the background of a write that already succeeded,
        # so a failed encoding costs the atom its vector, not the pipeline
        # its run: the atom stays reachable by label and by the lexical
        # index, and the next change to its text re-embeds it.
        if vec is None:
            try:
                vec = self.embed_engine.embed(name)
            except Exception:
                logger.warning("could not embed %r; created without a vector", name, exc_info=True)
                vec = None
        # Another writer, in this process or another, may have created the
        # same entity since the tiers above looked, most likely while the
        # vector was computed. The second look and the insert are one write
        # transaction, so of two writers creating one name the second finds
        # the first's atom. A unique index would do it with less, but a
        # graph may already hold duplicates, and a forgotten atom shares its
        # label with the one that replaced it.
        with self.db.transaction() as conn:
            existing = conn.execute(
                self._EXACT_MATCH_SQL.format(spaces="?"), (name, typed.value, tenant_id, space)
            ).fetchone()
            if existing is None:
                return AtomSpace(self.db, self.embed_engine).add_atom_in(
                    conn, atom, embedding=None if vec is None else list(vec),
                    require_vector=vec is not None,
                )
        self._boost_sti(existing["id"], tenant_id, space)
        return existing["id"]
