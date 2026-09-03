"""Scenario tests for retrieval: the engine must surface the facts, not the froth.

Each test replays a failure shape observed in a live graph, where "name all
the members of my family" returned the user's own past questions, the
agent's stored wrong answers, and sticky junk concepts — while the family
facts, present and healthy, never ranked.
"""
from __future__ import annotations

import json
import uuid

import pytest

from smrti import Smrti
from smrti.retrieval.fan_out import retrieve


@pytest.fixture
def mem(tmp_path):
    db_path = str(tmp_path / "scenarios.db")
    return Smrti(db_path=db_path, personality="balanced", tenant_id="test", write_space="default")


def _set(mem, atom_id: str, **cols) -> None:
    sets = ", ".join(f"{k} = ?" for k in cols)
    mem.db.execute(
        f"UPDATE atoms SET {sets} WHERE id = ?", (*cols.values(), atom_id)
    )


def _mark_agent(mem, atom_id: str) -> None:
    mem.db.execute(
        "UPDATE atoms SET metadata = ? WHERE id = ?",
        (json.dumps({"source": "agent"}), atom_id),
    )


def _recall(mem, query: str, top_k: int = 10):
    return retrieve(
        query, mem.tenant_id, [mem.write_space], mem.db, mem.embed,
        write_space=mem.write_space, top_k=top_k, min_confidence=0.0,
    )


def _rank(results, atom_id: str) -> int | None:
    for i, r in enumerate(results):
        if r.atom.id == atom_id:
            return i
    return None


# ── question echoes and the agent's own wrong answers ────────────────────────

def test_family_fact_surfaces_despite_echoes_and_denials(mem):
    """The failure being replayed: the fact never ranked at all.

    A stored utterance of the question is always nearer the query than any
    answer to it, and the agent's own failed reply ("I only have Lourdes")
    re-enters the graph as an episode. The fact — aged to the confidence
    floor, no attention left — must still make the default page, and the
    verbatim echoes must not be riding their similarity to get there.
    """
    fact = mem.believe(
        "Nicolás lives in San Benito with his wife Roxana and his daughter Esmeralda",
        probability=0.95,
    )
    # An aged fact: no short-term attention left, confidence at the floor.
    _set(mem, fact, sti=0.0, lti=0.36, confidence=0.1)

    query = "name all the members of my family"
    echo1 = mem.remember("name all the members of my family")
    echo2 = mem.remember("name all the members of my family please")
    denial = mem.remember("So far the only family member I have on record is Lourdes")
    _mark_agent(mem, denial)
    # Fresh chatter: just spoken, so attention is high.
    for atom_id in (echo1, echo2, denial):
        _set(mem, atom_id, sti=0.8)

    results = _recall(mem, query)
    assert _rank(results, fact) is not None, "the family fact never surfaced"
    for r in results:
        if r.atom.id in (echo1, echo2):
            assert r.similarity == 0.0, (
                "a verbatim echo of the query kept its similarity score"
            )


def test_question_echo_ranks_below_an_answer_of_equal_standing(mem):
    """With attention and confidence equal, the answer beats the asking."""
    query = "who are the members of my family"
    echo = mem.remember("who are the members of my family")
    answer = mem.remember("my family is Roxana, Esmeralda and Lourdes")
    _set(mem, echo, sti=0.5)
    _set(mem, answer, sti=0.5)

    results = _recall(mem, query)
    echo_rank, answer_rank = _rank(results, echo), _rank(results, answer)
    assert answer_rank is not None
    assert echo_rank is None or answer_rank < echo_rank


def test_agent_authored_twin_ranks_below_user_testimony(mem):
    """Source trust reaches ranking: the same words weigh less from the agent.

    The engine already decays and prunes agent content faster; recall was
    the one place the asymmetry was missing, so the agent's stored replies
    competed head-to-head with the testimony they were derived from.
    """
    text = "the password rotation for the vault happens monthly"
    user_atom = mem.remember(text)
    agent_atom = mem.remember(text + " as far as I recorded")
    _mark_agent(mem, agent_atom)
    for atom_id in (user_atom, agent_atom):
        _set(mem, atom_id, sti=0.3, confidence=0.5)

    results = _recall(mem, "how often does the vault password rotate")
    user_rank, agent_rank = _rank(results, user_atom), _rank(results, agent_atom)
    assert user_rank is not None
    assert agent_rank is None or user_rank < agent_rank


def test_verbatim_search_for_a_belief_still_returns_it_first(mem):
    """Echo suppression is for episodes; stating a fact must find the fact."""
    fact = mem.believe("the staging database is rebuilt every Sunday night", probability=0.9)
    mem.remember("what happens to the staging database on weekends")

    results = _recall(mem, "the staging database is rebuilt every Sunday night")
    assert results and results[0].atom.id == fact


# ── graph expansion carries real similarity ───────────────────────────────────

def test_fact_reached_through_the_graph_is_scored_on_its_true_similarity(mem):
    """A fact one edge away from a hit enters scoring with its real cosine.

    Sixty fillers sit nearer the query than the fact, so the fact cannot
    enter through KNN. The edge from an anchoring concept must carry it in —
    and with its stored similarity, not zero. A control fact with no edge
    stays out: candidacy comes from the graph, not from a full scan.
    """
    for i in range(60):
        atom_id = mem.remember(f"office note {i}: invoices, spreadsheets and paperwork filing")
        _set(mem, atom_id, sti=0.0)
    anchor = mem.remember("office paperwork and invoices", type="concept")
    linked = mem.believe("Esmeralda is the youngest daughter of Nicolás", probability=0.95)
    control = mem.believe("Lourdes studies literature in Buenos Aires", probability=0.95)
    mem.atomspace.link_atoms(anchor, linked, "mentions", mem.tenant_id, mem.write_space)

    results = _recall(mem, "office paperwork and invoices", top_k=80)
    ids = {r.atom.id for r in results}
    assert linked in ids, "the linked fact never entered the candidate set"
    assert control not in ids, "an unlinked fact entered without KNN or an edge"
    linked_result = next(r for r in results if r.atom.id == linked)
    assert linked_result.similarity > 0.0, (
        "an expanded candidate was scored as if it had no embedding"
    )


def test_expansion_budget_prefers_high_standing_endpoints(mem):
    """A hub's trivia cannot evict the one edge that leads somewhere.

    120 edges to worthless endpoints exceed the 100-edge budget; the single
    edge to a high-standing fact must be among the ones that survive.
    """
    hub = mem.remember("the roadmap planning hub", type="concept")
    fact = mem.believe("the roadmap deadline is in March", probability=0.9)
    _set(mem, fact, confidence=0.9, sti=0.5, lti=0.5)
    mem.atomspace.link_atoms(hub, fact, "mentions", mem.tenant_id, mem.write_space)
    for i in range(120):
        junk = str(uuid.uuid4())
        mem.db.execute(
            """INSERT INTO atoms (id, type, label, tenant_id, space,
                                  probability, confidence, sti, lti, valence, intensity)
               VALUES (?, 'concept', ?, ?, ?, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)""",
            (junk, f"trivia {i}", mem.tenant_id, mem.write_space),
        )
        mem.atomspace.link_atoms(hub, junk, "associated", mem.tenant_id, mem.write_space)

    results = _recall(mem, "roadmap planning hub", top_k=150)
    assert fact in {r.atom.id for r in results}, (
        "the high-standing edge was evicted by the hub's trivia"
    )


# ── the KNN gate scales with the graph ────────────────────────────────────────

def test_knn_pool_scales_with_graph_size(mem):
    """A fixed 50-atom gate starves scoring once the space outgrows it."""
    for i in range(200):
        mem.remember(f"daily standup note {i}: progress on the widget assembly line")

    results = _recall(mem, "widget assembly progress notes", top_k=100)
    assert len(results) > 50, (
        f"only {len(results)} candidates passed the gate in a 200-atom space"
    )
