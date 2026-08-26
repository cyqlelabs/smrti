"""Tests for reinforcement: being used is evidence a memory is worth keeping.

Before this, confidence had exactly one way up — the caller restating the
fact — and one way down, which every atom takes by default. An atom that
sinks below the surfacing floor can never be recalled, so it can never be
restated, so nothing lifts it back.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from smrti import Smrti
from smrti.evolution.reinforcement import CAP_PER_EPOCH, DEFAULT_WEIGHT


@pytest.fixture
def mem(tmp_path):
    return Smrti(
        db_path=str(tmp_path / "reinforce.db"),
        personality="balanced",
        tenant_id="test",
        write_space="default",
    )


def _confidence(mem, atom_id: str) -> float:
    return mem.db.fetchone(
        "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
    )["confidence"]


def _probability(mem, atom_id: str) -> float:
    return mem.db.fetchone(
        "SELECT probability FROM atoms WHERE id = ?", (atom_id,)
    )["probability"]


def _metadata(mem, atom_id: str) -> dict:
    return json.loads(
        mem.db.fetchone("SELECT metadata FROM atoms WHERE id = ?", (atom_id,))["metadata"]
    )


def _sink(mem, atom_id: str, confidence: float) -> None:
    mem.db.execute(
        "UPDATE atoms SET confidence = ? WHERE id = ?", (confidence, atom_id)
    )


def _mark_agent(mem, atom_id: str) -> None:
    mem.db.execute(
        "UPDATE atoms SET metadata = ? WHERE id = ?",
        (json.dumps({"source": "agent"}), atom_id),
    )


# ── the update ───────────────────────────────────────────────────────────────


def test_being_used_raises_confidence(mem):
    atom_id = mem.believe("Esmeralda is my daughter", probability=0.9)
    before = _confidence(mem, atom_id)

    mem.reinforce([atom_id])

    assert _confidence(mem, atom_id) > before


def test_being_used_leaves_the_probability_alone(mem):
    atom_id = mem.believe("Esmeralda is my daughter", probability=0.9)

    mem.reinforce([atom_id])

    assert _probability(mem, atom_id) == pytest.approx(0.9)


def test_repeated_use_converges_instead_of_exploding(mem):
    atom_id = mem.believe("Esmeralda is my daughter", probability=0.9)

    for _ in range(200):
        mem.db.execute(
            "UPDATE atoms SET metadata = json_remove(metadata, '$.reinforced_count') "
            "WHERE id = ?",
            (atom_id,),
        )
        mem.reinforce([atom_id])

    assert _confidence(mem, atom_id) <= 1.0


def test_reports_lift_a_floor_sitting_belief_back_above_the_floor(mem):
    """The death spiral, closed: repeated use returns an unsurfaceable fact."""
    atom_id = mem.believe("Roxana is my partner", probability=0.8)
    floor = mem.db.fetchone(
        "SELECT min_confidence_to_surface AS f FROM personality "
        "WHERE tenant_id = 'test' AND space = 'default'"
    )["f"]
    _sink(mem, atom_id, floor)
    assert mem.recall("Roxana", min_confidence=floor + 0.05) == []

    for epoch in range(4):
        mem.db.execute(
            "UPDATE personality SET epoch_count = ? WHERE tenant_id = ? AND space = ?",
            (epoch, "test", "default"),
        )
        mem.reinforce([atom_id])

    assert _confidence(mem, atom_id) > floor + 0.05
    assert mem.recall("Roxana", min_confidence=floor + 0.05)


def test_a_bigger_weight_moves_confidence_further(mem):
    modest = mem.believe("a modestly used fact", probability=0.8)
    heavy = mem.believe("a heavily used fact", probability=0.8)

    mem.reinforce([modest], weight=DEFAULT_WEIGHT)
    mem.reinforce([heavy], weight=1.0)

    assert _confidence(mem, heavy) > _confidence(mem, modest)


def test_a_weight_of_zero_changes_nothing(mem):
    atom_id = mem.believe("Esmeralda is my daughter", probability=0.9)
    before = _confidence(mem, atom_id)

    mem.reinforce([atom_id], weight=0.0)

    assert _confidence(mem, atom_id) == before


def test_an_absurd_weight_is_clamped_to_one(mem):
    absurd = mem.believe("a fact reported with a silly weight", probability=0.8)
    honest = mem.believe("a fact reported at full weight", probability=0.8)

    mem.reinforce([absurd], weight=99.0)
    mem.reinforce([honest], weight=1.0)

    assert _confidence(mem, absurd) == pytest.approx(_confidence(mem, honest))


# ── the guards ───────────────────────────────────────────────────────────────


def test_agent_authored_atoms_take_the_source_discount(mem):
    user_atom = mem.believe("the user said this", probability=0.8)
    agent_atom = mem.believe("the model said this", probability=0.8)
    _mark_agent(mem, agent_atom)
    _sink(mem, agent_atom, _confidence(mem, user_atom))

    mem.reinforce([user_atom, agent_atom])

    assert _confidence(mem, agent_atom) < _confidence(mem, user_atom)


def test_a_deliberately_forgotten_memory_is_never_reinforced(mem):
    atom_id = mem.believe("something the user asked to drop", probability=0.8)
    mem.forget("something the user asked to drop")
    before = _confidence(mem, atom_id)

    result = mem.reinforce([atom_id])

    assert _confidence(mem, atom_id) == before
    assert result["skipped"] == [{"id": atom_id, "reason": "forgotten"}]


def test_an_atom_can_only_bank_so_many_reports_per_epoch(mem):
    atom_id = mem.believe("a memory used every single turn", probability=0.8)

    for _ in range(CAP_PER_EPOCH):
        assert mem.reinforce([atom_id])["reinforced"] == [atom_id]
    capped = _confidence(mem, atom_id)

    result = mem.reinforce([atom_id])

    assert result["skipped"] == [{"id": atom_id, "reason": "capped"}]
    assert _confidence(mem, atom_id) == capped


def test_a_consolidation_pass_refreshes_the_cap(mem):
    atom_id = mem.believe("a memory used every single turn", probability=0.8)
    for _ in range(CAP_PER_EPOCH):
        mem.reinforce([atom_id])

    mem.reflect()

    assert mem.reinforce([atom_id])["reinforced"] == [atom_id]


def test_the_same_id_reported_twice_in_one_call_counts_once(mem):
    atom_id = mem.believe("Esmeralda is my daughter", probability=0.9)

    result = mem.reinforce([atom_id, atom_id, atom_id])

    assert result["reinforced"] == [atom_id]
    assert _metadata(mem, atom_id)["reinforced_count"] == 1


def test_an_id_from_another_space_is_unknown_here(tmp_path):
    shared = str(tmp_path / "spaces.db")
    home = Smrti(db_path=shared, tenant_id="test", write_space="home")
    work = Smrti(db_path=shared, tenant_id="test", write_space="work")
    atom_id = home.believe("Esmeralda is my daughter", probability=0.9)
    before = _confidence(home, atom_id)

    result = work.reinforce([atom_id])

    assert result == {"reinforced": [], "skipped": [{"id": atom_id, "reason": "unknown"}]}
    assert _confidence(home, atom_id) == before


def test_an_id_from_another_tenant_is_unknown_here(tmp_path):
    shared = str(tmp_path / "tenants.db")
    theirs = Smrti(db_path=shared, tenant_id="theirs", write_space="default")
    mine = Smrti(db_path=shared, tenant_id="mine", write_space="default")
    atom_id = theirs.believe("their private fact", probability=0.9)
    before = _confidence(theirs, atom_id)

    mine.reinforce([atom_id])

    assert _confidence(theirs, atom_id) == before


def test_malformed_metadata_does_not_stop_the_report(mem):
    atom_id = mem.believe("an atom with unreadable metadata", probability=0.8)
    mem.db.execute("UPDATE atoms SET metadata = 'not json' WHERE id = ?", (atom_id,))
    before = _confidence(mem, atom_id)

    result = mem.reinforce([atom_id])

    assert result["reinforced"] == [atom_id]
    assert _confidence(mem, atom_id) > before


def test_an_unreadable_reinforcement_count_does_not_stop_the_report(mem):
    atom_id = mem.believe("an atom with a mangled counter", probability=0.8)
    mem.db.execute(
        "UPDATE atoms SET metadata = json(?) WHERE id = ?",
        (json.dumps({"reinforced_epoch": 0, "reinforced_count": "lots"}), atom_id),
    )

    assert mem.reinforce([atom_id])["reinforced"] == [atom_id]


def test_metadata_that_is_valid_json_but_not_an_object_is_ignored(mem):
    """`json_valid` passes a bare list; the code still has to survive it."""
    atom_id = mem.believe("an atom whose metadata is a list", probability=0.8)
    mem.db.execute("UPDATE atoms SET metadata = '[1, 2]' WHERE id = ?", (atom_id,))
    before = _confidence(mem, atom_id)

    assert mem.reinforce([atom_id])["reinforced"] == [atom_id]
    assert _confidence(mem, atom_id) > before


def test_reporting_nothing_does_nothing(mem):
    assert mem.reinforce([]) == {"reinforced": [], "skipped": []}


# ── over the wire ────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    import smrti.servers.rest as rest
    from smrti.servers import config as cfg

    monkeypatch.setattr(cfg, "DB", str(tmp_path / "rest.db"))
    monkeypatch.setattr(cfg, "EXTRACT", False)
    monkeypatch.setattr(cfg, "TEMPORAL", False)
    monkeypatch.setattr(rest, "_mem", None)
    rest._space_mems.clear()
    with TestClient(rest.app) as c:
        yield c


def test_the_endpoint_reinforces_the_named_atoms(client):
    stored = client.post(
        "/remember", json={"content": "Esmeralda is my daughter", "type": "belief"}
    ).json()

    response = client.post("/reinforce", json={"atom_ids": [stored["atom_id"]]})

    assert response.status_code == 200
    assert response.json()["reinforced"] == [stored["atom_id"]]


def test_the_endpoint_reports_what_it_skipped(client):
    response = client.post("/reinforce", json={"atom_ids": ["no-such-atom"]})

    assert response.json()["skipped"] == [{"id": "no-such-atom", "reason": "unknown"}]


def test_the_endpoint_accepts_a_weight(client):
    stored = client.post(
        "/remember", json={"content": "Esmeralda is my daughter", "type": "belief"}
    ).json()

    response = client.post(
        "/reinforce", json={"atom_ids": [stored["atom_id"]], "weight": 0.5}
    )

    assert response.status_code == 200


def test_the_endpoint_refuses_an_empty_batch(client):
    assert client.post("/reinforce", json={"atom_ids": []}).status_code == 422


def test_the_endpoint_refuses_a_weight_outside_the_unit_interval(client):
    assert (
        client.post(
            "/reinforce", json={"atom_ids": ["x"], "weight": 5.0}
        ).status_code
        == 422
    )
