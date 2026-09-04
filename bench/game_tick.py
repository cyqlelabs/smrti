"""What one game tick costs the engine, and what a fresh citizen recalls.

Backs ``docs/game-engine-analysis.md``. Builds a small town — one ``World_Space``
with eight places, N citizen spaces with random presets — writes M experiences
per citizen with a stated valence, then times the calls a game loop makes:
``remember``, ``recall`` with and without the access boost, ``reflect``, and
one ``space_overlap``. It also prints each citizen's place moods before and
after the writes, which is the loop smrti-town leaves open (see the analysis).

    python bench/game_tick.py            # real embedding model, 5 citizens x 200
    python bench/game_tick.py 5 500      # 5 citizens x 500 experiences
    python bench/game_tick.py --stub     # hashed stand-in embedder: everything but ONNX

``--stub`` exists for hosts that cannot reach the model; its timings exclude
inference and its similarities are lexical, so read them beside the project's
own figure for a full ``remember`` with the real model (about 18 ms).
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import re
import statistics
import sys
import tempfile
import time

DIM = 384


class StubEmbedder:
    """Hashed bag-of-words + character trigrams, L2-normalised. Not a model."""

    dimensions = DIM

    @staticmethod
    def _feats(text: str) -> list[str]:
        toks = re.findall(r"\w+", text.casefold())
        feats = list(toks)
        for t in toks:
            t = f"#{t}#"
            feats += [t[i:i + 3] for i in range(len(t) - 2)]
        return feats

    def embed(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for f in self._feats(text):
            h = hashlib.blake2b(f.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % DIM
            v[idx] += 1.0 if h[4] & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


PLACES = ["Tavern", "Bakery", "Town Hall", "Library", "Clinic", "Market", "Park", "Church"]
PEOPLE = ["Alice", "Bruno", "Chen", "Dolores", "Emeka", "Farah", "Gustavo", "Hana",
          "Ivan", "Jia", "Kofi", "Lena", "Mateo", "Nadia", "Omar", "Priya", "Quinn",
          "Rosa", "Sven", "Tariq"]
VERBS_POS = ["had a wonderful meal at", "laughed with friends at", "found peace at", "sold well at"]
VERBS_NEG = ["was insulted at", "got food poisoning at", "was robbed near", "argued bitterly at"]


def _ms(samples: list[float]) -> str:
    p95 = sorted(samples)[max(0, int(len(samples) * 0.95) - 1)]
    return f"p50={statistics.median(samples) * 1000:.1f}ms p95={p95 * 1000:.1f}ms"


def _place_moods(mind, places: list[str]) -> dict[str, float]:
    """The town's ``_memory_mood``: mean drifting valence of memories naming the place."""
    out = {}
    for p in places:
        res = mind.recall(p, top_k=5, min_confidence=0.1, boost=False)
        m = [r.atom.valence.valence for r in res
             if p.casefold() in (r.atom.content or r.atom.label).casefold()]
        out[p] = statistics.mean(m) if m else 0.0
    return out


def main(n_citizens: int = 5, n_episodes: int = 200, stub: bool = False) -> None:
    if stub:
        import smrti.core.embed as embed_mod
        embed_mod._singleton = StubEmbedder()
    from smrti import Smrti

    random.seed(7)
    db = os.path.join(tempfile.mkdtemp(prefix="smrti-game-tick-"), "town.db")
    tenant = "bench"
    world = Smrti(db_path=db, tenant_id=tenant, write_space="World_Space")
    for p in PLACES:
        world.remember(f"{p} is a place in town", type="concept", probability=1.0, valence=0.1)

    citizens = []
    for name in PEOPLE[:n_citizens]:
        mind = Smrti(
            db_path=db, tenant_id=tenant,
            personality=random.choice(["balanced", "empathetic", "analytical", "curious"]),
            write_space=f"Agent_Space_{name}",
            read_spaces=[f"Agent_Space_{name}", "World_Space"],
        )
        mind.remember(f"{name} is a citizen who moved here last spring.", probability=0.9, valence=0.1)
        citizens.append((name, mind))

    name, mind = citizens[0]
    res = mind.recall("Tavern", top_k=5, min_confidence=0.1, boost=False)
    print(f"[fresh] {name} recalls 'Tavern': "
          + "; ".join(f"{r.atom.space}: {r.atom.label[:32]!r} v={r.atom.valence.valence:+.2f}" for r in res))
    moods = _place_moods(mind, PLACES)
    print("[fresh] place moods: " + ", ".join(f"{p}={v:+.2f}" for p, v in moods.items()))

    t = []
    for _ in range(30):
        t0 = time.perf_counter(); mind.embed.embed("at the Tavern with Alice"); t.append(time.perf_counter() - t0)
    print(f"[embed{' STUB' if stub else ''}] one query embedding: {_ms(t)}")

    w = []
    for name, mind in citizens:
        for _ in range(n_episodes):
            place = random.choice(PLACES)
            neg = random.random() < 0.3
            verb = random.choice(VERBS_NEG if neg else VERBS_POS)
            v = random.uniform(-0.9, -0.4) if neg else random.uniform(0.2, 0.8)
            t0 = time.perf_counter()
            mind.remember(f"{name} {verb} {place} with {random.choice(PEOPLE[:n_citizens])}",
                          valence=v, probability=0.9)
            w.append(time.perf_counter() - t0)
    total = mind.db.fetchone("SELECT COUNT(*) AS n FROM atoms")["n"]
    print(f"[remember] {len(w)} experiences across {n_citizens} minds ({total} atoms in file): {_ms(w)}")

    r_boost, r_mood = [], []
    for name, mind in citizens:
        for _ in range(40):
            place = random.choice(PLACES)
            t0 = time.perf_counter(); mind.recall(f"at {place} with Alice", top_k=5); r_boost.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); mind.recall(place, top_k=5, min_confidence=0.1, boost=False); r_mood.append(time.perf_counter() - t0)
    print(f"[recall] decision recall (top_k=5, boost): {_ms(r_boost)}")
    print(f"[recall] mood recall (top_k=5, no boost): {_ms(r_mood)}")

    name, mind = citizens[0]
    moods = _place_moods(mind, PLACES)
    print(f"[after writes] {name}: " + ", ".join(f"{p}={v:+.2f}" for p, v in moods.items()))

    ep = []
    for _, m in citizens:
        t0 = time.perf_counter(); r = m.reflect(); ep.append(time.perf_counter() - t0)
    print(f"[reflect] one epoch per mind (~{n_episodes} atoms each): {_ms(ep)}  "
          f"(last: decayed={r.atoms_decayed} promoted={r.lti_promoted} pruned={r.atoms_pruned})")
    ep10 = []
    for _ in range(10):
        t0 = time.perf_counter(); mind.reflect(); ep10.append(time.perf_counter() - t0)
    print(f"[reflect] 10 more epochs on {name}: {_ms(ep10)}")
    moods = _place_moods(mind, PLACES)
    print(f"[after 11 epochs] {name}: " + ", ".join(f"{p}={v:+.2f}" for p, v in moods.items()))
    st = mind.status()
    print(f"[status] {name}: {st['total_atoms']} atoms by type {st['by_type']}")

    if len(citizens) > 1:
        a, b = citizens[0][1], citizens[1][1]
        t0 = time.perf_counter(); ov = a.space_overlap(b.write_space, threshold=0.85); dt = time.perf_counter() - t0
        print(f"[space_overlap] {a.write_space} vs {b.write_space}: "
              f"jaccard={ov.jaccard:.3f} pairs={len(ov.pairs)} in {dt * 1000:.0f}ms")

    mind.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    size = os.path.getsize(db) + (os.path.getsize(db + "-wal") if os.path.exists(db + "-wal") else 0)
    total = mind.db.fetchone("SELECT COUNT(*) AS n FROM atoms")["n"]
    print(f"[disk] {size / 1024:.0f} KiB checkpointed for {total} atoms = {size / total / 1024:.1f} KiB per atom")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        int(args[0]) if len(args) > 0 else 5,
        int(args[1]) if len(args) > 1 else 200,
        stub="--stub" in sys.argv,
    )
