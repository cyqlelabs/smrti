"""What the corpora lack, written by a cheap generative model: translations
of English turns into the languages users write in, and the rare kinds the
labelled sets care about — corrections, rules, apologies, stalled tool
calls — in each of them.

The corpora are English; the users are not. ``LANGS`` weights the
translations and the synthesis: Spanish first because both gateways run in
it, then the languages the encoder was distilled on and an agent is likely
to meet. The teacher answers all of them, so every language gets labels of
the same quality.

Every call is cached on disk under its prompt's hash
(``data/distill/cache``), so a re-run of the pipeline pays nothing for what
an earlier run already generated, and a dropped connection resumes where
it stopped. The teacher labels everything generated here; the generator's
own opinion of what it wrote is never used as a label.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import httpx

from .sources import DATA

logger = logging.getLogger("distill.augment")

MODEL = os.environ.get("SMRTI_DISTILL_LLM", "google/gemini-2.5-flash-lite")
URL = os.environ.get("SMRTI_DISTILL_LLM_URL", "https://openrouter.ai/api/v1/chat/completions")
CACHE = Path(DATA) / "distill" / "cache"
CONCURRENCY = 6


def _key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; augmentation needs it")
    return key


def _complete(prompt: str, *, temperature: float = 0.9, seed: int = 0) -> Any:
    """One JSON answer for *prompt*, from cache when it has been asked."""
    CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{MODEL}\n{seed}\n{prompt}".encode()).hexdigest()
    path = CACHE / f"{digest}.json"
    if path.exists():
        return json.loads(path.read_text())
    body = {
        "model": MODEL,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You write training data. Answer with one JSON object only."},
            {"role": "user", "content": prompt},
        ],
    }
    data: Any = {}
    for attempt in range(4):
        try:
            body["seed"] = seed + attempt  # a fresh sample, not the same broken one
            r = httpx.post(URL, json=body, timeout=httpx.Timeout(120.0, connect=20.0),
                           headers={"Authorization": f"Bearer {_key()}"})
            r.raise_for_status()
            data = _parse(r.json()["choices"][0]["message"]["content"])
            break
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("augmentation call failed (%s); %s", exc, "retrying" if attempt < 3 else "giving up on it")
            data = {}
    if not data:
        # Not cached: an empty answer is a gap the next run may fill.
        return {}
    path.write_text(json.dumps(data, ensure_ascii=False))
    global _calls
    _calls += 1
    if _calls % 25 == 0:
        logger.info("%d generation calls made", _calls)
    return data


_calls = 0


def _parse(text: str) -> Any:
    """The JSON object in a reply. A model asked for one object sometimes
    writes several in a row — one per item — which is read as the list it
    meant."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    pos = text.find("{")
    while pos != -1 and pos < len(text):
        obj, end = decoder.raw_decode(text, pos)
        objects.append(obj)
        pos = text.find("{", end)
    if not objects:
        raise ValueError("no JSON object in the reply")
    if len(objects) == 1 or any(k in objects[0] for k in ("items", "translations")):
        return objects[0]
    return {"items": objects}


def _many(prompts: list[tuple[str, int]], temperature: float = 0.9) -> list[Any]:
    with cf.ThreadPoolExecutor(CONCURRENCY) as ex:
        return list(ex.map(lambda p: _complete(p[0], temperature=temperature, seed=p[1]), prompts))


# ── languages ──────────────────────────────────────────────────────────────

# Target languages and their share of the non-English corpus.
LANGS: dict[str, float] = {
    "es": 0.34, "pt": 0.12, "fr": 0.08, "de": 0.08, "it": 0.06, "ru": 0.05,
    "ja": 0.06, "zh": 0.06, "hi": 0.05, "ar": 0.05, "tr": 0.05,
}
_LANG = {
    "en": "English", "es": "Spanish (Argentina, voseo)", "pt": "Portuguese (Brazil)", "fr": "French",
    "de": "German", "it": "Italian", "ru": "Russian", "ja": "Japanese", "zh": "Simplified Chinese",
    "hi": "Hindi", "ar": "Arabic", "tr": "Turkish",
}


def split_by_language(n: int, rng: random.Random) -> list[str]:
    """*n* language codes drawn by ``LANGS``' weights."""
    codes, weights = zip(*LANGS.items())
    return rng.choices(codes, weights=weights, k=n)


# ── translation ────────────────────────────────────────────────────────────


def translate(texts: list[str], lang: str, batch: int = 10) -> list[str]:
    """*lang* for each text, in order; a text that comes back missing is
    kept as it was rather than dropped, so indices line up."""
    prompts = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        prompts.append((
            f"Translate each of these chat messages into natural {_LANG[lang]}. "
            "Keep the register, length, names, numbers and any formatting. "
            "Answer as {\"translations\": [..]} with exactly one string per input, same order.\n\n"
            + json.dumps(chunk, ensure_ascii=False), 0))
    out: list[str] = []
    for chunk_start, data in zip(range(0, len(texts), batch), _many(prompts, temperature=0.3)):
        got = data.get("translations") if isinstance(data, dict) else None
        chunk = texts[chunk_start : chunk_start + batch]
        if not isinstance(got, list) or len(got) != len(chunk):
            got = chunk
        out.extend(str(t) for t in got)
    return out


def translate_es(texts: list[str]) -> list[str]:
    return translate(texts, "es")


# ── synthesis ──────────────────────────────────────────────────────────────


def _batches(n: int, per: int = 25) -> list[int]:
    return [per] * (n // per) + ([n % per] if n % per else [])


def synth(kind: str, lang: str, n: int, *, seed: int = 0) -> list[dict[str, Any]]:
    """*n* items of *kind* in *lang*, as the JSON objects the prompt for that
    kind asks for. Deterministic for a (kind, lang, n, seed)."""
    spec = _KINDS[kind]
    prompts = []
    rng = random.Random(f"{kind}/{lang}/{seed}")
    for k, per in enumerate(_batches(n)):
        hint = rng.choice(spec["angles"])
        prompts.append((
            f"{spec['prompt']}\n\nLanguage: {_LANG[lang]}. Write {per} items, varied in topic, length and "
            f"phrasing; this batch leans towards: {hint}. Answer as {{\"items\": [..]}}, each item an object "
            f"with exactly these fields: {spec['fields']}.", seed * 1000 + k))
    out: list[dict[str, Any]] = []
    for data in _many(prompts):
        for item in (data.get("items") if isinstance(data, dict) else None) or []:
            if isinstance(item, dict) and all(f in item for f in spec["required"]):
                out.append(item)
    return out


_ANGLES_LIFE = [
    "work and employers", "where people live and move", "family and pets", "health and allergies",
    "software, tools and infrastructure", "food, hobbies and travel", "dates, plans and deadlines",
    "money, purchases and contracts", "cars, homes and errands", "rules a team agreed on",
]

_KINDS: dict[str, dict[str, Any]] = {
    # Chat messages of the kinds the routing gate must tell apart.
    "message": {
        "prompt": "Write chat messages a person or an assistant might send in a long-running conversation with a "
                  "personal AI agent. Mix these kinds, marking each: durable (a fact, preference, goal or plan "
                  "about the speaker or someone else), correction (explicitly replaces or updates something "
                  "said earlier: 'actually', 'no longer', 'I moved', 'not X, Y'), constraint (a rule or "
                  "standing instruction: never, always, don't ever), chatter (thanks, ok, greetings, "
                  "reactions, one-off questions or requests that reveal nothing lasting), echo (an assistant "
                  "restating what it already knew about the user, or offering help).",
        "fields": "text (the message), author ('user' or 'assistant'), kind (durable|correction|constraint|chatter|echo)",
        "required": ("text", "author", "kind"),
        "angles": _ANGLES_LIFE,
    },
    # Sentences with a negative charge, of both kinds the tone check separates.
    "tone": {
        "prompt": "Write short messages with a negative tone from a conversation between a person and an AI "
                  "agent that runs tools, browses and sends reports. Half must be about a THING: a failure, "
                  "an error, a bug, a danger, a broken process, or a rule stated because something went wrong. "
                  "Half must be the SPEAKER's own mood in the moment: frustration, impatience, an apology, "
                  "thanks after trouble, a curt order, a rhetorical complaint. Include ones that are hard to "
                  "tell apart.",
        "fields": "text, author ('user' or 'assistant'), about ('thing' or 'speaker')",
        "required": ("text", "author", "about"),
        "angles": ["scripts and cron jobs", "emails and reports", "browsers and websites", "servers and "
                   "deployments", "household devices", "money and accounts", "everyday errands"],
    },
    # Pairs of claims about one subject, across every relation the check names.
    "claims": {
        "prompt": "Write pairs of claims about the same subject and predicate, as an agent's memory would record "
                  "them (subject, predicate in snake_case such as lives_in, works_for, prefers, uses, "
                  "has_pet, plans, allergic_to, avoids; object as a short phrase), plus the sentence the "
                  "later claim was read from. Cover all five relations evenly: same (the later says the "
                  "same thing, maybe reworded), update (a move, a new employer, a changed preference, a rule "
                  "withdrawn), compatible (both true at once: a visit is not a move, a second favourite, a "
                  "project-specific choice), contradiction (cannot both be true, later stated as current "
                  "fact), unclear (the sentence does not say enough to tell).",
        "fields": "subject, predicate, earlier_object, later_object, source_text (the sentence the later claim "
                  "came from), relation (same|update|compatible|contradiction|unclear)",
        "required": ("subject", "predicate", "earlier_object", "later_object", "source_text"),
        "angles": _ANGLES_LIFE,
    },
    # A tool call that keeps returning the same thing.
    "stall": {
        "prompt": "Write situations where an AI agent's tool call returned the same result three times in a row. "
                  "Give the user's task, the repeated call as name(arg=value, ...) using tools such as exec, "
                  "browser_navigate, browser_click, read_file, write_file, web_search, send_email, "
                  "job_status, recall, and the result text it kept returning (an error line starting with "
                  "'ERROR: ', a page that has not changed, a lock, a rate limit, an empty listing, a login "
                  "wall, a permission refusal, a timeout). Cover cases where waiting would help, where "
                  "re-reading state would, where another route is needed, and where only the user can "
                  "unblock it.",
        "fields": "task, repeated_call, result",
        "required": ("task", "repeated_call", "result"),
        "angles": ["websites and forms", "files and shell commands", "email and messaging", "background jobs",
                   "APIs and credentials", "devices on the local network"],
    },
    # A whole agent turn: task, trajectory in Factor's rendering, final reply.
    "trajectory": {
        "prompt": "Write agent turns: the user's request; the trajectory as lines in this exact format —\n"
                  "user: <message>\ntool call: name(arg=value, ...)\nresult: <what the tool returned, clipped>\n"
                  "assistant: <text the assistant wrote mid-turn, optional>\n— with two to six tool calls "
                  "(exec, read_file, write_file, browser_navigate, browser_click, browser_eval, web_search, "
                  "send_email, send_telegram, job_start, recall, remember, config_get); and the final reply. "
                  "Mix outcomes: replies that report exactly what the results show, replies that admit a "
                  "failure or ask the user something, and replies that claim something done, sent, saved or "
                  "fixed that the trajectory does not show (the call failed, was never made, or returned "
                  "something else). Some trajectories should be reusable multi-step procedures worth keeping "
                  "as a skill; most should be one-off errands.",
        "fields": "task, trajectory, reply",
        "required": ("task", "trajectory", "reply"),
        "angles": ["email and reports", "web scraping", "file edits and scripts", "system maintenance",
                   "shopping and bookings", "home automation", "research questions"],
    },
}
