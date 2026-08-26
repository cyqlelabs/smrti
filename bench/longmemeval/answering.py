"""Optional end-to-end answering, kept apart from retrieval on purpose.

A judged answer needs a model and a key, so this half never gates anything —
``make bench`` measures retrieval and stops. What it is for is the other
direction: when retrieval holds steady and answers get worse, the answering
model moved, not the engine.
"""
from __future__ import annotations

import json

ANSWER_PROMPT = (
    "Answer the question using only the memories provided. "
    "Each memory is prefixed with the date it was recorded, and the question "
    "carries the date it was asked — use them to order events, to tell a later "
    "fact from the earlier one it replaced, and to answer anything relative "
    "(\"last month\", \"how long ago\"). "
    "When the question asks for a recommendation or a suggestion, make one, and "
    "make it follow what the memories say this person prefers — those questions "
    "are graded on whether the suggestion fits the person, not on naming a thing "
    "the memories already contain. "
    "Answer in one or two short sentences. Reply exactly \"I don't know\" only "
    "when the memories say nothing that bears on the question."
)

JUDGE_PROMPT = (
    "You are grading one answer against a reference. "
    'Reply with only valid JSON: {"correct": true} or {"correct": false}. '
    "When the reference states a fact, the answer is correct if it states the "
    "same fact, however it is worded. When the reference instead describes what "
    "this person would prefer, it is the grading rubric: the answer is correct "
    "if it satisfies that preference, and wrong if it ignores or contradicts it. "
    "An answer that declines to answer is never correct."
)


async def _chat(http, url: str, auth: str, model: str, system: str, user: str) -> str:
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = auth
    response = await http.post(
        f"{url}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        },
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    return (message.get("content") or "").strip()


async def answer_question(
    http,
    url: str,
    auth: str,
    model: str,
    question: str,
    memories: list[str],
    asked_on: str = "",
) -> str:
    """Answer from the recalled memories alone.

    ``asked_on`` is when the question was put. Without it a third of the
    benchmark is unanswerable however good retrieval was: "how long ago did I
    start" has no answer without a now, and two versions of a fact cannot be
    ordered without knowing which came later.
    """
    context = "\n".join(f"- {memory}" for memory in memories)
    header = f"[Question asked on]\n{asked_on}\n\n" if asked_on else ""
    return await _chat(
        http, url, auth, model, ANSWER_PROMPT,
        f"{header}[Memories]\n{context}\n\n[Question]\n{question}",
    )


async def judge_answer(
    http, url: str, auth: str, model: str, question: str, reference: str, answer: str
) -> bool:
    """True when the judge calls the answer correct.

    A judge that returns anything but the JSON it was asked for is read as a
    failure, not as a pass: a bench that scores unparseable output as correct
    reports the judge's mood.
    """
    raw = await _chat(
        http, url, auth, model, JUDGE_PROMPT,
        f"[Question]\n{question}\n\n[Reference]\n{reference}\n\n[Answer]\n{answer}",
    )
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return bool(json.loads(raw).get("correct"))
    except (ValueError, AttributeError):
        return False
