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
    "Answer in one short sentence. If the memories do not contain the answer, "
    "reply exactly: I don't know."
)

JUDGE_PROMPT = (
    "You are grading one answer against a reference answer. "
    'Reply with only valid JSON: {"correct": true} or {"correct": false}. '
    "The answer is correct when it states the same fact as the reference, "
    "however it is worded."
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
    http, url: str, auth: str, model: str, question: str, memories: list[str]
) -> str:
    context = "\n".join(f"- {memory}" for memory in memories)
    return await _chat(
        http, url, auth, model, ANSWER_PROMPT,
        f"[Memories]\n{context}\n\n[Question]\n{question}",
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
