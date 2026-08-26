"""Answering and grading, shared by every benchmark here.

Kept apart from retrieval on purpose. A model that answers well from a bad
candidate set hides the regression a retrieval gate exists to catch, so the
harnesses score the two separately and only ever gate on retrieval.

Calls run concurrently against one client. Sequentially, a 500-question
benchmark is an hour of waiting on a remote model that is idle between
requests; the work is independent per question and the only real limit is
the endpoint's rate limit, which ``concurrency`` bounds.
"""
from __future__ import annotations

import asyncio
import json

ANSWER_PROMPT = (
    "Answer the question using only the memories provided. "
    "Each memory is prefixed with the date it was recorded, and the question "
    "carries the date it was asked — use them to order events, to tell a later "
    "fact from the earlier one it replaced, and to answer anything relative "
    '("last month", "how long ago"). '
    "When the question asks for a recommendation or a suggestion, make one, and "
    "make it follow what the memories say this person prefers — those questions "
    "are graded on whether the suggestion fits the person, not on naming a thing "
    "the memories already contain. "
    'Answer in one or two short sentences. Reply exactly "I don\'t know" only '
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

# Three-way grading, because "wrong" and "silent" are not the same failure.
# A memory system that invents an answer is dangerous; one that admits it does
# not know is merely incomplete, and telling them apart is the whole point of
# a hallucination benchmark.
CLASSIFY_PROMPT = (
    "You are grading one answer against a reference. Reply with only valid "
    'JSON: {"verdict": "correct"}, {"verdict": "hallucination"}, or '
    '{"verdict": "omission"}.\n'
    '- "correct": the answer states what the reference states, however worded.\n'
    '- "omission": the answer declines — it says it does not know, or has no '
    "record — while the reference gives a real answer.\n"
    '- "hallucination": the answer asserts something the reference does not '
    "support, including inventing detail for a question the reference says is "
    "unanswerable.\n"
    "When the reference itself says the information is unknown or was never "
    'provided, declining is "correct" and asserting anything is '
    '"hallucination".'
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


def _parse_json(raw: str) -> dict:
    """The judge's reply as an object, or empty when it did not send one.

    A bench that reads unparseable output as a pass reports the judge's mood.
    """
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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

    ``asked_on`` is when the question was put. Without it a third of a
    long-horizon benchmark is unanswerable however good retrieval was: "how
    long ago did I start" has no answer without a now, and two versions of a
    fact cannot be ordered without knowing which came later.
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
    raw = await _chat(
        http, url, auth, model, JUDGE_PROMPT,
        f"[Question]\n{question}\n\n[Reference]\n{reference}\n\n[Answer]\n{answer}",
    )
    return bool(_parse_json(raw).get("correct"))


async def classify_answer(
    http, url: str, auth: str, model: str, question: str, reference: str, answer: str
) -> str:
    """One of correct / hallucination / omission.

    An unreadable verdict counts as a hallucination rather than being dropped:
    the run's denominators stay honest, and the benefit of the doubt is the
    one thing a hallucination score must not hand out.
    """
    raw = await _chat(
        http, url, auth, model, CLASSIFY_PROMPT,
        f"[Question]\n{question}\n\n[Reference]\n{reference}\n\n[Answer]\n{answer}",
    )
    verdict = _parse_json(raw).get("verdict")
    return verdict if verdict in ("correct", "hallucination", "omission") else "hallucination"


async def score_batch(
    answering: dict, items: list[dict], verdict: str = "binary", concurrency: int = 8
) -> list[dict]:
    """Answer and grade every item, at most *concurrency* in flight.

    Each item needs ``question``, ``reference`` and ``memories``, and may carry
    ``asked_on``. Results come back in the order they were given, each as
    ``{"answer": str, "verdict": bool | str}``, so a caller can zip them
    straight back onto its rows.
    """
    import httpx

    if not items:
        return []
    gate = asyncio.Semaphore(max(1, concurrency))

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as http:

        async def _one(item: dict) -> dict:
            async with gate:
                answer = await answer_question(
                    http, answering["url"], answering["auth"], answering["model"],
                    item["question"], item["memories"], item.get("asked_on", ""),
                )
                grade = classify_answer if verdict == "three_way" else judge_answer
                graded = await grade(
                    http, answering["url"], answering["auth"],
                    answering["judge_model"], item["question"],
                    item["reference"], answer,
                )
                return {"answer": answer, "verdict": graded}

        return await asyncio.gather(*(_one(item) for item in items))
