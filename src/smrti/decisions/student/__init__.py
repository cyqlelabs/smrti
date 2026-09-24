"""The student decision model: Laya's answers to a fixed set of questions,
distilled into a 6-layer multilingual encoder small enough for a machine that
cannot run Laya.

Laya reads the question text with the state and answers anything; that is
what costs 22 layers over every token of both, once per question. The
student never reads a question. Every question the engines ask is fixed in
code (:mod:`registry`), so each one is a trained head over one reading of
the state: routing's three nouls are one pass, a rerank candidate's four are
one pass, and no question text rides in the input at all. On the 2011 dual
core where Laya takes 30–48 s per question, a full-window pass is 1.1 s and a
routing-sized one under half a second, at ~150 MB resident.

The price is that the student answers only the questions it was trained
on. A question outside the registry is refused, and the caller takes the
deterministic path it has for a provider that cannot answer. Adding a
question is a registry entry and a re-run of the trainer
(``bench/decisions/distill``).
"""
