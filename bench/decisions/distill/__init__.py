"""Distilling Laya into the student (:mod:`smrti.decisions.student`).

    python -m bench.decisions.distill corpus     # states, from the corpora under data/
    python -m bench.decisions.distill label      # Laya's answers, over SMRTI_DECISIONS_URL
    python -m bench.decisions.distill train      # the encoder and its heads, on the GPU
    python -m bench.decisions.distill export     # ONNX int8 + student.json, packaged
    python -m bench.decisions.distill evaluate   # held-out agreement and the labelled sets
    python -m bench.decisions.distill all

Everything is keyed on the question registry
(:mod:`smrti.decisions.student.registry`): a new question is a registry
entry plus a state builder in :mod:`corpus`, and the pipeline re-run. Each
stage is resumable — states and labels are appended to JSONL keyed by id,
and a stage skips what an earlier run already produced — so re-running for
one new head costs that head's labels, not everyone's.
"""
