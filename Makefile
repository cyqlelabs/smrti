DATASET ?= data/longmemeval_s.json
HALUMEM ?= data/HaluMem-Medium.jsonl
BENCH_ARGS ?=

.PHONY: test datasets bench bench-baseline bench-halumem bench-decisions bench-tone bench-all distill distill-env distill-corpus distill-label distill-train distill-export distill-evaluate

test:
	pytest tests/ -q

# Fetch the benchmark datasets into data/. Neither ships with the repo, and
# without them every bench target below fails on a fresh clone.
datasets:
	PYTHONPATH=. python -m bench.fetch

# Retrieval regression harness. Required before releasing any change that
# touches retrieval; not a CI gate, since it needs the dataset and the
# embedding model. Fails when the retrieval hit rate drops against
# bench/longmemeval/baseline.json.
bench:
	PYTHONPATH=. python -m bench.longmemeval.run --dataset $(DATASET) $(BENCH_ARGS)

bench-baseline:
	PYTHONPATH=. python -m bench.longmemeval.run --dataset $(DATASET) --update-baseline $(BENCH_ARGS)

# HaluMem — what the system says when it does not know. Needs an answering
# model, and gates on the hallucination rate rising.
bench-halumem:
	PYTHONPATH=. python -m bench.halumem.run --dataset $(HALUMEM) $(BENCH_ARGS)

# The extraction routing gate against its labeled set — calls avoided
# beside missed durable claims, corrections and constraints. Uses the core
# local Laya runtime and fails when a recall floor is missed. A gate that saves
# calls by skipping valuable updates is a regression, so both are reported.
bench-decisions:
	PYTHONPATH=. python -m bench.decisions.run $(BENCH_ARGS)

bench-tone:
	PYTHONPATH=. python -m bench.decisions.tone $(BENCH_ARGS)

bench-all: bench bench-halumem

# Distilling Laya into the student decision model (bench/decisions/distill).
# `distill-env` builds the CUDA environment the trainer runs in; corpus and
# labelling run in the ordinary one against the teacher at SMRTI_DECISIONS_URL.
DISTILL_PY ?= $(HOME)/.venvs/smrti-distill/bin/python

distill-env:
	uv venv -q $(HOME)/.venvs/smrti-distill --python 3.12
	uv pip install -q --python $(DISTILL_PY) "torch==2.9.1+cu126" --index-url https://download.pytorch.org/whl/cu126
	uv pip install -q --python $(DISTILL_PY) "transformers>=4.45,<5" onnx onnxruntime onnxscript -e .

distill-corpus:
	PYTHONPATH=. python -m bench.decisions.distill corpus $(DISTILL_ARGS)

distill-label:
	PYTHONPATH=. python -m bench.decisions.distill label $(DISTILL_ARGS)

distill-train:
	PYTHONPATH=. $(DISTILL_PY) -m bench.decisions.distill train $(DISTILL_ARGS)

distill-export:
	PYTHONPATH=. $(DISTILL_PY) -m bench.decisions.distill export $(DISTILL_ARGS)

distill-evaluate:
	PYTHONPATH=. python -m bench.decisions.distill evaluate $(DISTILL_ARGS)

distill: distill-corpus distill-label distill-train distill-export distill-evaluate
