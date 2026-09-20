DATASET ?= data/longmemeval_s.json
HALUMEM ?= data/HaluMem-Medium.jsonl
BENCH_ARGS ?=

.PHONY: test datasets bench bench-baseline bench-halumem bench-decisions bench-all

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

bench-all: bench bench-halumem
