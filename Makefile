DATASET ?= data/longmemeval_s.json
HALUMEM ?= data/HaluMem-Medium.jsonl
BENCH_ARGS ?=

.PHONY: test bench bench-baseline bench-halumem bench-all

test:
	pytest tests/ -q

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

bench-all: bench bench-halumem
