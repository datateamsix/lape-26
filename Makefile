.PHONY: test validate test-explorer test-python test-ts test-parity \
        corpus-setup corpus-lock corpus-relock corpus-pipeline corpus-check corpus-provenance-check

test: validate test-explorer test-python test-ts test-parity

validate:
	python3 scripts/validate_mapping.py

test-explorer:
	python3 scripts/check_word_explorer_mapping.py

test-python:
	PYTHONPATH=python python3 -m unittest discover -s python/tests -v

test-ts:
	node --experimental-strip-types --test packages/core-ts/test/core.test.ts

test-parity:
	node --experimental-strip-types scripts/check_cross_runtime.ts

corpus-setup:
	python3 scripts/setup_corpus.py setup

corpus-lock:
	python3 scripts/setup_corpus.py lock

corpus-relock:
	python3 scripts/setup_corpus.py relock

corpus-pipeline:
	python3 scripts/build_corpus_pipeline.py

corpus-check:
	python3 scripts/check_corpus_pipeline.py

corpus-provenance-check:
	python3 scripts/check_corpus_provenance.py
