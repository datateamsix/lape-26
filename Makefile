.PHONY: test validate test-explorer test-python test-ts test-parity

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
