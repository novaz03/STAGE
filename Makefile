
.PHONY: setup fmt lint test

setup:
	pip install -e .

fmt:
	python -m pip install black ruff
	black src scripts tests
	ruff check --fix src scripts tests || true

lint:
	python -m pip install ruff
	ruff check src scripts tests

test:
	python -m pip install pytest
	pytest -q
