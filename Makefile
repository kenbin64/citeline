.PHONY: install lint typecheck test check migrate ingest eval serve clean

VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"

lint:
	$(VENV)/bin/ruff check src eval tests
	$(VENV)/bin/ruff format --check src eval tests

typecheck:
	$(VENV)/bin/mypy src/citeline

test:
	$(PY) -m pytest -q

# What CI runs, and what should pass before a deploy.
check: lint typecheck test

migrate:
	$(VENV)/bin/citeline migrate

ingest:
	$(VENV)/bin/citeline ingest

eval:
	$(PY) eval/run_eval.py

eval-retrieval:
	$(PY) eval/run_eval.py --retrieval-only

serve:
	$(VENV)/bin/citeline serve

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
