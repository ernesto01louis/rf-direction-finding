# rfdf development Makefile.
#
# Usage:
#   make dev        Install in editable mode with [dev] extras and wire pre-commit
#   make test       Run pytest (unit + demo-no-hardware suites; hardware tests skipped)
#   make lint       Ruff check + format check
#   make format     Ruff fix + format (mutates files)
#   make typecheck  mypy --strict on src/rfdf
#   make coverage   pytest --cov with the project floor
#   make docs       Build mkdocs site (lands in Stage 2/3)
#   make verify     Full local CI equivalent (lint + typecheck + test + coverage)
#   make clean      Remove caches + build artifacts

.PHONY: dev test lint format typecheck coverage docs verify clean

PY ?= python

dev:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e '.[dev]'
	pre-commit install --install-hooks
	pre-commit install --hook-type commit-msg

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

coverage:
	pytest --cov --cov-report=term-missing --cov-report=xml

docs:
	@echo "mkdocs site lands in Stage 2/3; mkdocs.yml not yet written."
	@test -f mkdocs.yml && mkdocs build || echo "(skipped: no mkdocs.yml)"

verify: lint typecheck test coverage

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .hypothesis/ .coverage coverage.xml htmlcov/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
