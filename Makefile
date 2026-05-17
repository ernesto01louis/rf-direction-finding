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
#
# Stage 6 infrastructure (Ansible ecosystem hosting):
#   make provision             Provision the whole ecosystem (playbooks 00-08)
#   make provision-foundation  Provision the bootstrap + Traefik + Authelia layer
#   make infra-verify          Run the 99-verify end-to-end smoke playbook
#   make infra-lint            yamllint + ansible-lint the IaC tree

.PHONY: dev test lint format typecheck coverage docs verify clean \
        infra-deps provision provision-foundation infra-verify infra-lint

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

# --- Stage 6 infrastructure -------------------------------------------------
ANSIBLE_DIR ?= ansible
# The encrypted vault is unlocked at run time from ansible/.vault-pass.
export ANSIBLE_VAULT_PASSWORD_FILE = .vault-pass

infra-deps:
	cd $(ANSIBLE_DIR) && ansible-galaxy collection install -r requirements.yml

provision: infra-deps
	cd $(ANSIBLE_DIR) && for pb in playbooks/[0-8][0-9]-*.yml; do \
	  echo "== running $$pb =="; ansible-playbook "$$pb" || exit 1; done

provision-foundation: infra-deps
	cd $(ANSIBLE_DIR) && ansible-playbook playbooks/00-bootstrap.yml
	cd $(ANSIBLE_DIR) && ansible-playbook playbooks/01-platform.yml
	cd $(ANSIBLE_DIR) && ansible-playbook playbooks/06-dashboard.yml

infra-verify: infra-deps
	cd $(ANSIBLE_DIR) && ansible-playbook playbooks/99-verify.yml

infra-lint:
	yamllint -c .yamllint ansible/ docker-compose/
	cd $(ANSIBLE_DIR) && ansible-lint
