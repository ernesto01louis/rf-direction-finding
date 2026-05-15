# CONTRIBUTING

Thanks for your interest in `rfdf`. The platform is open to contribution, but the
contribution surface is shaped by the platform-not-hub principle in [VISION.md](VISION.md):
keep core generic; domain-flavored work lives in your own consumer repo or in
`contrib/`.

## 1. Development setup

```bash
git clone https://github.com/ernesto01louis/rf-direction-finding.git
cd rf-direction-finding
python -m venv .venv
source .venv/bin/activate
make dev          # pip install -e '.[dev]' && pre-commit install
```

`make verify` runs the full local CI equivalent (ruff, mypy, pytest, coverage).

## 2. Code style

- **Python 3.11+**, type hints on all new code.
- **Ruff** for lint + format (`make lint`, `make format`).
- **mypy --strict** on `src/rfdf/` (`make typecheck`).
- **Line length 100.**
- **Docstrings: Google style** (`pydocstyle.convention = "google"` via ruff `[D]`
  rules).
- Pre-commit hooks enforce style on every commit. **Do not bypass with `--no-verify`
  unless explicitly authorized.**

## 3. Testing

- pytest. Markers: `hardware`, `slow`, `integration`, `gpu` (all skipped by default).
- **The `test-demo-no-hardware` CI job is the load-bearing gate.** Every algorithm and
  pipeline must work end-to-end on synthetic data. If your change breaks this job,
  you've coupled to hardware.
- For DOA / DSP / ML code, **property-based tests via Hypothesis** are encouraged for
  numerical correctness.
- Fixtures live in `tests/conftest.py`. Per-domain fixtures live alongside their tests.

## 4. Documentation

- Docstrings in Google style; ruff `[D]` rules check style.
- `mkdocs serve` for local doc preview (Stage 2+ once `mkdocs.yml` ships).
- Update `CHANGELOG.md` under `## [Unreleased]` for any user-visible change.

## 5. Branch + PR flow

1. Branch from `main`: `git checkout -b <type>/<short-slug>` (e.g.
   `feat/hal`, `fix/b210-clock-lock`).
2. Commit small, atomic changes following Conventional Commits.
3. Open a PR. CI must be green. One approval is required (solo operators: forced
   second-read of your own PR).
4. Squash-merge to `main`.

> **Stage 1 caveat:** branch protection on `main` is currently deferred. CI-green +
> self-review is enforced informally during scaffold/algo work; formal protection
> enables before Stage 5 (real-hardware integration). See
> [docs/operational-decisions.md](docs/operational-decisions.md).

## 6. Conventional Commits

Enforced by `commitlint-github-action` on PR titles and commits, and by `commitizen`
as a pre-commit hook.

**Format:** `<type>(<scope>): <description>`

**Allowed types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`,
`build`, `ci`, `revert`.

**Allowed scopes:** `hal`, `dsp`, `ml`, `backends`, `capture`, `cli`, `api`,
`orchestrator`, `config`, `docs`, `ci`, `tests`, `tools`, `examples`, `release`,
`deps`.

**Examples:**

```
feat(hal): add SdrSource Protocol class
fix(backends): handle empty SigMF metadata gracefully
docs(orchestrator): clarify lazy-import semantics
ci: pin ruff version in pre-commit config
```

Run `cz commit` for a guided commit message wizard.

## 7. Adding a new HAL backend

(Detailed guide ships in Stage 2 — `docs/adding-a-backend.md`.) Short version:

1. Pick a Protocol (`SdrSource`, `RotatorController`, `GeometryController`, or
   `ComputeBackend`).
2. Implement it in a new package under your fork or `contrib/rfdf-backend-<name>/`.
3. Register the entry point under `rfdf.backends.<group>`.
4. Pass the Protocol's property-based tests (Hypothesis fixtures from Stage 2).
5. Ship as `rfdf-backend-<name>` on PyPI or contribute to `contrib/`.

## 8. Adding a new compute backend

(Detailed guide ships in Stage 4 — `docs/ml/compute-backends.md`.)

## 9. Release process

Maintainers only:

1. Bump `version` in `pyproject.toml`.
2. Update `CHANGELOG.md`: move `[Unreleased]` content under `[<new-version>] - <date>`.
3. **Write `STAGE-N-OUTPUTS.md`** at the repo root capturing what actually shipped vs
   the stage handoff PDF. Convention defined in
   [STAGE-1-OUTPUTS.md §7](STAGE-1-OUTPUTS.md#7-convention-for-future-stages) — copy
   the seven sections verbatim and fill them in. **Skipping this is not allowed**;
   if a stage tags without it the next stage's session must produce it retroactively
   before any other work.
4. PR + merge.
5. Tag: `git tag v<version> -m "<one-line summary>"` then `git push origin v<version>`.
6. The `release.yml` workflow builds wheel + sdist and uploads to PyPI via Trusted
   Publishing (active from `v0.1.0` onward).

## 10. Code of conduct

Be kind. Be technically precise. Be patient with newcomers to RF / SDR / ML. We do not
have a separate CODE_OF_CONDUCT.md; the standard expectation is the
[Contributor Covenant](https://www.contributor-covenant.org/) at the operator's
discretion.
