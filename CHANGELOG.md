# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project (from `v0.1.0` onward) adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-`v0.1.0` releases (`v0.0.x`) are pre-alpha — breaking changes are allowed and
recorded here per the release.

## [Unreleased]

## [0.0.1] - 2026-05-15

### Added

- Repository scaffold: full directory tree per the Stage 1 handoff bundle
  (`src/rfdf/{hal,backends,dsp,ml,capture,api,cli,orchestrator}`,
  `tests/{unit,integration,hardware,demo_no_hardware}`, `examples/`, `docs/`,
  `ansible/`, `docker-compose/`, `contrib/`, `recipes/`).
- Meta documents: `README.md`, `VISION.md`, `ARCHITECTURE.md`, `ROADMAP.md`,
  `CLAUDE.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md` (this file).
- `AGENTS.md` symlink to `CLAUDE.md` (cross-tool AI-agent context).
- `pyproject.toml` with Hatchling build backend, Python `>=3.11`, **zero RF or ML
  base dependencies**, and the full extras matrix: `sdr-uhd`, `sdr-pluto`,
  `sdr-soapy`, `rotator-hamlib`, `ml`, `ml-onnx`, `ml-hailo`, `compute-runpod`,
  `compute-modal`, `compute-vastai`, `compute-skypilot`, `orchestrator`, `api`,
  `antenna`, `dev`, `docs`, plus convenience meta-extras `all-hardware`,
  `all-compute`, `all`.
- `rfdf` CLI entry point (Typer) with `--version` wired. Subcommands land per stage.
- CI workflows: `ci.yml` (lint, test-base on 3.11 + 3.12 matrix, test-demo-no-hardware,
  test-orchestrator, coverage upload to Codecov, readme-status-truthful,
  conventional-commits, zero-domain-deps), `release.yml` (PyPI Trusted Publishing —
  scaffolded but **inactive** until `v0.1.0`), `pre-commit.yml`, `docs.yml`.
- Pre-commit config: pinned `pre-commit-hooks v5.0.0`, `ruff-pre-commit v0.7.4`,
  `mirrors-mypy v1.13.0`, `commitizen v3.31.0`.
- Conventional Commits: `cz.toml` with locked scope list, `commitlint-github-action`
  on PRs.
- Codecov config (`codecov.yml`), Dependabot config for security advisories only.
- Apache-2.0 LICENSE (auto-created at repo init).
- `Makefile` with `dev / test / lint / format / typecheck / coverage / docs / clean /
  verify` targets.
- `.gitignore`, `.gitattributes`, `.editorconfig`.
- `docs/operational-decisions.md` capturing the branch-protection-deferred and
  SOPS-deferred decisions in **one canonical place** (kills the audit-lesson
  decision-carried-in-two-places pattern).

### Deferred (operator decision, documented in `docs/operational-decisions.md`)

- **Branch protection on `main`** — deferred to enable faster iteration during
  Stages 1–4 (scaffold + HAL + DOA + ML). Revisited and enabled before Stage 5
  (real-hardware integration).
- **SOPS / age-based secret-at-rest encryption** — deferred until a non-solo
  contributor joins. `.env` + `chmod 600` is the interim mechanism. See
  [SECURITY.md](SECURITY.md) §2.
- **Actual PyPI publish** — workflow scaffolded; first publish at `v0.1.0` (post-Stage
  7) per the Stage 1 handoff guidance.

[Unreleased]: https://github.com/ernesto01louis/rf-direction-finding/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/ernesto01louis/rf-direction-finding/releases/tag/v0.0.1
