# Operational Decisions

> **Single canonical record** of operational decisions made for this project. Audit
> lesson: do not duplicate this content into README / SECURITY / CONTRIBUTING.
> Those files **link here**; only the table below records the decision itself.

| Decision | Status | Decided | Revisit when | Rationale |
|---|---|---|---|---|
| Branch protection on `main` | **Deferred** | 2026-05-15 | Before Stage 5 (real-hardware integration, `v0.1.0-alpha`) | Solo operator chose faster iteration during Stages 1–4 scaffold + HAL + DOA + ML work. Self-review + CI-green is the interim discipline. |
| Secret-at-rest encryption (SOPS / age) | **Deferred** | 2026-05-15 | When a non-solo contributor joins | `.env` + `chmod 600`, gitignored, on encrypted TrueNAS-backed Proxmox host is sufficient for one operator. Process overhead of SOPS without proportional security gain. |
| PyPI publish | **Scaffolded, not triggered** | 2026-05-15 | At `v0.1.0` (post-Stage 7) | Per the Stage 1 handoff: workflow is reviewable now (`release.yml` with `if: false` guard); Trusted Publisher configured + the guard flipped at GA. |
| Python target | **3.11+** | 2026-05-15 | Permanent (revisit if Python 3.13 lands new features we need) | Matches orchestrator + aero; 3.11.2 available on the operator's host. CI tests 3.11 and 3.12. |
| Build backend | **Hatchling** | 2026-05-15 | Permanent | Stage 1 handoff recommendation; modern PEP 517 backend; clean extras handling. |
| Docstring style | **Google-style** | 2026-05-15 | Permanent (changing later is mechanical via ruff `--fix`) | Matches aero (consistency across consumer projects); ruff `[D]` rules enforce. |
| Codecov account | **Same as orchestrator** | 2026-05-15 | Permanent | Tokenless OSS GitHub App; one account = one dashboard for all consumer projects. |
| Pre-commit hook versions | **Pinned** | 2026-05-15 | Bump via Dependabot security advisories or explicit version-bump PR | Stage 1 handoff guidance; security: pinned versions prevent silent supply-chain churn. |
| Dependabot scope | **Security advisories only** at Stage 1 | 2026-05-15 | After v0.1.0 (consider Renovate for full version-bump cadence) | Avoid PR-noise floor during scaffold; security advisories still surface critical bumps. |
| GitHub Actions versioning | **Major-version float** (`@v4`, `@v6`) with Dependabot watch | 2026-05-15 | Permanent | Matches GitHub recommended pattern; Dependabot creates PRs on actual breaking changes. |

## How decisions are added or changed

1. A new row is appended (do **not** edit existing rows; mark them `**Superseded**`
   with a date and add a new row).
2. The relevant doc file (`SECURITY.md`, `CONTRIBUTING.md`, etc.) links here — it
   does **not** restate the decision.
3. The change lands via a `chore(docs): record operational decision <slug>` commit.
