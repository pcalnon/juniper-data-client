# AGENTS.md Audit Analysis

**Project**: juniper-data-client
**Audit Date**: 2026-04-02
**Auditor**: Claude Code (Opus 4.6)
**Branch**: chore/agents-md-audit
**AGENTS.md Last Updated**: 2026-02-20
**Current Package Version**: 0.3.2 (pyproject.toml) / 0.3.1 (__init__.py)

---

## Executive Summary

The juniper-data-client AGENTS.md file has significant drift from the current codebase state. The file was last updated on 2026-02-20 and has not been updated to reflect substantial feature additions (batch operations, dataset versioning, performance benchmarks), infrastructure changes (security hardening, CI/CD updates), documentation additions, and tooling changes made since that date.

**Severity**: Moderate-High. The AGENTS.md serves as the primary onboarding and operational guide for Claude Code agents working on this repository. Outdated information leads to incorrect assumptions, wasted context, and suboptimal agent behavior.

**Findings**: 15 drift items identified across 6 categories.

---

## Drift Analysis

### Category 1: Version and Metadata

| # | Finding | AGENTS.md States | Actual State | Severity |
|---|---------|-----------------|--------------|----------|
| 1 | **Version in AGENTS.md header** | 0.3.2 | 0.3.2 in pyproject.toml | OK |
| 2 | **Version in `__init__.py`** | Not referenced | `__version__ = "0.3.1"` while pyproject.toml has 0.3.2 | **High** |
| 3 | **Last Updated date** | 2026-02-20 | Significant changes made through 2026-04-02 | **Medium** |

**Analysis**: The `__init__.py` `__version__` string is out of sync with `pyproject.toml`. This is a code bug, not just a documentation issue. The AGENTS.md header version (0.3.2) matches pyproject.toml, so the AGENTS.md is correct, but `__init__.py` needs to be fixed.

---

### Category 2: Essential Commands (Quick Reference)

| # | Finding | AGENTS.md States | Actual State | Severity |
|---|---------|-----------------|--------------|----------|
| 4 | **Flake8 line length** | `--max-line-length=120` | `--max-line-length=512` (pyproject.toml, .pre-commit-config.yaml) | **High** |
| 5 | **Missing test runner script** | Not documented | `util/run_all_tests.bash` exists | **Low** |
| 6 | **Missing documentation validation** | Not documented | `scripts/check_doc_links.py` exists | **Low** |
| 7 | **Missing dependency docs script** | Not documented | `scripts/generate_dep_docs.sh` exists | **Low** |

**Analysis**: The flake8 line length discrepancy is the most critical finding here. An agent following the AGENTS.md command would use `--max-line-length=120`, which would produce false positives against the actual project standard of 512. The parent CLAUDE.md explicitly states "Line length: 512 for all linters" confirming 512 is the ecosystem standard.

---

### Category 3: Key Files and Directory Structure

| # | Finding | AGENTS.md States | Actual State | Severity |
|---|---------|-----------------|--------------|----------|
| 8 | **Key Files table** | Lists 4 files only (client.py, exceptions.py, __init__.py, tests/) | Repository contains 30+ significant files across 8+ directories | **High** |
| 9 | **No directory structure** | Absent | Complex layout: `juniper_data_client/`, `juniper_data_client/testing/`, `tests/`, `docs/`, `notes/`, `scripts/`, `util/`, `.github/workflows/`, `conf/` | **High** |
| 10 | **Testing submodule undocumented** | Not mentioned | `juniper_data_client/testing/` with `fake_client.py` (715 lines), `generators.py` (284 lines), `__init__.py` | **High** |

**Analysis**: The Key Files table is severely incomplete. The testing submodule (`juniper_data_client/testing/`) is a critical part of the package — it ships with the library and provides `FakeDataClient` and 4 synthetic generators used by consumers (juniper-cascor, juniper-canopy) for testing. Omitting it from AGENTS.md means agents are unaware of this significant public API surface.

Missing files/directories that should be documented:

- `juniper_data_client/testing/fake_client.py` — Drop-in mock client (715 lines)
- `juniper_data_client/testing/generators.py` — Synthetic dataset generators
- `juniper_data_client/py.typed` — PEP 561 type hint marker
- `docs/` — 4 documentation files (DOCUMENTATION_OVERVIEW, QUICK_START, REFERENCE, DEVELOPER_CHEATSHEET)
- `scripts/check_doc_links.py` — Documentation link validator
- `scripts/generate_dep_docs.sh` — Dependency documentation generator
- `util/run_all_tests.bash` — Test runner script
- `.github/workflows/ci.yml` — Main CI pipeline
- `.github/workflows/publish.yml` — PyPI publishing
- `.github/workflows/security-scan.yml` — Weekly security scanning
- `CHANGELOG.md` — Version history
- `.pre-commit-config.yaml` — Pre-commit hooks (20+ hooks)
- `.sops.yaml` — Secrets encryption config
- `.env.example` — Environment variables template
- `conf/` — Configuration directory

---

### Category 4: Public API Documentation

| # | Finding | AGENTS.md States | Actual State | Severity |
|---|---------|-----------------|--------------|----------|
| 11 | **Public API code example** | Shows 3 methods only (health_check, create_spiral_dataset, download_artifact_npz) | Client has 20+ public methods | **High** |
| 12 | **Missing API categories** | Not documented | Batch operations (4 methods), versioning (2 methods), generator discovery (2 methods), preview (1 method), readiness (2 methods) | **High** |

**Analysis**: The AGENTS.md Public API section shows a minimal 3-method example. The actual client has grown significantly since 2026-02-20 with the addition of:

**Batch Operations** (CAN-DEF-006):
- `batch_delete(dataset_ids)` — POST /v1/datasets/batch-delete
- `batch_create(datasets)` — POST /v1/datasets/batch-create
- `batch_update_tags(dataset_ids, add_tags, remove_tags)` — PATCH /v1/datasets/batch-tags
- `batch_export(dataset_ids)` — POST /v1/datasets/batch-export

**Dataset Versioning** (CAN-DEF-005):
- `list_versions(name)` — GET /v1/datasets/versions
- `get_latest(name)` — GET /v1/datasets/latest

**Generator Discovery**:
- `list_generators()` — GET /v1/generators
- `get_generator_schema(name)` — GET /v1/generators/{name}/schema

**Preview**:
- `get_preview(dataset_id, n)` — GET /v1/datasets/{id}/preview

**Readiness**:
- `is_ready()` — GET /v1/health/ready
- `wait_for_ready(timeout, poll_interval)` — Polling readiness check

---

### Category 5: Architecture and Design Patterns

| # | Finding | AGENTS.md States | Actual State | Severity |
|---|---------|-----------------|--------------|----------|
| 13 | **No architecture section** | Absent | Client implements retry strategy, connection pooling, error mapping, URL normalization, API key handling, context management | **Medium** |
| 14 | **Exception hierarchy incomplete** | Mentioned only in Key Files | Full hierarchy: JuniperDataClientError -> 5 specific exceptions with HTTP status code mapping | **Medium** |

**Analysis**: The AGENTS.md lacks an architecture or design patterns section. For an agent working on this codebase, understanding the retry strategy (HTTPAdapter with exponential backoff), error mapping pattern (HTTP status -> specific exception types), and URL normalization logic is essential context.

---

### Category 6: Infrastructure and CI/CD

| # | Finding | AGENTS.md States | Actual State | Severity |
|---|---------|-----------------|--------------|----------|
| 15 | **CI/CD not documented** | Absent | 3 GitHub Actions workflows: ci.yml (multi-version tests, security, quality gate), publish.yml (PyPI with attestations), security-scan.yml (weekly Bandit + pip-audit) | **Medium** |

**Analysis**: The AGENTS.md has no mention of the CI/CD infrastructure. Agents making changes should know:
- CI runs on Python 3.12, 3.13, 3.14 matrix
- Pre-commit hooks run in CI
- Coverage threshold is enforced (80%)
- Security scanning runs weekly
- Publishing uses trusted publishing (OIDC)

---

## Code Issues Discovered

### Issue 1: `__init__.py` Version Mismatch

- **File**: `juniper_data_client/__init__.py:10`
- **Current**: `__version__ = "0.3.1"`
- **Expected**: `__version__ = "0.3.2"` (to match pyproject.toml)
- **Impact**: Any code or consumer that reads `juniper_data_client.__version__` gets the wrong value
- **Fix**: Update `__init__.py` line 10 to `"0.3.2"`

---

## Summary of Required Changes

| Priority | Category | Items | Effort |
|----------|----------|-------|--------|
| **P0** | Code fix | `__init__.py` version mismatch | Trivial |
| **P0** | Commands | Fix flake8 line-length from 120 to 512 | Trivial |
| **P1** | Structure | Add complete directory layout section | Medium |
| **P1** | Key Files | Expand to cover all significant files | Medium |
| **P1** | Public API | Document all 20+ public methods | Medium |
| **P1** | Testing | Document testing submodule (FakeDataClient, generators) | Medium |
| **P2** | Architecture | Add architecture/design patterns section | Medium |
| **P2** | CI/CD | Add CI/CD documentation section | Low |
| **P2** | Exceptions | Document full exception hierarchy with HTTP mapping | Low |
| **P3** | Metadata | Update Last Updated date | Trivial |
| **P3** | Scripts | Document utility scripts | Low |
