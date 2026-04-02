# AGENTS.md Update Development Roadmap

**Project**: juniper-data-client
**Date**: 2026-04-02
**Companion Documents**:
- Analysis: `notes/AGENTS_MD_AUDIT_ANALYSIS_2026-04-02.md`
- Plan: `notes/AGENTS_MD_UPDATE_PLAN_2026-04-02.md`

---

## Roadmap Summary

| Phase | Priority | Description | Tasks | Status |
|-------|----------|-------------|-------|--------|
| 1 | P0 | Critical Fixes | 2 | Pending |
| 2 | P1 | Structural Documentation | 4 | Pending |
| 3 | P2 | Supplementary Documentation | 3 | Pending |
| 4 | P3 | Metadata and Polish | 3 | Pending |
| 5 | -- | Validation | 3 | Pending |
| 6 | -- | Delivery | 2 | Pending |

---

## Phase 1: Critical Fixes (P0)

> Must be resolved immediately. These cause incorrect agent behavior.

### Task 1.1: Fix `__init__.py` version to 0.3.2

- **Priority**: P0
- **File**: `juniper_data_client/__init__.py:10`
- **Change**: `"0.3.1"` -> `"0.3.2"`
- **Rationale**: Version mismatch between package metadata and runtime value
- **Effort**: Trivial

### Task 1.2: Fix flake8 line-length in AGENTS.md commands

- **Priority**: P0
- **Section**: Quick Reference > Essential Commands
- **Change**: `--max-line-length=120` -> `--max-line-length=512`
- **Rationale**: Incorrect lint command causes false positives; contradicts pyproject.toml and ecosystem convention
- **Effort**: Trivial

---

## Phase 2: Structural Documentation (P1)

> Core content gaps that leave agents without essential context.

### Task 2.1: Add directory structure section

- **Priority**: P1
- **Action**: New section "Directory Structure" with annotated tree
- **Content coverage**: All top-level and significant nested directories
- **Effort**: Medium

### Task 2.2: Expand Key Files table

- **Priority**: P1
- **Action**: Expand from 4 entries to ~20 entries organized by category
- **Categories**: Core package, testing submodule, documentation, configuration, scripts, CI/CD, project meta
- **Effort**: Medium

### Task 2.3: Document complete Public API

- **Priority**: P1
- **Action**: Replace 3-method snippet with full categorized method reference
- **Methods to add**: 17+ methods across 9 categories (batch ops, versioning, generators, preview, readiness, etc.)
- **Effort**: Medium

### Task 2.4: Document testing submodule

- **Priority**: P1
- **Action**: New section "Testing Utilities" for `juniper_data_client.testing`
- **Content**: FakeDataClient class, 4 generator functions, import paths, usage patterns
- **Effort**: Medium

---

## Phase 3: Supplementary Documentation (P2)

> Adds depth for agents doing implementation or debugging work.

### Task 3.1: Add architecture and design patterns section

- **Priority**: P2
- **Content**: Connection management, retry strategy, URL normalization, error mapping, API key handling
- **Effort**: Medium

### Task 3.2: Add CI/CD section

- **Priority**: P2
- **Content**: 3 GitHub Actions workflows, Python version matrix, quality gate checks, publishing process
- **Effort**: Low

### Task 3.3: Document exception hierarchy with HTTP mapping

- **Priority**: P2
- **Content**: Exception tree, status code mapping table, import paths
- **Effort**: Low

---

## Phase 4: Metadata and Polish (P3)

> Low-priority cleanup items.

### Task 4.1: Update header Last Updated date

- **Priority**: P3
- **Change**: `2026-02-20` -> `2026-04-02`
- **Effort**: Trivial

### Task 4.2: Document utility scripts

- **Priority**: P3
- **Content**: scripts/check_doc_links.py, scripts/generate_dep_docs.sh, util/run_all_tests.bash
- **Effort**: Low

### Task 4.3: Add environment variables section

- **Priority**: P3
- **Content**: JUNIPER_DATA_API_KEY, JUNIPER_DATA_URL, .env.example reference
- **Effort**: Low

---

## Phase 5: Validation

### Task 5.1: Run full test suite with coverage

- **Command**: `pytest tests/ -v --cov=juniper_data_client --cov-report=term-missing --cov-fail-under=80`
- **Success criteria**: All tests pass, coverage >= 80%

### Task 5.2: Verify version consistency

- **Check**: `__init__.py.__version__` == `pyproject.toml[project].version` == `0.3.2`

### Task 5.3: Cross-reference AGENTS.md against codebase

- **Check**: Every directory, public method, config value, and workflow in the codebase has a corresponding AGENTS.md entry

---

## Phase 6: Delivery

### Task 6.1: Commit and push

- **Branch**: `chore/agents-md-audit`
- **Files**: AGENTS.md, juniper_data_client/__init__.py, notes/ deliverables

### Task 6.2: Create pull request

- **Base**: main
- **Title**: `docs: comprehensive AGENTS.md audit and update`
- **Body**: Summary of 15 drift items resolved, code fix for version mismatch
