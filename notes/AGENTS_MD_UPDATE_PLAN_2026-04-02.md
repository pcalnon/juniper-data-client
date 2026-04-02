# AGENTS.md Update Plan

**Project**: juniper-data-client
**Date**: 2026-04-02
**Companion Analysis**: `notes/AGENTS_MD_AUDIT_ANALYSIS_2026-04-02.md`
**Branch**: chore/agents-md-audit

---

## Objective

Bring the juniper-data-client AGENTS.md file into full alignment with the current codebase state, resolving all 15 drift items identified in the audit analysis.

---

## Phase 1: Critical Fixes (P0)

### Step 1.1: Fix `__init__.py` Version Mismatch

- **File**: `juniper_data_client/__init__.py`
- **Action**: Change `__version__ = "0.3.1"` to `__version__ = "0.3.2"`
- **Validation**: Run `python -c "import juniper_data_client; print(juniper_data_client.__version__)"` and verify output is `0.3.2`

### Step 1.2: Fix Essential Commands — Flake8 Line Length

- **File**: `AGENTS.md`
- **Section**: Quick Reference > Essential Commands
- **Action**: Change `--max-line-length=120` to `--max-line-length=512`
- **Validation**: Verify matches pyproject.toml `[tool.black]` line-length, `.pre-commit-config.yaml` flake8 args, and parent CLAUDE.md convention

---

## Phase 2: Structural Documentation (P1)

### Step 2.1: Add Directory Structure Section

- **File**: `AGENTS.md`
- **Action**: Add a new "Directory Structure" section after "Project Overview" showing the full repository layout
- **Content**: Tree diagram of all directories and key files with purpose annotations
- **Includes**: `juniper_data_client/`, `juniper_data_client/testing/`, `tests/`, `docs/`, `notes/`, `scripts/`, `util/`, `.github/`, `conf/`

### Step 2.2: Expand Key Files Table

- **File**: `AGENTS.md`
- **Section**: Key Files
- **Action**: Expand from 4 entries to comprehensive table covering:
  - Core package files (client.py, exceptions.py, __init__.py, py.typed)
  - Testing submodule (fake_client.py, generators.py)
  - Documentation (docs/ directory files)
  - Configuration (pyproject.toml, .pre-commit-config.yaml, .sops.yaml)
  - Scripts (check_doc_links.py, generate_dep_docs.sh, run_all_tests.bash)
  - CI/CD (.github/workflows/)
  - Project meta (README.md, CHANGELOG.md, LICENSE)

### Step 2.3: Document Full Public API

- **File**: `AGENTS.md`
- **Section**: Public API
- **Action**: Replace minimal 3-method example with categorized method reference
- **Categories**:
  - Health & Readiness (health_check, is_ready, wait_for_ready)
  - Generator Discovery (list_generators, get_generator_schema)
  - Dataset Creation (create_dataset, create_spiral_dataset)
  - Dataset Versioning (list_versions, get_latest)
  - Dataset Operations (list_datasets, get_dataset_metadata, delete_dataset)
  - Artifact Download (download_artifact_bytes, download_artifact_npz)
  - Previews (get_preview)
  - Batch Operations (batch_delete, batch_create, batch_update_tags, batch_export)
  - Resource Management (close, context manager)

### Step 2.4: Document Testing Submodule

- **File**: `AGENTS.md`
- **Action**: Add "Testing Utilities" section documenting:
  - `FakeDataClient` — drop-in mock client for consumer testing
  - `generate_spiral()`, `generate_xor()`, `generate_circle()`, `generate_moon()` — synthetic generators
  - Import path: `from juniper_data_client.testing import FakeDataClient, generate_spiral`
  - Usage in consumer projects (juniper-cascor, juniper-canopy)

---

## Phase 3: Supplementary Documentation (P2)

### Step 3.1: Add Architecture Section

- **File**: `AGENTS.md`
- **Action**: Add "Architecture & Design Patterns" section covering:
  - Connection management (requests.Session with HTTPAdapter)
  - Retry strategy (exponential backoff, retries on 429/5xx)
  - URL normalization (scheme addition, trailing slash removal, /v1 suffix handling)
  - Error mapping (HTTP status codes -> specific exception types)
  - API key handling (constructor param or JUNIPER_DATA_API_KEY env var)
  - Context manager pattern

### Step 3.2: Add CI/CD Section

- **File**: `AGENTS.md`
- **Action**: Add "CI/CD" section documenting:
  - `ci.yml` — Multi-version testing (3.12, 3.13, 3.14), pre-commit, coverage, security, quality gate
  - `publish.yml` — PyPI publishing with trusted publishing (OIDC) and build attestations
  - `security-scan.yml` — Weekly Bandit + pip-audit scanning

### Step 3.3: Document Exception Hierarchy

- **File**: `AGENTS.md`
- **Action**: Add or expand exception documentation showing:
  - Full hierarchy tree
  - HTTP status code -> exception mapping table
  - Import paths

---

## Phase 4: Metadata and Polish (P3)

### Step 4.1: Update Header Metadata

- **File**: `AGENTS.md`
- **Action**: Update "Last Updated" to 2026-04-02

### Step 4.2: Document Utility Scripts

- **File**: `AGENTS.md`
- **Action**: Add entries for scripts/ and util/ directories

### Step 4.3: Add Environment Variables Section

- **File**: `AGENTS.md`
- **Action**: Document `JUNIPER_DATA_API_KEY` and `JUNIPER_DATA_URL`

---

## Phase 5: Validation

### Step 5.1: Run Full Test Suite

- Command: `pytest tests/ -v --cov=juniper_data_client --cov-report=term-missing --cov-fail-under=80`
- Expected: All tests pass, coverage >= 80%

### Step 5.2: Verify Version Consistency

- `python -c "import juniper_data_client; print(juniper_data_client.__version__)"` -> 0.3.2
- `python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"` -> 0.3.2

### Step 5.3: Cross-Reference AGENTS.md Completeness

- Every directory in the tree is represented
- Every public method in client.py is documented
- All tool configurations match actual config files
- All CI/CD workflows are mentioned

---

## Phase 6: Commit, Push, and PR

### Step 6.1: Commit Changes

- Stage: `AGENTS.md`, `juniper_data_client/__init__.py`, `notes/` deliverables
- Message: `docs: comprehensive AGENTS.md audit and update to reflect current codebase state`

### Step 6.2: Push and Create PR

- Push branch: `chore/agents-md-audit`
- Create PR with analysis summary and change list
