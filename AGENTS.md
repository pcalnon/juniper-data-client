# juniper-data-client - Agent Development Guide

**Project**: juniper-data-client — HTTP Client for JuniperData
**Version**: 0.3.2
**License**: MIT License
**Author**: Paul Calnon
**Last Updated**: 2026-04-02

---

## Quick Reference

### Essential Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run all tests via script
bash util/run_all_tests.bash

# Run unit tests only
pytest tests/ -m unit -v

# Run with coverage
pytest tests/ --cov=juniper_data_client --cov-report=term-missing --cov-fail-under=80

# Type checking (strict mode)
mypy juniper_data_client --strict

# Linting
flake8 juniper_data_client --max-line-length=512
black --check --diff juniper_data_client
isort --check-only --diff juniper_data_client

# Validate documentation links
python scripts/check_doc_links.py

# Generate dependency docs
bash scripts/generate_dep_docs.sh
```

---

## Project Overview

`juniper-data-client` is the official Python client library for the JuniperData dataset generation service. It is a shared dependency used by both **JuniperCascor** (neural network backend) and **JuniperCanopy** (web dashboard).

### Consumers

- **JuniperCascor**: `SpiralDataProvider` uses this client for dataset retrieval
- **JuniperCanopy**: `DemoMode` and `CascorIntegration` use this client

### Data Contract

NPZ artifacts with keys: `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`)

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `JUNIPER_DATA_API_KEY` | API key for authentication (sent as `X-API-Key` header) | None (optional) |
| `JUNIPER_DATA_URL` | Service URL (used by consuming applications) | `http://localhost:8100` |

---

## Directory Structure

```bash
juniper-data-client/
├── juniper_data_client/           # Main Python package
│   ├── __init__.py                # Public API exports, __version__
│   ├── client.py                  # JuniperDataClient class (all API methods)
│   ├── exceptions.py              # Exception hierarchy
│   ├── py.typed                   # PEP 561 type hint marker
│   └── testing/                   # Testing utilities submodule (ships with package)
│       ├── __init__.py            # Exports FakeDataClient + generators
│       ├── fake_client.py         # Drop-in mock client for consumer testing
│       └── generators.py          # Synthetic dataset generators (spiral, xor, circle, moon)
├── tests/                         # Test suite (pytest)
│   ├── conftest.py                # Shared fixtures (FakeDataClient)
│   ├── test_client.py             # JuniperDataClient unit tests (HTTP mocking)
│   ├── test_fake_client.py        # FakeDataClient tests
│   ├── test_fake_client_batch.py  # Batch operation tests
│   ├── test_performance.py        # Performance benchmarks
│   └── test_versioning.py         # Dataset versioning tests
├── docs/                          # User documentation
│   ├── DOCUMENTATION_OVERVIEW.md  # Navigation index
│   ├── QUICK_START.md             # 5-minute getting started guide
│   ├── REFERENCE.md               # Complete API reference
│   └── DEVELOPER_CHEATSHEET.md    # Developer quick-reference card
├── notes/                         # Developer notes and procedures
│   ├── history/                   # Archived procedures
│   └── pull_requests/             # PR tracking notes
├── scripts/                       # Utility scripts
│   ├── check_doc_links.py         # Documentation link validator
│   └── generate_dep_docs.sh       # Dependency docs generator
├── util/                          # Shell utilities
│   └── run_all_tests.bash         # Full test runner script
├── .github/                       # GitHub configuration
│   ├── workflows/ci.yml           # CI pipeline (multi-version tests, security, quality gate)
│   ├── workflows/publish.yml      # PyPI publishing (trusted publishing + attestations)
│   ├── workflows/security-scan.yml# Weekly security scanning (Bandit + pip-audit)
│   ├── CODEOWNERS                 # Code ownership routing
│   └── dependabot.yml             # Automated dependency updates
├── AGENTS.md                      # This file
├── CLAUDE.md -> AGENTS.md         # Symlink for Claude Code
├── CHANGELOG.md                   # Version history
├── README.md                      # PyPI landing page / project overview
├── pyproject.toml                 # Package metadata, dependencies, tool config
├── .pre-commit-config.yaml        # Pre-commit hooks (20+ hooks)
├── .sops.yaml                     # SOPS encryption config for secrets
├── .env.example                   # Environment variables template
└── LICENSE                        # MIT License
```

---

## Key Files

| File | Purpose |
|------|---------|
| `juniper_data_client/client.py` | `JuniperDataClient` class — all HTTP API methods |
| `juniper_data_client/constants.py` | Module-level constants (endpoint paths, header names, defaults, generator parameter defaults) |
| `juniper_data_client/exceptions.py` | Exception hierarchy (5 specific exception types) |
| `juniper_data_client/__init__.py` | Public API exports and `__version__` |
| `juniper_data_client/py.typed` | PEP 561 marker enabling type checking for consumers |
| `juniper_data_client/testing/fake_client.py` | `FakeDataClient` — drop-in mock for consumer tests |
| `juniper_data_client/testing/generators.py` | Synthetic dataset generators (spiral, xor, circle, moon) |
| `tests/` | Test suite — unit, integration, performance, versioning |
| `docs/REFERENCE.md` | Complete API reference documentation |
| `docs/QUICK_START.md` | Getting started guide |
| `pyproject.toml` | Package config, dependencies, tool settings |
| `.pre-commit-config.yaml` | Pre-commit hooks configuration |
| `.github/workflows/ci.yml` | CI pipeline (Python 3.12/3.13/3.14, coverage, security) |
| `.github/workflows/publish.yml` | PyPI publishing with trusted publishing (OIDC) |
| `CHANGELOG.md` | Version history and release notes |
| `scripts/check_doc_links.py` | Documentation link validator |
| `util/run_all_tests.bash` | Full test runner script |

---

## Public API

### Quick Start

```python
from juniper_data_client import JuniperDataClient

client = JuniperDataClient("http://localhost:8100")
client.health_check()
client.create_spiral_dataset(n_spirals=2, n_points_per_spiral=100, noise=0.1, seed=42)
client.download_artifact_npz(dataset_id)
```

### Method Reference

#### Health & Readiness

| Method | Endpoint | Description |
|--------|----------|-------------|
| `health_check()` | GET /v1/health | Returns service health status |
| `is_ready()` | GET /v1/health/ready | Returns boolean readiness |
| `wait_for_ready(timeout, poll_interval)` | GET /v1/health/ready | Polls until service is ready |

#### Generator Discovery

| Method | Endpoint | Description |
|--------|----------|-------------|
| `list_generators()` | GET /v1/generators | Lists available dataset generators |
| `get_generator_schema(name)` | GET /v1/generators/{name}/schema | Returns parameter schema for a generator |

#### Dataset Creation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `create_dataset(generator, params, ...)` | POST /v1/datasets | Creates a dataset with any generator |
| `create_spiral_dataset(**kwargs)` | POST /v1/datasets | Convenience method for spiral datasets |

#### Dataset Versioning

| Method | Endpoint | Description |
|--------|----------|-------------|
| `list_versions(name)` | GET /v1/datasets/versions | Lists all versions of a named dataset |
| `get_latest(name)` | GET /v1/datasets/latest | Gets the latest version metadata |

#### Dataset Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `list_datasets(limit, offset)` | GET /v1/datasets | Lists dataset IDs with pagination |
| `get_dataset_metadata(dataset_id)` | GET /v1/datasets/{id} | Returns dataset metadata |
| `delete_dataset(dataset_id)` | DELETE /v1/datasets/{id} | Deletes a dataset |

#### Artifact Download

| Method | Endpoint | Description |
|--------|----------|-------------|
| `download_artifact_bytes(dataset_id)` | GET /v1/datasets/{id}/artifact | Returns raw NPZ bytes |
| `download_artifact_npz(dataset_id)` | GET /v1/datasets/{id}/artifact | Returns numpy dict with array keys |

#### Previews

| Method | Endpoint | Description |
|--------|----------|-------------|
| `get_preview(dataset_id, n)` | GET /v1/datasets/{id}/preview | Returns JSON preview of first n rows |

#### Batch Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `batch_delete(dataset_ids)` | POST /v1/datasets/batch-delete | Deletes multiple datasets |
| `batch_create(datasets)` | POST /v1/datasets/batch-create | Creates multiple datasets |
| `batch_update_tags(dataset_ids, add_tags, remove_tags)` | PATCH /v1/datasets/batch-tags | Updates tags on multiple datasets |
| `batch_export(dataset_ids)` | POST /v1/datasets/batch-export | Exports multiple datasets as ZIP |

#### Resource Management

| Pattern | Description |
|---------|-------------|
| `client.close()` | Closes the HTTP session |
| `with JuniperDataClient(...) as client:` | Context manager (auto-closes) |

---

## Exception Hierarchy

```bash
JuniperDataClientError (base)
├── JuniperDataConnectionError   — Connection failures
├── JuniperDataTimeoutError      — Request timeouts
├── JuniperDataNotFoundError     — HTTP 404
├── JuniperDataValidationError   — HTTP 400/422
└── JuniperDataConfigurationError — Invalid client configuration
```

| HTTP Status | Exception |
|-------------|-----------|
| 400, 422 | `JuniperDataValidationError` |
| 404 | `JuniperDataNotFoundError` |
| Connection failure | `JuniperDataConnectionError` |
| Timeout | `JuniperDataTimeoutError` |

---

## Testing Utilities

The `juniper_data_client.testing` submodule ships with the package and provides tools for consumer projects to test without a live juniper-data service.

### FakeDataClient

Drop-in replacement for `JuniperDataClient` that stores datasets in memory using synthetic generators. Implements the same public API — no network calls required.

```python
from juniper_data_client.testing import FakeDataClient

client = FakeDataClient()
result = client.create_spiral_dataset(n_spirals=2, n_points_per_spiral=100)
data = client.download_artifact_npz(result["dataset_id"])
```

### Synthetic Generators

```python
from juniper_data_client.testing import generate_spiral, generate_xor, generate_circle, generate_moon
```

| Generator | Description | Output |
|-----------|-------------|--------|
| `generate_spiral(n_spirals, n_points_per_spiral, noise, seed)` | Archimedean spiral classification | Dict with X_train, y_train, etc. |
| `generate_xor(n_points, noise, seed)` | XOR classification | Dict with X_train, y_train, etc. |
| `generate_circle(n_points, noise, factor, seed)` | Concentric circles | Dict with X_train, y_train, etc. |
| `generate_moon(n_points, noise, seed)` | Two half-moons | Dict with X_train, y_train, etc. |

All generators return `Dict[str, np.ndarray]` with keys `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`).

---

## Architecture & Design Patterns

### Connection Management

- Uses `requests.Session` with `HTTPAdapter` for connection pooling
- Max connections: 10, max pool size: 10
- Automatic retry via `urllib3.util.Retry` on status codes 429, 500, 502, 503, 504
- Configurable retry count (default: 3) and exponential backoff factor (default: 0.5)

### URL Normalization

- Auto-adds `http://` scheme if missing
- Strips trailing slashes
- Removes `/v1` suffix from base URL (client adds `/v1/` to all endpoint paths)

### API Key Handling

- Accepts `api_key` constructor parameter or reads `JUNIPER_DATA_API_KEY` environment variable
- Sent as `X-API-Key` header on all requests when configured

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | str | required | JuniperData service URL |
| `timeout` | int | 30 | Request timeout in seconds |
| `retries` | int | 3 | Max retry attempts |
| `backoff_factor` | float | 0.5 | Exponential backoff multiplier |
| `api_key` | str | None | API key (or use env var) |

---

## Constants

All numeric, string, and structural defaults used by the client and its testing utilities are centralized in `juniper_data_client/constants.py`. Application code (`client.py`, `testing/fake_client.py`, `testing/generators.py`) imports from this module rather than embedding inline literals.

### Categories

| Prefix / Group | Examples | Purpose |
|----------------|----------|---------|
| `API_KEY_*`, `API_VERSION_*` | `API_KEY_HEADER_NAME='X-API-Key'`, `API_KEY_ENV_VAR='JUNIPER_DATA_API_KEY'`, `API_VERSION_PATH_SUFFIX='/v1'` | Wire-protocol identifiers shared with the `juniper-data` server |
| `ENDPOINT_*` | `ENDPOINT_DATASETS='/v1/datasets'`, `ENDPOINT_HEALTH='/v1/health'`, `ENDPOINT_DATASET_BY_ID_TEMPLATE` | Full HTTP paths for every server endpoint the client calls (incl. f-string templates) |
| `DEFAULT_*` | `DEFAULT_TIMEOUT_SECONDS=30`, `DEFAULT_RETRIES=3`, `DEFAULT_BACKOFF_FACTOR=0.5` | Constructor defaults for `JuniperDataClient` |
| `RETRY_*` | `RETRY_STATUS_CODES_DEFAULT`, `RETRY_TOTAL_DEFAULT` | Retry/backoff tuning |
| Generator parameter defaults | `SPIRAL_*`, `XOR_*`, `CIRCLES_*`, `GAUSSIAN_*`, `CHECKERBOARD_*` | Default values for the synthetic dataset generators in `testing/generators.py` |

### Alignment with `juniper-data`

`API_KEY_HEADER_NAME` and `API_VERSION_PATH_SUFFIX` are bit-identical to the corresponding values on the server side (`juniper_data.api.constants.HEADER_X_API_KEY` and the `/v1` router prefix). All `ENDPOINT_*` paths equal `<server router prefix> + <relative route>`.

### Modifying

When adding a new HTTP endpoint or constructor parameter:

1. Add the constant to `constants.py` first (with a docstring noting any cross-repo coupling)
2. Reference it from `client.py` (or `fake_client.py` / `generators.py`)
3. Never embed the literal value inline in application code

---

## CI/CD

### GitHub Actions Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | Push/PR to main | Pre-commit, tests (Python 3.12/3.13/3.14 matrix), coverage (80% min), doc link validation, security scanning (Gitleaks, Bandit, pip-audit), build verification, quality gate |
| `publish.yml` | GitHub Release | Publishes to TestPyPI (with install verification) then PyPI; trusted publishing (OIDC); build attestations |
| `security-scan.yml` | Weekly schedule | Bandit code scanning + pip-audit dependency vulnerability check |

### Pre-Commit Hooks

20+ hooks enforcing: Black formatting (line-length=512), isort import sorting, Flake8 linting (strict for source, relaxed for tests), MyPy type checking, Bandit security scanning, markdownlint, shellcheck, yamllint, SOPS `.env` file blocking.

### Tool Configuration (pyproject.toml)

| Tool | Key Setting |
|------|-------------|
| Black | line-length=512, target py312/py313 |
| isort | profile=black, line-length=512 |
| MyPy | strict=true, python_version=3.12 |
| Coverage | fail_under=80, branch=true |
| Pytest | timeout=30s, markers: unit, integration, performance |

---

## Worktree Procedures (Mandatory — Task Isolation)

> **OPERATING INSTRUCTION**: All feature, bugfix, and task work SHOULD use git worktrees for isolation. Worktrees keep the main working directory on the default branch while task work proceeds in a separate checkout.

### What This Is

Git worktrees allow multiple branches of a repository to be checked out simultaneously in separate directories. For the Juniper ecosystem, all worktrees are centralized in **`/home/pcalnon/Development/python/Juniper/worktrees/`** using a standardized naming convention.

The full setup and cleanup procedures are defined in:

- **`notes/WORKTREE_SETUP_PROCEDURE.md`** — Creating a worktree for a new task
- **`notes/WORKTREE_CLEANUP_PROCEDURE_V2.md`** — Merging, removing, and pushing after task completion (V2 — fixes CWD-trap bug)

Read the appropriate file when starting or completing a task.

### Worktree Directory Naming

Format: `<repo-name>--<branch-name>--<YYYYMMDD-HHMM>--<short-hash>`

Example: `juniper-data-client--feature--add-retry--20260225-1430--73294fc1`

- Slashes in branch names are replaced with `--`
- All worktrees reside in `/home/pcalnon/Development/python/Juniper/worktrees/`

### When to Use Worktrees

| Scenario | Use Worktree? |
| -------- | ------------- |
| Feature development (new feature branch) | **Yes** |
| Bug fix requiring a dedicated branch | **Yes** |
| Quick single-file documentation fix on main | No |
| Exploratory work that may be discarded | **Yes** |
| Hotfix requiring immediate merge | **Yes** |

### Quick Reference

**Setup** (full procedure in `notes/WORKTREE_SETUP_PROCEDURE.md`):

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-data-client
git fetch origin && git checkout main && git pull origin main
BRANCH_NAME="feature/my-task"
git branch "$BRANCH_NAME" main
REPO_NAME=$(basename "$(pwd)")
SAFE_BRANCH=$(echo "$BRANCH_NAME" | sed 's|/|--|g')
WORKTREE_DIR="/home/pcalnon/Development/python/Juniper/worktrees/${REPO_NAME}--${SAFE_BRANCH}--$(date +%Y%m%d-%H%M)--$(git rev-parse --short=8 HEAD)"
git worktree add "$WORKTREE_DIR" "$BRANCH_NAME"
cd "$WORKTREE_DIR"
```

**Cleanup** (full procedure in `notes/WORKTREE_CLEANUP_PROCEDURE_V2.md`):

```bash
# Phase 1: Push current work
cd "$OLD_WORKTREE_DIR" && git push origin "$OLD_BRANCH"
# Phase 2: Create new worktree BEFORE removing old (prevents CWD-trap)
git fetch origin
git worktree add "$NEW_WORKTREE_DIR" -b "$NEW_BRANCH" origin/main
cd "$NEW_WORKTREE_DIR"
# Phase 3: Create PR (do NOT merge directly to main)
gh pr create --base main --head "$OLD_BRANCH" --title "<title>" --body "<body>"
# Phase 4: Cleanup
git worktree remove "$OLD_WORKTREE_DIR"
git branch -d "$OLD_BRANCH"
git worktree prune
```

### Rules

- **Centralized location**: All worktrees go in `/home/pcalnon/Development/python/Juniper/worktrees/`. Never create worktrees inside the repo directory.
- **Clean before you start**: Ensure the main working directory is clean before creating a worktree.
- **Push before you merge**: Always push the working branch to remote before merging (backup).
- **Prune after cleanup**: Run `git worktree prune` after removing a worktree to clean metadata.
- **Do not leave stale worktrees**: Clean up worktrees promptly after merging.

---

## Thread Handoff (Mandatory — Replaces Thread Compaction)

> **CRITICAL OPERATING INSTRUCTION**: Thread handoff MUST be performed instead of thread compaction. This section defines mandatory behavior that overrides Claude Code's default compaction process.

### What This Is

When a long-running conversation approaches context limits, Claude Code normally performs **thread compaction** — summarizing prior context to free token capacity. This introduces information loss. Instead, Claude Code instances working on this project MUST perform a **proactive thread handoff**: transferring a curated, high-signal summary to a fresh thread with full context capacity.

The full handoff protocol is defined in **`notes/THREAD_HANDOFF_PROCEDURE.md`**. Read that file when a handoff is triggered.

### When to Trigger a Handoff

**Automatic trigger (pre-compaction threshold):** Initiate a thread handoff when token utilization reaches **95% to 99%** of the level at which thread compaction would normally be triggered. This means the handoff fires when you are within **1% to 5%** of the compaction threshold, ensuring the handoff completes before compaction would occur.

Concretely:

- If compaction would trigger at N% context utilization, begin handoff at (N − 5)% to (N − 1)%.
- **Self-assessment rule**: At each turn where you are performing multi-step work, assess whether you are approaching the compaction threshold. If you estimate you are within 5% of it, begin the handoff protocol immediately.
- When the system compresses prior messages or you receive a context compression notification, treat this as a signal that handoff should have already occurred — immediately initiate one.

**Additional triggers** (from `notes/THREAD_HANDOFF_PROCEDURE.md`):

| Condition                   | Indicator                                                            |
| --------------------------- | -------------------------------------------------------------------- |
| **Context saturation**      | Thread has performed 15+ tool calls or edited 5+ files               |
| **Phase boundary**          | A logical phase of work is complete                                  |
| **Degraded recall**         | Re-reading a file already read, or re-asking a resolved question     |
| **Multi-module transition** | Moving between major components                                      |
| **User request**            | User says "hand off", "new thread", or similar                       |

**Do NOT handoff** when:

- The task is nearly complete (< 2 remaining steps)
- The current thread is still sharp and producing correct output
- The work is tightly coupled and splitting would lose critical in-flight state

### How to Execute a Handoff

1. **Checkpoint**: Inventory what was done, what remains, what was discovered, and what files are in play
2. **Compose the handoff goal**: Write a concise, actionable summary (see templates in `notes/THREAD_HANDOFF_PROCEDURE.md`)
3. **Present to user**: Output the handoff goal to the user and recommend starting a new thread with that goal as the initial prompt
4. **Include verification commands**: Always specify how the new thread should verify its starting state (test commands, file checks)
5. **State git status**: Mention branch, staged files, and any uncommitted work

### Rules

- **This is not optional.** Every Claude Code instance on this project must follow these rules.
- **Handoff early, not late.** A handoff at 70% context usage is better than compaction at 95%.
- **Do not duplicate CLAUDE.md content** in the handoff goal — the new thread reads CLAUDE.md automatically.
- **Be specific** in the handoff goal: include file paths, decisions made, and test status.
