# Reference

## juniper-data-client Technical Reference

**Version:** 0.4.2
**Status:** Active
**Last Updated:** August 24, 2026
**Project:** Juniper - Dataset Service Client Library

---

## Table of Contents

- [Client API](#client-api)
- [Constructor Parameters](#constructor-parameters)
- [Methods Reference](#methods-reference)
- [Convenience Methods](#convenience-methods)
- [Batch Operations](#batch-operations)
- [Versioning Methods](#versioning-methods)
- [Exception Hierarchy](#exception-hierarchy)
- [Testing Utilities](#testing-utilities)
- [Configuration Reference](#configuration-reference)
- [Public API Reference](#public-api-reference)
- [Directory Structure Reference](#directory-structure-reference)
- [Key Files Reference](#key-files-reference)
- [Architecture and Design Patterns Reference](#architecture-and-design-patterns-reference)
- [Constants Reference](#constants-reference)
- [CI/CD Reference](#cicd-reference)
- [NPZ Artifact Schema](#npz-artifact-schema)
- [`validate_npz_contract`](#validate_npz_contract)
- [HTTP Behavior](#http-behavior)
- [Environment Variables](#environment-variables)
- [Test Markers and Commands](#test-markers-and-commands)

---

## Project Overview Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

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

> **This table is the relocated summary, not the canonical list.** [§ Environment Variables](#environment-variables) below is authoritative: it also documents `JUNIPER_DATA_API_KEY_FILE` (Docker-secret indirection) and the precedence between the three. Prefer it, and update it rather than this table.

---

## Client API

### Import

```python
from juniper_data_client import JuniperDataClient
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `"http://localhost:8100"` | JuniperData service URL. Normalized at construction (see [URL Normalization](#url-normalization)). Hostless values raise `JuniperDataConfigurationError` immediately. |
| `timeout` | `int` | `30` | Request timeout in seconds |
| `retries` | `int` | `3` | Number of retry attempts for failed requests (idempotent methods only; see [Retry Strategy](#retry-strategy)) |
| `backoff_factor` | `float` | `0.5` | Backoff multiplier between retries |
| `api_key` | `Optional[str]` | `None` | API key. Wins over both env vars. If omitted, the client reads `JUNIPER_DATA_API_KEY_FILE` then `JUNIPER_DATA_API_KEY`. |
| `on_request` | `Optional[RequestHook]` | no-op | Instrumentation hook `(method, url, status, duration_ms, error)` fired once per HTTP call. Hook exceptions are logged at WARNING and never crash the request. |

### Context Manager

```python
with JuniperDataClient("http://localhost:8100") as client:
    # Use client
    pass
# Session automatically closed
```

---

## Methods Reference

### Health and Readiness

| Method | Returns | Description |
|--------|---------|-------------|
| `health_check()` | `Dict[str, Any]` | Service health status (`{"status": "ok", "version": "..."}`) |
| `is_ready()` | `bool` | `True` if service is ready, `False` otherwise |
| `wait_for_ready(timeout=30.0, poll_interval=0.5)` | `bool` | Block until service ready or timeout |

### Generator Discovery

| Method | Returns | Description |
|--------|---------|-------------|
| `list_generators()` | `List[Dict]` | All available generators with descriptions |
| `get_generator_schema(name)` | `Dict` | JSON schema for a generator's parameters |

### Dataset Operations

| Method | Returns | Description |
|--------|---------|-------------|
| `create_dataset(generator, params, persist=True, name=None, description=None, created_by=None, parent_dataset_id=None, tags=None, ttl_seconds=None)` | `Dict` | Create dataset; returns `dataset_id`, `generator`, `meta`, `artifact_url` |
| `list_datasets(limit=100, offset=0)` | `List[str]` | List dataset ID strings with pagination |
| `get_dataset_metadata(dataset_id)` | `Dict` | Metadata for a specific dataset |
| `delete_dataset(dataset_id)` | `bool` | Delete a dataset; returns `True` on success |

#### `create_dataset` Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `Optional[str]` | `None` | Dataset name for versioning. When provided, the service automatically assigns an incrementing version number. |
| `description` | `Optional[str]` | `None` | Human-readable description of the dataset. |
| `created_by` | `Optional[str]` | `None` | Identifier for the creator (user or system). |
| `parent_dataset_id` | `Optional[str]` | `None` | ID of the parent dataset this was derived from. |
| `tags` | `Optional[List[str]]` | `None` | Tags forwarded to the server's `CreateDatasetRequest.tags`. Searchable via `list_datasets` filters. |
| `ttl_seconds` | `Optional[int]` | `None` | Time-to-live in seconds. Must be `>= 1` when provided; the fake client raises `JuniperDataValidationError` with `status_code=422` for a non-positive value, matching FastAPI/pydantic. |

### Artifact Download

| Method | Returns | Description |
|--------|---------|-------------|
| `download_artifact_npz(dataset_id)` | `Dict[str, ndarray]` | Download and parse NPZ artifact into numpy arrays |
| `download_artifact_bytes(dataset_id)` | `bytes` | Download raw NPZ file bytes |
| `get_preview(dataset_id, n=100)` | `Dict` | JSON preview of first `n` samples (max 1000) |

### Session Management

| Method | Returns | Description |
|--------|---------|-------------|
| `close()` | `None` | Close the HTTP session and release resources |

---

## Convenience Methods

### `create_spiral_dataset(**kwargs)`

Convenience wrapper for creating spiral datasets without building the params dict manually.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_spirals` | `int` | `2` | Number of spiral arms |
| `n_points_per_spiral` | `int` | `100` | Points per spiral arm |
| `noise` | `float` | `0.1` | Noise level |
| `seed` | `Optional[int]` | `None` | Random seed for reproducibility |
| `algorithm` | `str` | `"modern"` | `"modern"` or `"legacy_cascor"` |
| `train_ratio` | `float` | `0.8` | Fraction of data for training split |
| `**kwargs` | `Any` | -- | Additional parameters passed to generator |

**Returns:** `Dict[str, Any]` -- Dataset creation response with `dataset_id` and metadata.

---

## Batch Operations

Methods for operating on multiple datasets in a single request.

### `batch_delete(dataset_ids: List[str])`

Delete multiple datasets in a single request.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_ids` | `List[str]` | List of dataset IDs to delete (1-100) |

**Returns:** `Dict[str, Any]` -- Dictionary with `deleted`, `not_found`, and `total_deleted`.

### `batch_create(datasets: List[Dict[str, Any]])`

Create multiple datasets in a single request. Each item should have `generator` and `params` keys, and optionally `persist`, `tags`, and `ttl_seconds`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `datasets` | `List[Dict[str, Any]]` | List of dataset specifications (1-50) |

**Returns:** `Dict[str, Any]` -- Dictionary with `results`, `total_created`, and `total_failed`.

### `batch_update_tags(dataset_ids: List[str], add_tags: Optional[List[str]] = None, remove_tags: Optional[List[str]] = None)`

Add or remove tags from multiple datasets. Uses PATCH method.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_ids` | `List[str]` | -- | List of dataset IDs to update (1-100) |
| `add_tags` | `Optional[List[str]]` | `None` | Tags to add to all specified datasets |
| `remove_tags` | `Optional[List[str]]` | `None` | Tags to remove from all specified datasets |

**Returns:** `Dict[str, Any]` -- Dictionary with `updated`, `not_found`, and `total_updated`.

### `batch_export(dataset_ids: List[str])`

Export multiple datasets as a ZIP archive of NPZ files.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_ids` | `List[str]` | List of dataset IDs to export (1-50) |

**Returns:** `bytes` -- Raw bytes of the ZIP archive.

**Raises:** `JuniperDataNotFoundError` if none of the datasets exist.

---

## Versioning Methods

Methods for working with named dataset versions. When a dataset is created with a `name` parameter, the service automatically assigns an incrementing version number.

### `list_versions(name: str)`

List all versions of a named dataset.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Dataset name to list versions for |

**Returns:** `Dict[str, Any]` -- Dictionary with `dataset_name`, `versions` list, `total` count, and `latest_version`.

### `get_latest(name: str)`

Get the latest version of a named dataset.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Dataset name to get latest version of |

**Returns:** `Dict[str, Any]` -- Dataset metadata for the latest version.

**Raises:** `JuniperDataNotFoundError` if no versions exist for the given name.

---

## Exception Hierarchy

```
JuniperDataClientError (base)
├── JuniperDataConnectionError     # Connection to service failed
├── JuniperDataTimeoutError        # Request timed out
├── JuniperDataNotFoundError       # 404 - Resource not found
├── JuniperDataValidationError     # 400/422 - Invalid parameters
├── JuniperDataConfigurationError  # Missing or invalid config
└── JuniperDataContractError       # NPZ contract violation (also a ValueError)
```

### Import

```python
from juniper_data_client import (
    JuniperDataClientError,
    JuniperDataConfigurationError,
    JuniperDataConnectionError,
    JuniperDataContractError,
    JuniperDataNotFoundError,
    JuniperDataTimeoutError,
    JuniperDataValidationError,
)
```

### Exception context

Every exception in the hierarchy carries four attributes set by the base `__init__`. The extra parameters are **keyword-only**, so `raise JuniperDataNotFoundError("missing")` still works.

| Attribute | Meaning |
|-----------|---------|
| `message` | Human-readable summary; also what `str(exc)` returns. |
| `status_code` | HTTP status of the originating response, or `None` when the error was raised without one (configuration, connection, timeout, retry-exhausted, contract). |
| `detail` | The server's `detail` payload **exactly as decoded** — a `str` for most handlers, a `list[dict]` for FastAPI's 422. Never stringified. |
| `response` | The originating `requests.Response`, when there was one. |

`status_code` is the **only** thing separating a 400 from a 422 — both raise `JuniperDataValidationError`.

A FastAPI 422 `detail` list stays on `exc.detail` unmodified. The message renders it as `body.seed: Field required` via `_render_error_detail`. Interpolating the list into the message produced an unparseable Python repr.

Locally raised errors (`JuniperDataConfigurationError`, connection, timeout, `JuniperDataContractError`) leave `status_code` / `detail` / `response` as `None`.

`FakeDataClient` populates `status_code` on every error it raises (400 unknown generator, 422 `ttl_seconds` violation, 404 missing resource). A double that raised the right type with `status_code=None` would let a consumer test pass against behaviour production does not have.

`JuniperDataContractError` subclasses **both** `JuniperDataClientError` and `ValueError`, so `except ValueError` still catches `validate_npz_contract` failures (the original documented contract).

### HTTP Status Code Mapping

| Status Code | Exception Raised |
|-------------|-----------------|
| 400 | `JuniperDataValidationError` |
| 404 | `JuniperDataNotFoundError` |
| 422 | `JuniperDataValidationError` |
| Connection failure | `JuniperDataConnectionError` |
| Timeout | `JuniperDataTimeoutError` |
| Other 4xx/5xx | `JuniperDataClientError` |
| Hostless `base_url` | `JuniperDataConfigurationError` (`status_code=None`) |
| NPZ contract violation | `JuniperDataContractError` (`status_code=None`) |

---

## Testing Utilities

### FakeDataClient

Drop-in replacement for `JuniperDataClient` that generates synthetic datasets in-memory. No HTTP calls are made. `base_url` is stored as-is and is **not** run through `_normalize_url`, so a hostless fake URL does not raise. Errors it raises do populate `status_code` (400 unknown generator, 422 non-positive `ttl_seconds`, 404 missing resource).

```python
from juniper_data_client.testing import FakeDataClient

with FakeDataClient() as client:
    result = client.create_spiral_dataset(n_spirals=2, seed=42)
    arrays = client.download_artifact_npz(result["dataset_id"])
    X_train = arrays["X_train"]
```

### Synthetic Generators

Available via `juniper_data_client.testing`:

| Function | Description |
|----------|-------------|
| `generate_spiral(...)` | Synthetic spiral dataset |
| `generate_xor(...)` | XOR classification dataset |
| `generate_circle(...)` | Concentric circles dataset |
| `generate_moon(...)` | Half-moon classification dataset |

---

## Testing Utilities Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

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

---

## Configuration Reference

### Constructor Defaults (Class Constants)

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_TIMEOUT` | `30` | Request timeout in seconds |
| `DEFAULT_RETRIES` | `3` | Retry attempts for transient failures |
| `DEFAULT_BACKOFF_FACTOR` | `0.5` | Exponential backoff multiplier |

### URL Normalization

`JuniperDataClient.__init__` calls `_normalize_url` before any request (`APD-DCLIENT-004`, hardened in `#166`):

1. `url.strip()` — surrounding whitespace is ignored.
2. If the stripped URL does not **case-insensitively** start with `http://` or `https://`, prefix `http://`. Scheme matching follows RFC 3986 §3.1: a case-sensitive `startswith` would re-prefix `HTTPS://host` into `http://HTTPS://host` — a silent TLS downgrade that sends the API key over HTTP to hostname `https`.
3. `urlparse`; empty **`hostname`** (not `netloc`) raises `JuniperDataConfigurationError(f"base_url must include a host; got {url!r}")` with `status_code=None`. `netloc` is truthy for a userinfo-only `http://user:secret@` while `hostname` is `None`.
4. Rebuild `f"{parsed.scheme}://{parsed.netloc}{parsed.path}"` (scheme stored lowercase by `urlparse`) then `rstrip("/")`.
5. Strip a trailing `/v1` — the client adds `/v1/` to every endpoint path.

Valid forms that still construct: schemeless hosts (`localhost:8100`), trailing slashes, `/v1` suffixes, mixed-case `Http://`, uppercase `HTTPS://`.

Hostless shapes that fail at construction: `""`, `"   "`, `"http://"`, `"https://"`, `"/v1"`, `"http:///v1"`, `"http://user:secret@"`.

**Deliberate gap:** `FakeDataClient` stores `base_url` as-is and never contacts it. A hostless fake URL does **not** raise. Do not use the fake to pin construction-time URL guards.

### URL Normalization (examples)

```python
JuniperDataClient("localhost:8100").base_url          # http://localhost:8100
JuniperDataClient("HTTPS://api.example.com:8100").base_url  # https://api.example.com:8100
JuniperDataClient("http://localhost:8100/v1/").base_url     # http://localhost:8100
JuniperDataClient("http://")  # raises JuniperDataConfigurationError
```

---

## Public API Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

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

---

## Directory Structure Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

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

---

## Key Files Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

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

---

## Architecture and Design Patterns Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### Connection Management

- Uses `requests.Session` with `HTTPAdapter` for connection pooling
- Max connections: 10, max pool size: 10
- Automatic retry via `urllib3.util.Retry` on status codes 429, 500, 502, 503, 504
- **Retried methods:** `HEAD`, `GET`, `PUT` only (`RETRY_ALLOWED_METHODS`). POST, PATCH, and DELETE are not auto-retried (XREPO-11: duplicate creates / repeated deletes).
- Configurable retry count (default: 3) and exponential backoff factor (default: 0.5)

### URL Normalization

`JuniperDataClient._normalize_url` runs at construction (`APD-DCLIENT-004`, `#166`):

- Strip surrounding whitespace
- If the URL does not **case-insensitively** start with `http://` or `https://`, prefix `http://` (a case-sensitive check would re-prefix `HTTPS://host` into `http://HTTPS://host` — silent TLS downgrade, API key sent to hostname `https`)
- `urlparse`; empty **`hostname`** (not `netloc`) raises `JuniperDataConfigurationError` naming the value. `netloc` is truthy for userinfo-only `http://user:secret@`
- Rebuild scheme/netloc/path, strip trailing `/`, then strip trailing `/v1` (the client adds `/v1/` to every endpoint)

`FakeDataClient` stores `base_url` as-is and does not run this guard.

### API Key Handling

- Constructor `api_key=` wins over both environment variables
- Else `JUNIPER_DATA_API_KEY_FILE` (path whose stripped contents are the key, e.g. `/run/secrets/juniper_data_api_keys`); unreadable or empty file falls through
- Else `JUNIPER_DATA_API_KEY`
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

---

## Constants Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

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

---

## CI/CD Reference

Relocated verbatim from `AGENTS.md` (P3 of the shared-session-memory plan) so it is read on demand rather than loaded into every session.

### GitHub Actions Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | Push/PR to main | Pre-commit, tests (Python 3.12/3.13/3.14 matrix), coverage (80% min), doc link validation, security scanning (Gitleaks, Bandit, pip-audit), build verification, quality gate |
| `publish.yml` | GitHub Release | Publishes to TestPyPI (with install verification) then PyPI; trusted publishing (OIDC); build attestations |
| `security-scan.yml` | Weekly schedule | Bandit code scanning + pip-audit dependency vulnerability check |
| `sequence-safety.yml` | PR to main/develop | **Advisory** per-PR compositional-loss net (rollout Wave 2): AST symbol-loss screen (scope `juniper_data_client/**/*.py` + `tests/**/*.py`) + docs deletion-magnitude screen, from the published `juniper-ci-tools` console scripts. Standalone, never a required check; the `allow-symbol-loss` / `docs-rewrite` labels demote a FAIL to WARN-only |
| `main-verify.yml` | Push to main | **Advisory** post-merge, bypass-proof run of the same two screens over the catch-up base .. merge tip (per-SHA, no-cancel so every merge is verified during a storm); screens-only (no regression battery); files one stable-title tracking issue per red streak on failure |

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

### PR base-branch guard (required check)

`.github/workflows/pr-base-branch-guard.yml` fails any PR whose base branch is not the
default branch. Its job name -- **`Guard PR base branch`** -- is a **required status check**
in this repo's ruleset, so renaming the job or deleting the file makes `main` unmergeable
until the context is un-required first.

**What it protects against.** A PR based on another feature branch can squash-merge into
that branch, stranding its content off `main` behind a green **MERGED** badge. It has
happened three times in this ecosystem (`juniper-recurrence#7`/`#8`, `juniper-canopy#365`).

**Why it matters more than it looks.** Both rulesets here are scoped to `~DEFAULT_BRANCH`, so
a PR whose base is a feature branch is governed by **no ruleset at all** -- it has zero
required status checks and merges clean with nothing having run:

```bash
gh api repos/pcalnon/<repo>/rules/branches/feature%2Fanything --jq length   # -> 0
gh api repos/pcalnon/<repo>/rules/branches/main               --jq length   # -> 9
```

This workflow carries no `branches:` filter, so it is the **only** check that runs on such a
PR. It cannot block the merge there -- no ruleset applies -- but it turns a silent merge into
a visibly red one.

**If it fails.** Re-open the work against the default branch. The house practice is
**close and re-open** a fresh PR titled `[retarget #NNN]`. Retargeting in place is *not*
sufficient on its own: every `ci*.yml` here uses the default `pull_request` types
`[opened, synchronize, reopened]`, which exclude `edited`, so a retarget re-runs this guard
and nothing else -- the PR stays blocked on its other required contexts until a push or a
close/re-open.

**`stacked-pr` label.** Silences this guard for a deliberate stack. It does **not** make the
PR mergeable into `main`, and it does **not** re-land the stack -- do that separately.

Rollout and rationale: [juniper-ml#434](https://github.com/pcalnon/juniper-ml/issues/434).

---

## NPZ Artifact Schema

All arrays are `float32` dtype.

| Key | Shape | Description |
|-----|-------|-------------|
| `X_train` | `(n_train, n_features)` | Training features |
| `y_train` | `(n_train, n_classes)` | Training labels (one-hot) |
| `X_test` | `(n_test, n_features)` | Test features |
| `y_test` | `(n_test, n_classes)` | Test labels (one-hot) |
| `X_full` | `(n_total, n_features)` | Full dataset features |
| `y_full` | `(n_total, n_classes)` | Full dataset labels (one-hot) |

Default split: 80% training, 20% test (controlled by `train_ratio`).

### `validate_npz_contract`

Public helper that classifies a loaded artifact and enforces the WS-1 sequence rules.

```python
from juniper_data_client import validate_npz_contract, ContractKind

kind: ContractKind = validate_npz_contract(arrays)  # "tabular" or "sequence"
```

| Return | Meaning |
|--------|---------|
| `"tabular"` | 2-D `X` (legacy path; no further checks). |
| `"sequence"` | 3-D `X` `(W, L, F)` with irregular-Δt keys; `dt >= 0`, `dt[:, 0] == 0`, optional binary masks, consistent `t` / `dt`. |

**Raises:** `JuniperDataContractError` (also a `ValueError`) when `X` is neither 2-D nor 3-D, or any 3-D rule fails. Contract violations are detected locally after download, so `status_code` stays `None`. The return type is `ContractKind` (`Literal["tabular", "sequence"]`); `CONTRACT_KIND_TABULAR` / `CONTRACT_KIND_SEQUENCE` are `Final[ContractKind]`.

Optional `dt_atol` (default `1e-6`) is the absolute tolerance for the `t` / `dt` consistency check.

---

## HTTP Behavior

### Retry Strategy

- **Retried status codes:** 429, 500, 502, 503, 504
- **Retried methods:** `HEAD`, `GET`, `PUT` only (`RETRY_ALLOWED_METHODS`). POST, PATCH, and DELETE are **not** auto-retried — a transient 5xx after the server had already applied the mutation used to duplicate dataset creation or repeat deletes (XREPO-11). Callers that need retry for mutations must layer their own idempotency (for example a client-supplied dataset `name` so POST collapses via server-side dedupe).
- **Backoff:** Exponential with configurable factor (default 0.5s)
- **Connection pooling:** 10 connections, 10 max pool size

### Authentication

If `api_key` is provided, it is sent as `X-API-Key` on every request. Otherwise the client reads `JUNIPER_DATA_API_KEY_FILE` (a path whose stripped contents are the key, e.g. `/run/secrets/juniper_data_api_keys`) and then `JUNIPER_DATA_API_KEY`. An unreadable or empty `_FILE` falls through to the plain env var.

### Request correlation

When `juniper-observability` is installed and the calling thread has a non-empty `request_id_var`, `_request()` copies it into outbound `X-Request-ID`. `ImportError` and `LookupError` silently no-op. Caller-supplied `X-Request-ID` headers always win. Standalone install: `pip install juniper-data-client[observability]`.

### API Prefix

All requests target `/v1/` endpoints on the configured `base_url`.

---

## Environment Variables

| Variable | Purpose | Used By |
|----------|---------|---------|
| `JUNIPER_DATA_API_KEY` | API key for authentication (fallback if `api_key=` and `_FILE` are unset) | `JuniperDataClient.__init__` |
| `JUNIPER_DATA_API_KEY_FILE` | Path to a file whose stripped contents are the API key (Docker-secret indirection; wins over the plain env var, loses to `api_key=`) | `_resolve_api_key_from_env` |
| `JUNIPER_DATA_URL` | Service URL used by consuming applications (not read by this client) | juniper-cascor, juniper-canopy |

---

## Test Markers and Commands

### Running Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/ -m unit -v            # Unit tests only
pytest tests/ --cov=juniper_data_client --cov-report=term-missing --cov-fail-under=80
```

### Test Files

| File | Purpose |
|------|---------|
| `tests/test_client.py` | Unit tests for `JuniperDataClient` (includes URL normalization) |
| `tests/test_fake_client.py` | Tests for `FakeDataClient` testing utility |
| `tests/test_contract.py` | `validate_npz_contract` raise sites |
| `tests/test_retry_policy.py` | `RETRY_ALLOWED_METHODS` / `RETRYABLE_STATUS_CODES` |
| `tests/conftest.py` | Shared fixtures |

### Quality Checks

```bash
mypy juniper_data_client --strict    # Type checking
flake8 juniper_data_client           # Linting
black --check juniper_data_client    # Format check
isort --check-only juniper_data_client  # Import order
```

---

**Last Updated:** August 24, 2026
**Version:** 0.4.2
**Maintainer:** Paul Calnon
