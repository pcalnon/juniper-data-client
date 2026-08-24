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
- [NPZ Artifact Schema](#npz-artifact-schema)
- [`validate_npz_contract`](#validate_npz_contract)
- [HTTP Behavior](#http-behavior)
- [Environment Variables](#environment-variables)
- [Test Markers and Commands](#test-markers-and-commands)

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
