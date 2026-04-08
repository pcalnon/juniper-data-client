# Reference

## juniper-data-client Technical Reference

**Version:** 0.4.0
**Status:** Active
**Last Updated:** April 8, 2026
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
| `base_url` | `str` | `"http://localhost:8100"` | JuniperData service URL |
| `timeout` | `int` | `30` | Request timeout in seconds |
| `retries` | `int` | `3` | Number of retry attempts for failed requests |
| `backoff_factor` | `float` | `0.5` | Backoff multiplier between retries |
| `api_key` | `Optional[str]` | `None` | API key; falls back to `JUNIPER_DATA_API_KEY` env var |

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
| `create_dataset(generator, params, persist=True, name=None, description=None, created_by=None, parent_dataset_id=None)` | `Dict` | Create dataset; returns `dataset_id`, `generator`, `meta`, `artifact_url` |
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
├── JuniperDataConnectionError    # Connection to service failed
├── JuniperDataTimeoutError       # Request timed out
├── JuniperDataNotFoundError      # 404 - Resource not found
├── JuniperDataValidationError    # 400/422 - Invalid parameters
└── JuniperDataConfigurationError # Missing or invalid config
```

### Import

```python
from juniper_data_client import (
    JuniperDataClientError,
    JuniperDataConfigurationError,
    JuniperDataConnectionError,
    JuniperDataNotFoundError,
    JuniperDataTimeoutError,
    JuniperDataValidationError,
)
```

### HTTP Status Code Mapping

| Status Code | Exception Raised |
|-------------|-----------------|
| 400 | `JuniperDataValidationError` |
| 404 | `JuniperDataNotFoundError` |
| 422 | `JuniperDataValidationError` |
| Connection failure | `JuniperDataConnectionError` |
| Timeout | `JuniperDataTimeoutError` |
| Other 4xx/5xx | `JuniperDataClientError` |

---

## Testing Utilities

### FakeDataClient

Drop-in replacement for `JuniperDataClient` that generates synthetic datasets in-memory. No HTTP calls are made.

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

The client automatically normalizes the `base_url`:
- Adds `http://` scheme if missing
- Strips trailing slashes
- Strips trailing `/v1` suffix

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

---

## HTTP Behavior

### Retry Strategy

- **Retried status codes:** 429, 500, 502, 503, 504
- **Retried methods:** HEAD, GET, POST, DELETE
- **Backoff:** Exponential with configurable factor (default 0.5s)
- **Connection pooling:** 10 connections, 10 max pool size

### Authentication

If `api_key` is provided (or `JUNIPER_DATA_API_KEY` is set), the client sends an `X-API-Key` header with every request.

### API Prefix

All requests target `/v1/` endpoints on the configured `base_url`.

---

## Environment Variables

| Variable | Purpose | Used By |
|----------|---------|---------|
| `JUNIPER_DATA_API_KEY` | API key for authentication (fallback if not passed to constructor) | `JuniperDataClient.__init__` |
| `JUNIPER_DATA_URL` | Service URL used by consuming applications (not read by client directly) | juniper-cascor, juniper-canopy |

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
| `tests/test_client.py` | Unit tests for `JuniperDataClient` |
| `tests/test_fake_client.py` | Tests for `FakeDataClient` testing utility |
| `tests/conftest.py` | Shared fixtures |

### Quality Checks

```bash
mypy juniper_data_client --strict    # Type checking
flake8 juniper_data_client           # Linting
black --check juniper_data_client    # Format check
isort --check-only juniper_data_client  # Import order
```

---

**Last Updated:** April 8, 2026
**Version:** 0.4.0
**Maintainer:** Paul Calnon
