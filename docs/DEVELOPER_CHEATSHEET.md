# Developer Cheatsheet — juniper-data-client

**Version**: 1.0.1
**Date**: 2026-08-24
**Project**: juniper-data-client

---

## Common Commands

| Command                                                                                 | Description                 |
|-----------------------------------------------------------------------------------------|-----------------------------|
| `pip install -e ".[dev]"`                                                               | Install in development mode |
| `pip install juniper-data-client`                                                       | Install from PyPI           |
| `pytest tests/ -v`                                                                      | Run all tests               |
| `pytest tests/ -m unit -v`                                                              | Run unit tests only         |
| `pytest tests/ --cov=juniper_data_client --cov-report=term-missing --cov-fail-under=80` | Run with coverage           |
| `mypy juniper_data_client --strict`                                                     | Type checking (strict)      |
| `flake8 juniper_data_client --max-line-length=512` | Linting                     |
| `black --check --diff juniper_data_client`                                              | Format check                |
| `isort --check-only --diff juniper_data_client`                                         | Import order check          |

---

## Client Usage

### Initialization

```python
from juniper_data_client import JuniperDataClient

# Basic (default base_url is http://localhost:8100)
client = JuniperDataClient("http://localhost:8100")

# Hostless values fail at construction — not on the first request
# JuniperDataClient("http://")  # JuniperDataConfigurationError

# HTTPS:// is preserved (scheme matching is case-insensitive)
# client = JuniperDataClient("HTTPS://api.example.com:8100")

# With options
client = JuniperDataClient(
    base_url="http://localhost:8100",
    timeout=30,
    retries=3,
    backoff_factor=0.5,
    api_key="my-key",  # wins over JUNIPER_DATA_API_KEY_FILE and JUNIPER_DATA_API_KEY
)

# Context manager (auto-closes session)
with JuniperDataClient("http://localhost:8100") as client:
    client.health_check()
```

### Key Methods

| Method                                | Returns              | Description                          |
|---------------------------------------|----------------------|--------------------------------------|
| `health_check()`                      | `Dict`               | Service health status                |
| `is_ready()`                          | `bool`               | `True` if service is ready           |
| `wait_for_ready(timeout=30.0)`        | `bool`               | Block until ready or timeout         |
| `list_generators()`                   | `List[Dict]`         | Available generators                 |
| `create_dataset(generator, params)`   | `Dict`               | Create dataset, returns `dataset_id` |
| `create_spiral_dataset(**kwargs)`     | `Dict`               | Convenience spiral creator           |
| `download_artifact_npz(dataset_id)`   | `Dict[str, ndarray]` | Download parsed NPZ arrays           |
| `download_artifact_bytes(dataset_id)` | `bytes`              | Download raw NPZ bytes               |
| `list_datasets(limit=100)`            | `List[str]`          | List dataset IDs                     |
| `delete_dataset(dataset_id)`          | `bool`               | Delete a dataset                     |

> See: [docs/REFERENCE.md](REFERENCE.md) for full method signatures and parameters.

---

## Data Contract (NPZ Format)

All arrays are `float32` dtype.

| Key       | Shape                   | Description                   |
|-----------|-------------------------|-------------------------------|
| `X_train` | `(n_train, n_features)` | Training features             |
| `y_train` | `(n_train, n_classes)`  | Training labels (one-hot)     |
| `X_test`  | `(n_test, n_features)`  | Test features                 |
| `y_test`  | `(n_test, n_classes)`   | Test labels (one-hot)         |
| `X_full`  | `(n_total, n_features)` | Full dataset features         |
| `y_full`  | `(n_total, n_classes)`  | Full dataset labels (one-hot) |

---

## FakeClient for Testing

```python
from juniper_data_client.testing import FakeDataClient

# Drop-in replacement -- no HTTP calls
with FakeDataClient() as client:
    result = client.create_spiral_dataset(n_spirals=2, seed=42)
    arrays = client.download_artifact_npz(result["dataset_id"])
    X_train = arrays["X_train"]  # numpy array, float32
```

Synthetic generators available via `juniper_data_client.testing`: `generate_spiral`, `generate_xor`, `generate_circle`, `generate_moon`.

> See: [docs/REFERENCE.md](REFERENCE.md#testing-utilities) for FakeDataClient constructor and synthetic generators.

---

## Error Handling

```bash
JuniperDataClientError (base)
+-- JuniperDataConnectionError     # Connection failed
+-- JuniperDataTimeoutError        # Request timed out
+-- JuniperDataNotFoundError       # 404
+-- JuniperDataValidationError     # 400/422 — use exc.status_code to tell them apart
+-- JuniperDataConfigurationError  # Missing/invalid config (hostless base_url)
+-- JuniperDataContractError       # NPZ contract (also ValueError)
```

Every exception carries `message`, `status_code`, `detail`, and `response`. Keyword-only extras; locally raised errors have `status_code=None`. FastAPI 422 keeps the list on `exc.detail`; the message renders `body.seed: Field required`.

### Error Recovery Pattern

```python
from juniper_data_client import (
    JuniperDataClient,
    JuniperDataConnectionError,
    JuniperDataTimeoutError,
)

client = JuniperDataClient("http://localhost:8100", retries=3)
try:
    result = client.create_dataset("spiral", {"n_points": 200})
except JuniperDataConnectionError:
    # Service unreachable -- retries already exhausted
    pass
except JuniperDataTimeoutError:
    # Request exceeded timeout -- consider increasing timeout param
    pass
```

> See: [docs/REFERENCE.md](REFERENCE.md#exception-hierarchy) for full HTTP status code mapping.

---

## Environment Variables

| Variable               | Default                 | Description                                             |
|------------------------|-------------------------|---------------------------------------------------------|
| `JUNIPER_DATA_API_KEY`      | *(unset)*               | API key fallback if `api_key=` and `_FILE` are unset        |
| `JUNIPER_DATA_API_KEY_FILE` | *(unset)*               | Path to a file whose stripped contents are the API key      |
| `JUNIPER_DATA_URL`          | `http://localhost:8100` | Used by consuming apps (juniper-cascor, juniper-canopy)     |

---

## Troubleshooting

| Symptom                                | Cause                      | Fix                                                             |
|----------------------------------------|----------------------------|-----------------------------------------------------------------|
| `JuniperDataConfigurationError` at init | Hostless `base_url`        | Pass a host (`localhost:8100` or `https://api.example.com`). Empty string, `http://`, `/v1`, and `http://user:secret@` all fail here. |
| `JuniperDataConnectionError`           | Service not running        | Start juniper-data: `make up` in juniper-deploy or run natively |
| `JuniperDataValidationError` on create | Invalid generator params   | Check `client.get_generator_schema(name)`; use `exc.status_code` (400 vs 422) and `exc.detail` |
| `JuniperDataNotFoundError` on download | Dataset ID expired/invalid | Re-create the dataset; artifacts may have been cleaned          |
| `JuniperDataContractError` after download | Sequence NPZ failed Δt/mask rules | Call `validate_npz_contract(arrays)` and inspect `str(exc)`     |
| Duplicate datasets after a 5xx POST    | Mutations are not auto-retried | Expected: only HEAD/GET/PUT retry. Use a dataset `name` for server-side dedupe. |
| NPZ arrays have wrong shape            | Generator params mismatch  | Verify `n_points`, `train_ratio` params                         |
| Auth failures (401/403)                | Missing or wrong API key   | Pass `api_key=`, or set `JUNIPER_DATA_API_KEY_FILE` / `JUNIPER_DATA_API_KEY` |

---

## Cross-References

- [juniper-data-client REFERENCE.md](REFERENCE.md) -- Full API reference
- [juniper-data-client QUICK_START.md](QUICK_START.md) -- Getting started guide
- [juniper-data-client AGENTS.md](../AGENTS.md) -- Agent development guide
- [Ecosystem Cheatsheet](https://github.com/pcalnon/juniper-ml/blob/main/docs/DEVELOPER_CHEATSHEET_JUNIPER-ML.md) -- Cross-project procedures
- [Data Contract](../CLAUDE.md#data-contract) -- NPZ artifact format specification
